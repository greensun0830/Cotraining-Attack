import tensorflow as tf
from tensorflow.contrib import slim
import os
import numpy as np
import scipy.sparse as sp
import math
import time
from time import strftime
from time import localtime
import utils

flags = tf.flags
FLAGS = flags.FLAGS


class SVD:
    def __init__(self, dataset):
        self.num_users = dataset.num_users
        self.origin_num_users = dataset.origin_num_users
        self.num_items = dataset.num_items
        self.num_factors = FLAGS.embed_size
        self.dataset = dataset
        self.train_matrix = dataset.trainMatrix.toarray()
        self.reg = 1e-12
        self.coo_mx = self.dataset.trainMatrix.tocoo()
        self.mask = self.dataset.trainMatrix.toarray() != 0

    def create_placeholders(self):
        with tf.variable_scope('placeholder'):
            self.users_holder = tf.placeholder(tf.int32, shape=[None, 1], name='users')
            self.items_holder = tf.placeholder(tf.int32, shape=[None, 1], name='items')
            self.ratings_holder = tf.placeholder(tf.float32, shape=[None, 1], name='ratings')

    def create_model(self, i):
        num_users = self.num_users
        num_items = self.num_items
        num_factors = self.num_factors

        w_init = slim.xavier_initializer
        self.user_embeddings = tf.get_variable(shape=[num_users, num_factors],
                                          initializer=w_init(), regularizer=slim.l2_regularizer(self.reg),
                                          name="user_%d" % i)
        p_u = tf.reduce_sum(tf.nn.embedding_lookup(
            self.user_embeddings,
            self.users_holder), axis=1)
        self.item_embeddings = tf.get_variable(shape=[num_items, num_factors],
                                          initializer=w_init(), regularizer=slim.l2_regularizer(self.reg),
                                          name="item_%d" % i)
        q_i = tf.reduce_sum(tf.nn.embedding_lookup(
            self.item_embeddings,
            self.items_holder), axis=1)
        pred = tf.expand_dims(tf.reduce_sum(tf.multiply(p_u, q_i), axis=1), axis=-1)
        loss = tf.reduce_mean(tf.pow(self.ratings_holder - pred, 2))
        loss = tf.add(loss,
                      (tf.reduce_mean(p_u * p_u) + tf.reduce_mean(q_i * q_i)) * self.reg)
        self.rate = tf.matmul(self.user_embeddings, tf.transpose(self.item_embeddings))
        self.rate_attack = tf.sigmoid(
            tf.matmul(self.user_embeddings[:self.dataset.origin_num_users], tf.transpose(self.item_embeddings)))
        value = tf.sort(-self.rate_attack, axis=-1)[:, FLAGS.top_k - 1:FLAGS.top_k]
        attack_loss = tf.reduce_sum(tf.add_n(
            [tf.reduce_sum(tf.maximum(value - self.rate_attack[:, t:t + 1], -0.2))
             for t in range(len(FLAGS.target_item))]))

        self.optimizer = tf.train.AdadeltaOptimizer(20.)
        train_op = self.optimizer.minimize(loss)
        train_op1 = self.optimizer.minimize(loss + attack_loss, name='optimizer1')
        return self.rate, train_op, self.rate_attack, train_op1

    def build_graph(self):
        self.create_placeholders()

    def train(self, dataset, attack_size, fillter_size, nb_epochs=40):
        batch_size = 2048
        if (FLAGS.dataset == 'filmtrust'):
            batch_size = 4096
        # select_p = np.ones(dataset.num_items)
        rate_list = []
        attack_rate_list = []
        model_list = []
        attack_model_list = []
        for i in range(3):
            rate, opt, attack_rate, opt1 = self.create_model(i)
            rate_list.append(rate)
            attack_rate_list.append(attack_rate)
            model_list.append(opt)
            attack_model_list.append(opt1)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        flag = float('inf')

        train_matrix = dataset.trainMatrix[:self.dataset.origin_num_users]

        sample = utils.sampling1(train_matrix.tocoo())
        for cur_epochs in range(nb_epochs):
            for j in range(3):
                if (cur_epochs > FLAGS.pre):
                    fake_rate = self.get_label(rate_list[(j + 1) % 3], rate_list[(j + 2) % 3])
                    cur_sample = self.extend_sample(sample, fake_rate)
                else:
                    cur_sample = sample
                batchs = utils.get_batchs(cur_sample, FLAGS.batch_size)
                batchs_origin=utils.get_batchs(sample,FLAGS.batch_size)
                if(cur_epochs < nb_epochs//2):
                    for i in range(len(batchs)):
                        users, items, rates = batchs[i]
                        feed_dict = {self.users_holder: users,
                                    self.items_holder: items,
                                    self.ratings_holder: rates}
                        self.sess.run([model_list[j]], feed_dict)
                else:
                    for i in range(len(batchs_origin)):
                        users, items, rates = batchs_origin[i]
                        feed_dict = {self.users_holder: users,
                                    self.items_holder: items,
                                    self.ratings_holder: rates}
                        self.sess.run([attack_model_list[j]], feed_dict)
            rate = self.sess.run(rate_list[0])
            hr, hr1, ndcg, rmse = utils.train_evalute(rate, dataset, cur_epochs)
            if rmse < flag:
                best_hr, best_hr1 = hr, hr1
                flag = rmse
        rate = self.sess.run(
            tf.sigmoid(
                tf.matmul(self.user_embeddings[self.dataset.origin_num_users:self.dataset.origin_num_users + attack_size], tf.transpose(self.item_embeddings))))
        m_rate = rate
        index_all = np.argsort(-m_rate, axis=-1)[:,:fillter_size]
        for i in range(len(index_all)):
            index=index_all[i]
            # select_p[index] *= 0.9
            cur_user = np.zeros((1, dataset.num_items))
            cur_user[0, index] = np.clip(np.round(m_rate[i] * dataset.max_rate) / dataset.max_rate, 0, 1)[index]

            cur_user[:, FLAGS.target_item] = 1.
            train_matrix = sp.vstack([train_matrix, sp.csr_matrix(cur_user)]).todok()
            # (cur_attack - self.dataset.origin_num_users, np.mean(cur_user > 0))
        return train_matrix[self.dataset.origin_num_users:].toarray()

    def get_label(self, rate1, rate2):
        pred1, pred2 = self.sess.run([rate1, rate2])
        pred1 = np.round(pred1 * self.dataset.max_rate) / self.dataset.max_rate
        pred2 = np.round(pred2 * self.dataset.max_rate) / self.dataset.max_rate
        # print(np.mean(pred1<0),np.mean(pred1>1))
        # rate_mask = (np.abs(pred1 - pred2)<0.01) * (1 - self.mask)
        rate_mask = (pred1 == pred2) * (1 - self.mask)
        # print(rate_mask)
        # pred11 = (pred1 > 0)*pred1
        # pred111 =(pred11 < 1)*pred11
        mask1 = np.random.binomial(1, FLAGS.mask_rate, self.dataset.trainMatrix.toarray().shape)
        rate = (pred1 + 1e-5) * rate_mask * mask1
        # print(rate)
        # rate = (pred1 + 1e-5) * (1 - self.mask)
        return rate

    def extend_sample(self, sample, fake_rate):
        temp = sp.coo_matrix(fake_rate)
        user_input = np.array(temp.row)[:, None]
        item_input = np.array(temp.col)[:, None]
        rate_input = np.array(temp.data)[:, None]
        # print(rate_input.shape,sample[0].shape)
        user_input = np.concatenate([sample[0], user_input], axis=0)
        item_input = np.concatenate([sample[1], item_input], axis=0)
        rate_input = np.concatenate([sample[2], rate_input], axis=0)
        return [user_input, item_input, rate_input]
