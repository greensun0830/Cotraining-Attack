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
    def __init__(self, num_users, num_items, dataset):
        self.num_users = num_users
        self.num_items = num_items
        self.num_factors = FLAGS.embed_size
        self.reg = FLAGS.reg
        self.dataset = dataset
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
        user_embeddings = tf.get_variable(shape=[num_users, num_factors],
                                          initializer=w_init(), regularizer=slim.l2_regularizer(self.reg),
                                          name="user_%d" % i)
        p_u = tf.reduce_sum(tf.nn.embedding_lookup(
            user_embeddings,
            self.users_holder), axis=1)
        item_embeddings = tf.get_variable(shape=[num_items, num_factors],
                                          initializer=w_init(), regularizer=slim.l2_regularizer(self.reg),name="item_%d" % i)
        q_i = tf.reduce_sum(tf.nn.embedding_lookup(
            item_embeddings,
            self.items_holder), axis=1)
        pred = tf.expand_dims(tf.reduce_sum(tf.multiply(p_u, q_i), axis=1), axis=-1)
        loss = tf.reduce_mean(tf.pow(self.ratings_holder - pred, 2))
        loss = tf.add(loss,
                      (tf.reduce_mean(p_u * p_u) + tf.reduce_mean(q_i * q_i)) * self.reg)
        self.rate = tf.matmul(user_embeddings, tf.transpose(item_embeddings))
        self.optimizer = tf.train.AdadeltaOptimizer(20.)
        # self.optimizer = tf.train.AdamOptimizer(0.001)
        # if(FLAGS.dataset == 'yelp'):
        #     self.optimizer = tf.train.AdagradOptimizer(15.)
        train_op = self.optimizer.minimize(loss)
        return self.rate, train_op

    def build_graph(self):
        self.create_placeholders()

    def train(self, dataset, is_train, nb_epochs, weight1, use_weight=True):
        rate_list = []
        model_list = []
        for i in range(3):
            rate, opt = self.create_model(i)
            rate_list.append(rate)
            model_list.append(opt)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

        sample = utils.sampling(dataset, 0)
        flag = float('inf')
        for cur_epochs in range(nb_epochs):
            for j in range(3):
                if (cur_epochs > FLAGS.pre):
                    fake_rate = self.get_label(rate_list[(j + 1) % 3], rate_list[(j + 2) % 3])
                    cur_sample = self.extend_sample(sample, fake_rate)
                else:
                    cur_sample = sample
                batchs = utils.get_batchs(cur_sample, FLAGS.batch_size)
                for i in range(len(batchs)):
                    users, items, rates = batchs[i]
                    feed_dict = {self.users_holder: users,
                                 self.items_holder: items,
                                 self.ratings_holder: rates}
                    self.sess.run([model_list[j]], feed_dict)
            rate = self.sess.run(rate_list[0])
            hr, hr1,ndcg,rmse = utils.train_evalute(rate, dataset, cur_epochs)
            if rmse < flag:
                    best_hr, best_hr1 = hr,hr1
                    flag = rmse
        return hr,hr1, ndcg,best_hr, best_hr1
        
    def get_label(self, rate1, rate2):
        pred1, pred2 = self.sess.run([rate1, rate2])
        pred1 = np.round(pred1 * self.dataset.max_rate)/self.dataset.max_rate
        pred2 = np.round(pred2 * self.dataset.max_rate)/self.dataset.max_rate
        # print(np.mean(pred1<0),np.mean(pred1>1))
        # rate_mask = (np.abs(pred1 - pred2)<0.01) * (1 - self.mask)
        rate_mask = (pred1 == pred2) * (1 - self.mask)
        # print(rate_mask)
        # pred11 = (pred1 > 0)*pred1
        # pred111 =(pred11 < 1)*pred11
        mask1 = np.random.binomial(1,FLAGS.mask_rate,self.dataset.trainMatrix.toarray().shape)
        rate = (pred1 + 1e-5) * rate_mask*mask1
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
