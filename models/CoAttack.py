import tensorflow as tf
from tensorflow.contrib import slim
import os
import numpy as np
import math
import scipy.sparse as sp
import utils

flags = tf.flags
FLAGS = flags.FLAGS


class PMF:
    def __init__(self, dataset):
        self.num_users = dataset.num_users
        self.origin_num_users = dataset.origin_num_users
        self.num_items = dataset.num_items
        self.num_factors = FLAGS.embed_size
        self.reg = 1e-12
        self.dataset = dataset
        self.train_matrix = dataset.trainMatrix.toarray()

    def create_placeholders(self):
        with tf.variable_scope('placeholder'):
            self.users_holder = tf.placeholder(tf.int32, shape=[None, 1], name='users')
            self.items_holder = tf.placeholder(tf.int32, shape=[None, 1], name='items')
            self.ratings_holder = tf.placeholder(tf.float32, shape=[None, 1], name='ratings')

            self.users_holder_inf = tf.placeholder(tf.int32, shape=[None, 1], name='users')
            self.items_holder_inf = tf.placeholder(tf.int32, shape=[None, 1], name='items')

    def create_user_terms(self):
        num_users = self.num_users
        num_factors = self.num_factors

        w_init = slim.xavier_initializer
        with tf.variable_scope('user'):
            self.user_embeddings = tf.get_variable(
                name='embedding',
                shape=[num_users, num_factors],
                initializer=w_init(), regularizer=slim.l2_regularizer(self.reg))
            self.p_u = tf.reduce_sum(tf.nn.embedding_lookup(
                self.user_embeddings,
                self.users_holder,
                name='p_u'), axis=1)

            self.p_u_inf = tf.reduce_sum(tf.nn.embedding_lookup(
                self.user_embeddings,
                self.users_holder_inf,
                name='p_u_inf'), axis=1)

    def create_item_terms(self):
        num_items = self.num_items
        num_factors = self.num_factors

        w_init = slim.xavier_initializer
        with tf.variable_scope('item'):
            self.item_embeddings = tf.get_variable(
                name='embedding',
                shape=[num_items, num_factors],
                initializer=w_init(), regularizer=slim.l2_regularizer(self.reg))
            self.q_i = tf.reduce_sum(tf.nn.embedding_lookup(
                self.item_embeddings,
                self.items_holder,
                name='q_i'), axis=1)

            self.q_i_inf = tf.reduce_sum(tf.nn.embedding_lookup(
                self.item_embeddings,
                self.items_holder_inf,
                name='q_i_inf'), axis=1)

    def create_prediction(self):
        with tf.variable_scope('prediction'):
            pred = tf.sigmoid(tf.reduce_sum(tf.multiply(self.p_u, self.q_i), axis=1))
            self.pred = tf.expand_dims(pred, axis=-1)
            self.rate = tf.sigmoid(tf.matmul(self.user_embeddings, tf.transpose(self.item_embeddings)))
            self.rate_partial = tf.sigmoid(tf.matmul(self.user_embeddings[:100], tf.transpose(self.item_embeddings)))
            self.rate_attack = tf.sigmoid(tf.matmul(self.user_embeddings[:self.dataset.origin_num_users], tf.transpose(self.item_embeddings)))

    def create_optimizer(self):
        with tf.variable_scope('loss'):
            self.loss_rate = tf.reduce_mean(tf.pow(self.ratings_holder - self.pred, 2))
            self.weights = [self.p_u, self.q_i]
            reg_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.weights])
            self.loss = self.loss_rate + reg_loss * self.reg
            if (FLAGS.data_size == 0.01 and FLAGS.dataset == 'filmtrust'):
                self.optimizer = tf.train.AdagradOptimizer(1.)
            else:
                self.optimizer = tf.train.AdagradOptimizer(10.)

            value = tf.sort(-self.rate_attack, axis=-1)[:, FLAGS.top_k - 1:FLAGS.top_k]

            attack_loss = tf.reduce_sum(tf.add_n(
                [tf.reduce_sum(tf.maximum(value - self.rate_attack[:, t:t + 1], -0.2))
                 for t in range(len(FLAGS.target_item))]))

            self.train_op = self.optimizer.minimize(self.loss, name='optimizer')
            self.train_op1 = self.optimizer.minimize(self.loss + attack_loss * 1., name='optimizer1')

    def build_graph(self):
        self.create_placeholders()
        self.create_user_terms()
        self.create_item_terms()
        self.create_prediction()
        self.create_optimizer()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def train(self, dataset, attack_size, fillter_size, nb_epochs=40):
        batch_size = 2048
        if (FLAGS.dataset == 'filmtrust'):
            batch_size = 4096
        # select_p = np.ones(dataset.num_items)

        train_matrix = dataset.trainMatrix[:self.dataset.origin_num_users]
        import time
        samples = utils.sampling1(train_matrix.tocoo())
        for cur_epochs in range(nb_epochs):
            batchs = utils.get_batchs(samples, batch_size)
            for i in range(len(batchs)):
                users, items, rates = batchs[i]
                feed_dict = {self.users_holder: users,
                                self.items_holder: items,
                                self.ratings_holder: rates}
                if (cur_epochs < nb_epochs - 20):
                    self.sess.run([self.train_op], feed_dict)
                else:
                    self.sess.run([self.train_op1], feed_dict)

            if (cur_epochs % FLAGS.per_epochs == 0 or cur_epochs == nb_epochs - 1):
                if (FLAGS.dataset == 'yelp'):
                    rate = self.sess.run(self.rate_partial)
                else:
                    rate = self.sess.run(self.rate)
                hr, ndcg = utils.train_evalute1(rate, dataset, cur_epochs)
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

    def get_embeddings(self):
        results = self.sess.run([self.rate, self.user_embeddings, self.item_embeddings])
        return results

    def get_rate(self):
        return self.sess.run(self.rate)
