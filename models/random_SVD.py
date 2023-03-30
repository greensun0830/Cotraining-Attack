import tensorflow as tf
from tensorflow.contrib import slim
import os
import numpy as np
import math
import time
from time import strftime
from time import localtime
import utils
import copy

flags = tf.flags
FLAGS = flags.FLAGS


class SVD:
    def __init__(self, num_users, num_items, dataset):
        self.num_users = num_users
        self.num_items = num_items
        self.num_factors = FLAGS.embed_size
        self.reg = FLAGS.reg
        self.dataset = dataset
        self.train_user = self.dataset.trainMatrix.toarray()
        self.train_mask = (self.train_user != 0).astype(np.float)
        self.coo_mx = self.dataset.trainMatrix.tocoo()
        self.mu_np = np.mean(self.coo_mx.data)
        self.eps = 0.05
        # self.adv = 'adv'
        self.adv = 'random'

    def create_placeholders(self):
        with tf.variable_scope('placeholder'):
            self.users_holder = tf.placeholder(tf.int32, shape=[None, 1], name='users')
            self.items_holder = tf.placeholder(tf.int32, shape=[None, 1], name='items')
            self.ratings_holder = tf.placeholder(tf.float32, shape=[None, 1], name='ratings')
            self.mask = tf.placeholder(tf.float32, name='mask')
            self.weight = tf.placeholder(tf.float32, name='weight')
            self.user_delta = tf.get_variable(
                name='user_delta',
                shape=[self.num_users, self.num_factors],
                initializer=slim.xavier_initializer(), regularizer=slim.l2_regularizer(self.reg), trainable=False)
            self.item_delta = tf.get_variable(
                name='item_delta',
                shape=[self.num_items, self.num_factors],
                initializer=slim.xavier_initializer(), regularizer=slim.l2_regularizer(self.reg), trainable=False)

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

    def create_prediction(self):
        with tf.variable_scope('prediction'):
            self.p_u_adv = tf.reduce_sum(tf.nn.embedding_lookup(self.user_delta, self.users_holder), axis=1)
            self.q_i_adv = tf.reduce_sum(tf.nn.embedding_lookup(self.item_delta, self.items_holder), axis=1)
            self.p_plus = self.p_u + self.p_u_adv
            self.q_plus = self.q_i + self.q_i_adv
            pred1 = tf.reduce_sum(tf.multiply(self.p_plus, self.q_plus), axis=1)
            self.pred1 = tf.expand_dims(pred1, axis=-1)
            pred2 = tf.reduce_sum(tf.multiply(self.p_u, self.q_i), axis=1)
            self.pred2 = tf.expand_dims(pred2, axis=-1)
            self.rate = tf.matmul(self.user_embeddings, tf.transpose(self.item_embeddings))
            self.rate_partial = tf.matmul(self.user_embeddings[:100], tf.transpose(self.item_embeddings))

    def create_adv(self):
        if (self.adv == 'random'):
            self.user_adv = tf.truncated_normal(shape=[self.num_users, self.num_factors], mean=0.0, stddev=0.01)
            self.item_adv = tf.truncated_normal(shape=[self.num_items, self.num_factors], mean=0.0, stddev=0.01)
            self.update_user = self.user_delta.assign(tf.nn.l2_normalize(self.user_adv, 1) * self.eps)
            self.update_item = self.item_delta.assign(tf.nn.l2_normalize(self.item_adv, 1) * self.eps)
        else:
            self.user_grad, self.item_grad = tf.gradients(self.loss1, [self.user_embeddings, self.item_embeddings])
            self.user_adv = tf.stop_gradient(self.user_grad)
            self.item_adv = tf.stop_gradient(self.item_grad)
            self.update_user = self.user_delta.assign(tf.nn.l2_normalize(self.user_adv, 1) * self.eps)
            self.update_item = self.item_delta.assign(tf.nn.l2_normalize(self.item_adv, 1) * self.eps)

    def create_optimizer(self):
        with tf.variable_scope('loss'):
            loss = tf.reduce_mean(tf.pow(self.ratings_holder - self.pred2, 2))
            self.loss1 = loss + (tf.reduce_mean(self.p_u * self.p_u) + tf.reduce_mean(self.q_i * self.q_i)) * self.reg
            self.loss = loss + (tf.reduce_mean(self.p_u * self.p_u) + tf.reduce_mean(self.q_i * self.q_i)) * self.reg
            self.loss += tf.reduce_mean(tf.pow(self.ratings_holder - self.pred1, 2)) * self.weight
            # self.loss += (tf.reduce_mean(self.p_plus * self.p_plus) + tf.reduce_mean(
            #     self.q_plus * self.q_plus)) * self.reg * self.weight
            # self.optimizer = tf.train.AdamOptimizer(0.1)
            self.optimizer = tf.train.AdadeltaOptimizer(20.)
            # self.optimizer = tf.train.AdamOptimizer(0.01)
            # self.optimizer = tf.train.AdamOptimizer(0.001)
        # if(FLAGS.dataset == 'yelp'):
        #     self.optimizer = tf.train.AdagradOptimizer(15.)
            self.train_op = self.optimizer.minimize(self.loss, name='optimizer')

    def build_graph(self):
        self.create_placeholders()
        self.create_user_terms()
        self.create_item_terms()
        self.create_prediction()
        self.create_optimizer()
        self.create_adv()

    def train(self, dataset, is_train, nb_epochs, weight1, use_weight=True):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        ckpt_save_path = "pretrain/%s/%s/embed_%d/model_%s_%s_%d" % (
            FLAGS.dataset, FLAGS.rs, FLAGS.embed_size, FLAGS.attack_type, FLAGS.gpu, FLAGS.target_item[0])
        if (not os.path.exists(ckpt_save_path)):
            os.makedirs(ckpt_save_path)

        saver_ckpt = tf.train.Saver()

        # pretrain or not
        self.sess.run(tf.global_variables_initializer())

        if (is_train == False):
            print("restore")
            saver_ckpt.restore(self.sess, ckpt_save_path)

        # initialize test data for Evaluate
        test_data = utils.prepare_test(self.dataset)
        samples = utils.sampling(dataset, 0)
        flag = float('inf')
        if (FLAGS.dataset == 'ml-100k'):
            pre_training = 5
        elif (FLAGS.dataset == 'filmtrust'):
            pre_training = 5
        elif (FLAGS.dataset == 'ml-1m'):
            pre_training = 5
        elif (FLAGS.dataset == 'yelp'):
            pre_training = 10
        # pre_training = 5
        weight = 0.
        for cur_epochs in range(nb_epochs):
            if (cur_epochs >= pre_training):
                weight = 1.
            batchs = utils.get_batchs(samples, FLAGS.batch_size)
            for i in range(len(batchs)):
                users, items, rates = batchs[i]
                feed_dict = {self.users_holder: users,
                             self.items_holder: items,
                             self.ratings_holder: rates,
                             self.weight: weight}
                self.sess.run([self.update_user, self.update_item], feed_dict)
                self.sess.run([self.train_op], feed_dict)

            if (cur_epochs % FLAGS.per_epochs == 0 or cur_epochs == nb_epochs - 1):
                if (FLAGS.dataset == 'yelp' and cur_epochs != nb_epochs - 1):
                    rate = self.sess.run(self.rate_partial)
                    # ttt=self.sess.run(self.user_embeddings)
                else:
                    rate = self.sess.run(self.rate)
                hr, hr1,ndcg,rmse = utils.train_evalute(rate, dataset, cur_epochs)
                if rmse < flag:
                    best_hr, best_hr1 = hr,hr1
                    flag = rmse
                # utils.save_model(ckpt_save_path, saver_ckpt, self.sess)
        # self.output_evaluate(self.sess, dataset, test_data, 0)
        return hr,hr1, ndcg,best_hr, best_hr1

    def output_evaluate(self, sess, dataset, eval_feed_dicts, epoch_count):
        result = self.evaluate(sess, eval_feed_dicts)
        hr, ndcg, auc = np.swapaxes(result, 0, 1)[-1]

        rate = self.sess.run(self.rate)
        user = dataset.trainMatrix.toarray()
        mask = user != 0
        rate[mask] = -np.inf
        count = 0
        for i in range(dataset.origin_num_users):
            idx = np.argsort(rate[i])[::-1][:10]
            count += (self.dataset.testRatings[i][1] in idx)

        res = "Epoch %d: HR = %.4f, NDCG = %.4f HR = %.4f" % (epoch_count, hr, ndcg, count / dataset.origin_num_users)
        print(res)
        return count / dataset.origin_num_users

    def evaluate(self, sess, feed_dicts):
        res = []
        for user in range(len(feed_dicts)):
            res.append(self.eval_by_user(user, feed_dicts, sess))
        res = np.array(res)
        hr, ndcg, auc = (res.mean(axis=0)).tolist()
        return hr, ndcg, auc

    def eval_by_user(self, user, feed_dicts, sess, _K=10):
        # get prredictions of data in testing set
        user_input, item_input = feed_dicts[user]
        feed_dict = {self.users_holder: user_input, self.items_holder: item_input}
        predictions = sess.run(self.pred2, feed_dict)

        neg_predict, pos_predict = predictions[:-1], predictions[-1]
        position = (neg_predict >= pos_predict).sum()

        # calculate from HR@1 to HR@100, and from NDCG@1 to NDCG@100, AUC
        hr, ndcg, auc = [], [], []
        for k in range(1, _K + 1):
            hr.append(position < k)
            ndcg.append(math.log(2) / math.log(position + 2) if position < k else 0)
            auc.append(
                1 - (position / len(neg_predict)))  # formula: [#(Xui>Xuj) / #(Items)] = [1 - #(Xui<=Xuj) / #(Items)]

        return hr, ndcg, auc

    def get_embeddings(self):
        results = self.sess.run([self.rate, self.user_embeddings, self.item_embeddings])
        return results
