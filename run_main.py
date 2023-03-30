import warnings

warnings.filterwarnings("ignore")
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import time

from Dataset import Dataset
import utils
from models.pure_SVD import SVD as PureSVD
from models.co_SVD import SVD as COSVD
from models.adv_SVD import SVD as AdvSVD
from models.random_SVD import SVD as RandomSVD

flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("dataset", "filmtrust", "Choose a dataset.")
flags.DEFINE_string('path', 'Data/', 'Input data path.')
flags.DEFINE_string('gpu', '0', 'Input data path.')
flags.DEFINE_integer('verbose', 1, 'Evaluate per X epochs.')
flags.DEFINE_integer('batch_size',2048 , 'batch_size')
flags.DEFINE_integer('epochs',40, 'Number of epochs.')
flags.DEFINE_integer('embed_size', 128, 'Embedding size.')
flags.DEFINE_integer('dns', 0, 'number of negative sample for each positive in dns.')
flags.DEFINE_integer('per_epochs', 1, 'number of negative sample for each positive in dns.')
flags.DEFINE_float('reg', 1e-4, 'Regularization for user and item embeddings.')
flags.DEFINE_float('lr', 0.05, 'Learning rate.')
flags.DEFINE_bool('reg_data', True, 'Regularization for adversarial loss')
flags.DEFINE_string('rs', 'svd', 'recommender system')
flags.DEFINE_bool("is_train", True, "train online or load model")
flags.DEFINE_bool("attack_load", False, "train online or load model")
flags.DEFINE_bool("use_second", False, "train online or load model")
flags.DEFINE_integer("top_k", 40, "pass")
flags.DEFINE_list("target_item", [1679], "pass")
flags.DEFINE_string('pretrain', '0', 'ckpt path')
flags.DEFINE_string('512', 'a', 'ckpt path')
flags.DEFINE_float("attack_size", 0.03, "pass")
flags.DEFINE_string("attack_type", "TNA", "attack type")
flags.DEFINE_float("data_size", 1., "pass")
flags.DEFINE_integer('target_index', 4, 'Embedding size.')
flags.DEFINE_float('extend', 0.05, 'adversarial training size.')
flags.DEFINE_float('mask_rate', 1, 'adversarial training size.')
flags.DEFINE_integer('pre', 0, 'Evaluate per X epochs.')
os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu


def get_rs(rs, dataset, extend):
    if (rs == 'puresvd'):
        rs = PureSVD(dataset.num_users, dataset.num_items, dataset)
    elif(rs == 'advsvd'):
        rs = AdvSVD(dataset.num_users, dataset.num_items, dataset)
    elif(rs == 'randomsvd'):
        rs = RandomSVD(dataset.num_users, dataset.num_items, dataset)
    elif (rs == 'cosvd'):
        rs = COSVD(dataset.num_users, dataset.num_items, dataset)
    else:
        print("error")
        exit(0)
    return rs


if __name__ == '__main__':
    extend = 0

    a = [[1485, 1320, 821, 1562, 1531],
         [1018, 946, 597, 575, 516],
         [3639, 3698, 3622, 3570, 3503],
         [1032, 3033, 2797, 2060, 1366],
         [1576, 926, 942, 848, 107],
         [539, 117, 1600, 1326, 208],
         [2504, 19779, 9624, 24064, 17390],
         [2417, 21817, 13064, 3348, 15085]]
    FLAGS.target_item = a[FLAGS.target_index]

    import time

    cur_time = time.strftime("%Y-%m-%d", time.localtime())
    dataset = Dataset(FLAGS.path + FLAGS.dataset, FLAGS.reg_data)
    # for attack in ['average', 'random', 'aush']:
    # for attack in ['average', 'random', 'aush', 'PGA', 'TNA', 'mixrand', 'mixinf']:
    for attack in ['average', 'random', 'aush', 'PGA', 'TNA']:
    # for attack in [FLAGS.attack_type]:
    # for attack in ['average']:
    # for attack in ['mixinf']:
        # initialize dataset
        dataset = Dataset(FLAGS.path + FLAGS.dataset, FLAGS.reg_data)
        num_users=dataset.num_users

        hr_list = []
        test_hr_list = []
        ndcg_list = []
        test_ndcg_list = []
        ps_list = []
        rank_list = []

        FLAGS.attack_type = attack
        t_epochs = 1
        t1, t2, t3, t4, t5 = [], [], [], [], []
        for i in range(t_epochs):
            # print("Epoch: {}".format(i))
            RS = get_rs("puresvd", dataset, extend)
            tf.reset_default_graph()
            RS.build_graph()
            # print("Initialize %s" % FLAGS.rs)
        #
        #     # start training
            test_hr, test_hr1,test_ndcg,best_hr, best_hr1 = RS.train(dataset, FLAGS.is_train, FLAGS.epochs, np.ones(dataset.num_users), False)
        #     # target item recommendation
            # print("origin: target item: ", FLAGS.target_item)
            hr, ndcg, ps, rank = utils.recommend(RS, dataset, FLAGS.target_item, FLAGS.top_k)
            # print("recommend all user: HR-{}, HR1-{}, NDCG-{}".format(test_hr, test_hr1, ndcg))
            print("best: HR-{}, HR1-{}".format(best_hr, best_hr1))
            t1.append(hr)
            t2.append(ndcg)
            t3.append(ps)
            t4.append(test_hr)
            t5.append(test_ndcg)
            rank_list.append(rank)
        hr_list.append(np.mean(t1))
        test_hr_list.append(np.mean(t4))
        ndcg_list.append(np.mean(t2))
        ps_list.append(np.mean(t3))
        test_ndcg_list.append(np.mean(t5))
        np.save("temp/hr_%s_%s_%d_origin.npy"%(FLAGS.dataset,FLAGS.attack_type,FLAGS.target_index),np.array(t1))
        np.save("temp/test_hr_%s_%s_%d_origin.npy" % (FLAGS.dataset, FLAGS.attack_type, FLAGS.target_index),
                np.array(t4))
        
        # attack
        attack_size = int(dataset.full_num_users * FLAGS.attack_size)
        poison_user = np.load("./temp/%s/full/%s_poisoning_%d_%d_0.400000.npy" % (
            FLAGS.dataset, FLAGS.attack_type, a[FLAGS.target_index][0], attack_size))
        temp_user = np.mean(dataset.trainMatrix.toarray(), axis=0, keepdims=True)
        temp_user = np.round(temp_user * dataset.max_rate) / dataset.max_rate
        dataset = utils.estimate_dataset(dataset, poison_user)
        # print("the users after attack:", dataset.num_users)

# after poisoning
        extend = int(num_users * FLAGS.extend)
        t1, t2, t3, t4, t5 = [], [], [], [], []
        for i in range(t_epochs):
            # print("cur ", i)
            RS = get_rs("puresvd", dataset, extend)
            tf.reset_default_graph()
            RS.build_graph()
            test_hr, test_hr1,test_ndcg,best_hr, best_hr1 = RS.train(dataset, FLAGS.is_train, FLAGS.epochs, np.ones(dataset.num_users), False)
        #     # target item recommendation
            # print("origin: target item: ", FLAGS.target_item)
            hr, ndcg, ps, rank = utils.recommend(RS, dataset, FLAGS.target_item, FLAGS.top_k)
            # print("recommend all user: HR-{}, HR1-{}, NDCG-{}".format(test_hr, test_hr1, ndcg))
            print("best: HR-{}, HR1-{}".format(best_hr, best_hr1))

            t1.append(hr)
            t2.append(ndcg)
            t3.append(ps)
            t4.append(test_hr)
            t5.append(test_ndcg)
            rank_list.append(rank)
        hr_list.append(np.mean(t1))
        test_hr_list.append(np.mean(t4))
        ndcg_list.append(np.mean(t2))
        ps_list.append(np.mean(t3))
        test_ndcg_list.append(np.mean(t5))

# after poisoning
        extend = int(num_users * FLAGS.extend)
        t1, t2, t3, t4, t5 = [], [], [], [], []
        for i in range(t_epochs):
            # print("cur ", i)
            RS = get_rs("advsvd", dataset, extend)
            tf.reset_default_graph()
            RS.build_graph()
            test_hr, test_hr1,test_ndcg,best_hr, best_hr1 = RS.train(dataset, FLAGS.is_train, FLAGS.epochs, np.ones(dataset.num_users), False)
        #     # target item recommendation
            # print("origin: target item: ", FLAGS.target_item)
            hr, ndcg, ps, rank = utils.recommend(RS, dataset, FLAGS.target_item, FLAGS.top_k)
            # print("recommend all user: HR-{}, HR1-{}, NDCG-{}".format(test_hr, test_hr1, ndcg))
            print("best: HR-{}, HR1-{}".format(best_hr, best_hr1))

            t1.append(hr)
            t2.append(ndcg)
            t3.append(ps)
            t4.append(test_hr)
            t5.append(test_ndcg)
            rank_list.append(rank)
        hr_list.append(np.mean(t1))
        test_hr_list.append(np.mean(t4))
        ndcg_list.append(np.mean(t2))
        ps_list.append(np.mean(t3))
        test_ndcg_list.append(np.mean(t5))

# after poisoning
        extend = int(num_users * FLAGS.extend)
        t1, t2, t3, t4, t5 = [], [], [], [], []
        for i in range(t_epochs):
            # print("cur ", i)
            RS = get_rs("randomsvd", dataset, extend)
            tf.reset_default_graph()
            RS.build_graph()
            test_hr, test_hr1,test_ndcg,best_hr, best_hr1 = RS.train(dataset, FLAGS.is_train, FLAGS.epochs, np.ones(dataset.num_users), False)
        #     # target item recommendation
            # print("origin: target item: ", FLAGS.target_item)
            hr, ndcg, ps, rank = utils.recommend(RS, dataset, FLAGS.target_item, FLAGS.top_k)
            # print("recommend all user: HR-{}, HR1-{}, NDCG-{}".format(test_hr, test_hr1, ndcg))
            print("best: HR-{}, HR1-{}".format(best_hr, best_hr1))

            t1.append(hr)
            t2.append(ndcg)
            t3.append(ps)
            t4.append(test_hr)
            t5.append(test_ndcg)
            rank_list.append(rank)
        hr_list.append(np.mean(t1))
        test_hr_list.append(np.mean(t4))
        ndcg_list.append(np.mean(t2))
        ps_list.append(np.mean(t3))
        test_ndcg_list.append(np.mean(t5))

        # after poisoning
# after poisoning
        extend = int(num_users * FLAGS.extend)
        t1, t2, t3, t4, t5 = [], [], [], [], []
        for i in range(t_epochs):
            # print("cur ", i)
            RS = get_rs("cosvd", dataset, extend)
            tf.reset_default_graph()
            RS.build_graph()
            test_hr, test_hr1,test_ndcg,best_hr, best_hr1 = RS.train(dataset, FLAGS.is_train, FLAGS.epochs, np.ones(dataset.num_users), False)
        #     # target item recommendation
            # print("origin: target item: ", FLAGS.target_item)
            hr, ndcg, ps, rank = utils.recommend(RS, dataset, FLAGS.target_item, FLAGS.top_k)
            # print("recommend all user: HR-{}, HR1-{}, NDCG-{}".format(test_hr, test_hr1, ndcg))
            print("best: HR-{}, HR1-{}".format(best_hr, best_hr1))

            t1.append(hr)
            t2.append(ndcg)
            t3.append(ps)
            t4.append(test_hr)
            t5.append(test_ndcg)
            rank_list.append(rank)
        hr_list.append(np.mean(t1))
        test_hr_list.append(np.mean(t4))
        ndcg_list.append(np.mean(t2))
        ps_list.append(np.mean(t3))
        test_ndcg_list.append(np.mean(t5))


