import os
import sys
import numpy as np
import tensorflow as tf

from data_io import IO
from model import grainedDKTModel, BatchGenerator, run

SILENT_WARNINGS = True # to silent the warnings (https://github.com/tensorflow/tensorflow/issues/8037)

if SILENT_WARNINGS:
    os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

batch_size = 16
n_epoch = 5
n_step = 1001
keep_prob = 0.5
n_hidden_units = 200
random_embedding = False
multi_granined = True

PrepData = IO()

train_response_list, question_list = PrepData.load_model_input('../data/training_ASSISTment_all.csv', sep=',')
test_response_list, question_list = PrepData.load_model_input('../data/testing_ASSISTment_all.csv', sep=',', question_list=question_list)
id_encoding = PrepData.question_id_1hotencoding(question_list)
category_map_dict = PrepData.load_category_map('../data/skill-category(UTF-8).csv', sep=',')
category_encoding = PrepData.category_id_1hotencoding(category_map_dict)

skill2category_map =PrepData.skill_idx_2_category_idx(category_map_dict, category_encoding)
n_id = len(id_encoding)
n_categories = len(category_encoding)

# print {skill: category_encoding[category_map_dict[skill]] for skill in category_map_dict.keys()}
# print skill2category_map

train_batches = BatchGenerator(train_response_list, batch_size, id_encoding, n_id, n_id, random_embedding=random_embedding, skill_to_category_dict=skill2category_map)
test_batches = BatchGenerator(test_response_list, batch_size, id_encoding, n_id, n_id, random_embedding=random_embedding, skill_to_category_dict=skill2category_map)

sess = tf.Session()
run(sess, train_batches, test_batches, \
    option='step', n_step=n_step, random_embedding=random_embedding, multi_granined=multi_granined, \
    n_categories=n_categories, out_folder='../data/', out_file='results.csv', \
    keep_prob=keep_prob, n_hidden_units=n_hidden_units)
# run(sess, train_batches, test_batches, option='epoch', n_epoch=n_epoch)
# tensorboard --logdir multilayer_logs
writer = tf.summary.FileWriter("../multilayer_logs/", sess.graph) # http://localhost:6006/#graphs on mac
sess.close()
