import os
import sys
import tensorflow as tf
import numpy as np
import pandas as pd
import csv

SILENT_WARNINGS = False # to silent the warnings (https://github.com/tensorflow/tensorflow/issues/8037)

if SILENT_WARNINGS:
    os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

class MathFunc:
    # creating one-hot vectors
    def one_hot(self, hot, size):
        vec = np.zeros(size)
        vec[hot] = 1.0
        return vec

class IO:
    class CSVReader:
        def __init__(self, filename, delimiter='\t'):
            self.csvfile = open(filename, "r")
            self.csvreader = csv.reader(self.csvfile, delimiter=delimiter)
        def __del__(self):
            self.csvfile.close()
        def read_next_line(self):
            next_line = None
            while not next_line:
                try:
                    next_line = next(self.csvreader)
                except StopIteration:
                    raise StopIteration
                    break
            return next_line
    def load_model_input(self, filename, question_list=[], sep='\t'):
        # question_list = []
        response_list = []
        csvreader = self.CSVReader(filename, delimiter=sep)
        while True:
            try:
                seq_length_line = csvreader.read_next_line()
                seq_questionsID = csvreader.read_next_line()
                seq_correctness = csvreader.read_next_line()
                seq_length = int(seq_length_line[0])
                assert len(seq_length_line) == 1 and seq_length == len(seq_questionsID) and seq_length == len(seq_correctness), \
                    "Unexpected format of input CSV file\n"
                if seq_length > 1: # only when there are at least two questions together is the sequence meaningful
                    question_list += [question for question in set(seq_questionsID) if question not in question_list]
                    response_list.append((seq_length, list(zip(map(int, seq_questionsID), map(int, seq_correctness)))))
            except StopIteration:
                print "reached the end of the file\n"
                break
        del csvreader
        return response_list, question_list
    def question_id_1hotencoding(self, question_list):
        id_encoding = { int(j): int(i) for i, j in enumerate(question_list)}
        return id_encoding

class BatchGenerator:
    def __init__(self, data, batch_size, id_encoding):
        self.data = sorted(data, key = lambda x: x[0])
        self.batch_size = batch_size
    def one_hot(self, hot, size):
        vec = np.zeros(size)
        vec[hot] = 1.0
        return vec

# tuples, id2idx = read_data_from_csv_file('0910_c_test.csv')
# print tuples
# print id2idx

batch_size = 16

PrepData = IO()
response_list, question_list = PrepData.load_model_input('datatest.csv', sep=',')
id_encoding = PrepData.question_id_1hotencoding(question_list)
# sess = tf.Session()
# print response_list
batches = BatchGenerator(response_list, batch_size, id_encoding)
