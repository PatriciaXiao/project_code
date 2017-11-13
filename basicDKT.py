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
        self.id_encoding = id_encoding
        self.vec_length = len(id_encoding)
        self.data_size = len(data)
        self.cursor = 0 # cursor of the current batch's starting index
    def one_hot(self, hot, size):
        vec = np.zeros(size)
        vec[hot] = 1.0
        return vec
    def next_batch(self):
        qa_sequences = []
        len_sequences = []
        max_sequence_len = 0
        for i in range(self.batch_size):
            tmp_sequence = self.data[self.cursor][1]
            tmp_sequence_len = len(tmp_sequence)
            qa_sequences.append(tmp_sequence)
            len_sequences.append(tmp_sequence_len)
            if tmp_sequence_len > max_sequence_len:
                max_sequence_len = tmp_sequence_len
            self.cursor = (self.cursor + 1) % self.data_size
        # initialize the Xs and Ys
        Xs = np.zeros((self.batch_size, max_sequence_len),dtype=np.int32)
        Ys = np.zeros((self.batch_size, max_sequence_len, self.vec_length), dtype=np.int32)
        targets = np.zeros((self.batch_size, max_sequence_len),dtype=np.int32)
        for i, sequence in enumerate(qa_sequences):
            padding_length = max_sequence_len - len(sequence)
            Xs[i] = np.pad([2 + self.id_encoding[s[0]] + s[1] * self.vec_length for s in sequence[:-1]],
                (1, padding_length), 'constant', constant_values=(1,0))
            Ys[i] = np.pad([self.one_hot(self.id_encoding[s[0]],self.vec_length) for s in sequence], 
                ((0, padding_length), (0, 0)), 'constant', constant_values=0)
            targets[i] = np.pad([s[1] for s in sequence],
                (0, padding_length), 'constant', constant_values=0)
        return Xs, Ys, targets, len_sequences

class basicDKTModel:
    def __init__(self, batch_size, vec_length):
        # Inputs
        Xs = self._Xs = tf.placeholder(tf.int32, shape=[batch_size, None])
        Ys = self._Ys = tf.placeholder(tf.float32, shape=[batch_size, None, vec_length])
        targets = self._targets = tf.placeholder(tf.float32, shape=[batch_size, None])
        sequence_length = self._seqlen = tf.placeholder(tf.int32, shape=[batch_size])

batch_size = 16

PrepData = IO()
response_list, question_list = PrepData.load_model_input('datatest.csv', sep=',')
# response_list, question_list = PrepData.load_model_input('0910_c_train.csv', sep=',')
id_encoding = PrepData.question_id_1hotencoding(question_list)
# sess = tf.Session()
# print response_list
batches = BatchGenerator(response_list, batch_size, id_encoding)
Xs, Ys, targets, len_sequences = batches.next_batch()
print Xs, Ys, targets, len_sequences