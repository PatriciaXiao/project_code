import os
import sys
import tensorflow as tf
import numpy as np
import pandas as pd
import csv
from sklearn.metrics import roc_auc_score

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
                print "reached the end of the file {0}\n".format(filename)
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
    def reset(self):
        self.cursor = 0
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
            # s in sequence: s[0] - question id; s[1] - correctness
            Xs[i] = np.pad([2 + self.id_encoding[s[0]] + s[1] * self.vec_length for s in sequence[:-1]],
                (1, padding_length), 'constant', constant_values=(1,0))
            Ys[i] = np.pad([self.one_hot(self.id_encoding[s[0]],self.vec_length) for s in sequence], 
                ((0, padding_length), (0, 0)), 'constant', constant_values=0)
            targets[i] = np.pad([s[1] for s in sequence],
                (0, padding_length), 'constant', constant_values=0)
        return Xs, Ys, targets, len_sequences

class basicDKTModel:
    def __init__(
            self, 
            batch_size, 
            vec_length,                     # number of questions in dataset
            initial_learning_rate=0.001,
            final_learning_rate=0.00001,
            n_hidden=200,                   # number of hidden units in the hidden layer
            embedding_size=200,
            keep_prob=0.5,
            epsilon=0.001):
        # Rules
        assert keep_prob > 0 and keep_prob <= 1, "keep_prob parameter should be in (0, 1]"

        # Inputs: to be received from the outside
        Xs = tf.placeholder(tf.int32, shape=[batch_size, None], name='Xs_input')
        Ys = tf.placeholder(tf.float32, shape=[batch_size, None, vec_length], name='Ys_input')
        targets = tf.placeholder(tf.float32, shape=[batch_size, None], name='targets_input')
        sequence_length = tf.placeholder(tf.int32, shape=[batch_size], name='sequence_ength_input')

        # Global parameters initialized
        global_step = tf.Variable(0, trainable=False, name='global_step')
        learning_rate = tf.train.polynomial_decay(initial_learning_rate, global_step, 5000, final_learning_rate, name='learning_rate')

        # LSTM parameters initialized
        w = tf.Variable(tf.truncated_normal([n_hidden, vec_length], stddev=1.0/np.sqrt(vec_length)), name='Weight') # Weight
        b = tf.Variable(tf.truncated_normal([vec_length], stddev=1.0/np.sqrt(vec_length)), name='Bias') # Bias
        embeddings = tf.Variable(tf.random_uniform([2 * vec_length + 2, embedding_size], -1.0, 1.0), name='X_Embeddings')
        cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
        initial_state = cell.zero_state(batch_size, tf.float32)

        # LSTM Training options initialized
        inputsX = tf.nn.embedding_lookup(embeddings, Xs, name='Xs_embedded') # Xs embedded
        outputs, state = tf.nn.dynamic_rnn(cell, inputsX, sequence_length, initial_state=initial_state)
        if keep_prob != 1:
            outputs = tf.nn.dropout(outputs, keep_prob)
        outputs_flat = tf.reshape(outputs,shape=[-1, n_hidden])
        logits = tf.reshape(tf.nn.xw_plus_b(outputs_flat, w, b), shape=[batch_size,-1,vec_length])
        pred = tf.reduce_max(logits*Ys, axis=2)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=targets)
        mask = tf.sign(tf.abs(pred))
        loss_masked = mask*loss
        loss_masked_by_s = tf.reduce_sum(loss_masked, axis=1)
        mean_loss = tf.reduce_mean(loss_masked_by_s/tf.to_float(sequence_length))

        # Optimizer defined
        optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate, \
            epsilon=epsilon) \
            .minimize(mean_loss,global_step=global_step)
        
        # Saver defined
        saver = tf.train.Saver()

        # LSTM Validation options
        test_outputs, test_state = tf.nn.dynamic_rnn(cell,inputsX,sequence_length,initial_state)
        test_outputs_flat = tf.reshape(test_outputs, shape=[-1,n_hidden])
        test_logits = tf.reshape(tf.nn.xw_plus_b(test_outputs_flat,w,b),shape=[batch_size,-1,vec_length])
        test_pred = tf.sigmoid(tf.reduce_max(test_logits*Ys, axis=2))

        # assigning the attributes
        self._Xs = Xs
        self._Ys = Ys
        self._targets = targets
        self._seqlen = sequence_length
        self._loss = mean_loss
        self._train = optimizer
        self._saver = saver
        self._pred = test_pred

    @property
    def Xs(self):
        return self._Xs
    @property
    def Ys(self):
        return self._Ys
    @property
    def targets(self):
        return self._targets
    @property
    def seq_len(self):
        return self._seqlen
    @property
    def loss(self):
        return self._loss
    @property
    def train_op(self):
        return self._train
    @property
    def saver(self):
        return self._saver
    @property
    def predict(self):
        return self._pred


def run(session, 
        train_batchgen, test_batchgen, 
        option, n_epoch=0, n_step=0, 
        report_loss_interval=100, report_score_interval=500,
        model_saved_path='model.ckpt'):
    assert option in ['step', 'epoch'], "Run with either epochs or steps"
    steps_to_test = test_batchgen.data_size//train_batchgen.batch_size
    assert steps_to_test > 0, "Test set too small"
    def calc_score(m):
        auc = 0
        test_batchgen.reset()
        for i in range(steps_to_test):
            test_batch_Xs, test_batch_Ys, test_batch_labels, test_batch_sequence_lengths = test_batchgen.next_batch()
            test_feed_dict= {m.Xs : test_batch_Xs, m.Ys : test_batch_Ys, 
                        m.seq_len : test_batch_sequence_lengths}
            pred = session.run([m.predict], feed_dict=test_feed_dict)
            auc += roc_auc_score(test_batch_labels.reshape(-1),np.array(pred).reshape(-1))/50
        return auc
    m = basicDKTModel(train_batchgen.batch_size, train_batchgen.vec_length)
    with session.as_default():
        tf.global_variables_initializer().run()
        if option == 'step':
            sum_loss = 0
            for step in range(n_step):
                batch_Xs, batch_Ys, batch_labels, batch_sequence_lengths = train_batchgen.next_batch()
                feed_dict = {m.Xs : batch_Xs, m.Ys : batch_Ys, 
                        m.seq_len : batch_sequence_lengths, m.targets : batch_labels}
                _, batch_loss = session.run([m.train_op,m.loss], feed_dict=feed_dict)
                sum_loss += batch_loss
                if step % report_loss_interval == 0:
                    average_loss = sum_loss / min(report_loss_interval, step+1)
                    print ('Average loss at step (%d/%d): %f' % (step, n_step, average_loss))
                    sum_loss = 0
                if step % report_score_interval == 0:
                    auc = calc_score(m)
                    print('AUC score: {}'.format(auc))   
                    save_path = m.saver.save(session, model_saved_path)
                    print('Model saved in {}'.format(save_path))
        elif option == 'epoch':
            steps_per_epoch = train_batchgen.data_size//train_batchgen.batch_size
            for epoch in range(n_epoch):
                print ('Start epoch (%d/%d)' % (epoch, n_epoch))
                sum_loss = 0
                for step in range(steps_per_epoch):
                    batch_Xs, batch_Ys, batch_labels, batch_sequence_lengths = train_batchgen.next_batch()
                    feed_dict = {m.Xs : batch_Xs, m.Ys : batch_Ys, 
                            m.seq_len : batch_sequence_lengths, m.targets : batch_labels}
                    _, batch_loss = session.run([m.train_op,m.loss], feed_dict=feed_dict)
                    sum_loss += batch_loss
                    if step % report_loss_interval == 0:
                        average_loss = sum_loss / min(report_loss_interval, step+1)
                        print ('Average loss at step (%d/%d): %f' % (step, steps_per_epoch, average_loss))
                        sum_loss = 0
                    if step % report_score_interval == 0:
                        auc = calc_score(m)
                        print('AUC score: {}'.format(auc))   
                        save_path = m.saver.save(session, model_saved_path)
                        print('Model saved in {}'.format(save_path))
                print ('End epoch (%d/%d)' % (epoch, n_epoch))
                auc = calc_score(m)
                print('AUC score: {}'.format(auc))   
                save_path = m.saver.save(session, model_saved_path)
                print('Model saved in {}'.format(save_path))
    pass

batch_size = 16
n_epoch = 5
n_step = 1001

PrepData = IO()
# response_list, question_list = PrepData.load_model_input('datatest.csv', sep=',')
# batches = BatchGenerator(response_list, batch_size, id_encoding)
# Xs, Ys, targets, len_sequences = batches.next_batch()
# print Xs, Ys, targets, len_sequences
train_response_list, question_list = PrepData.load_model_input('0910_c_train.csv', sep=',')
test_response_list, question_list = PrepData.load_model_input('0910_c_test.csv', sep=',', question_list=question_list)
id_encoding = PrepData.question_id_1hotencoding(question_list)

train_batches = BatchGenerator(train_response_list, batch_size, id_encoding)
test_batches = BatchGenerator(test_response_list, batch_size, id_encoding)

sess = tf.Session()
run(sess, train_batches, test_batches, option='step', n_step=n_step)
# run(sess, train_batches, test_batches, option='epoch', n_epoch=n_epoch)
# tensorboard --logdir logs
writer = tf.summary.FileWriter("logs/", sess.graph) # http://localhost:6006/#graphs on mac
sess.close()