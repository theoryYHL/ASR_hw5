from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, MultiRNNCell, DropoutWrapper, LSTMStateTuple

######################################################
# Embedding metrix를 이용하기 위한 tensorflow의 layer
# 업로드 해주신 실습 자료 그대로 사용
######################################################
class Embed(object):

    def __init__(self, input_dim, output_dim, name,
                 w_initializer=tf.random_normal_initializer(0.0, 0.1)):
        self.input_dim = input_dim
        self.output_dim = output_dim

        scope_name = name + '_Embed'
        with tf.variable_scope(scope_name):
            self.W = tf.get_variable('W', [input_dim, output_dim], tf.float32, w_initializer)

    def __call__(self, input_):
        return tf.nn.embedding_lookup(self.W, input_)

    def get_params(self):
        return [self.W]

######################################################
# 마지막 softmax layer에 사용
# 업로드 해주신 실습 자료 그대로 사용
######################################################
class Dense(object):

    def __init__(self, input_dim, output_dim, name='',
                 w_initializer=tf.random_normal_initializer(0.0, 0.02),
                 b_initializer=tf.zeros_initializer(),
                 use_bias=True):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias

        scope_name = name + '_Dense'
        with tf.variable_scope(scope_name):
            self.W = tf.get_variable('W', [input_dim, output_dim], tf.float32, w_initializer)
            self.b = tf.get_variable('b', [output_dim], tf.float32, b_initializer)

    def __call__(self, input_):
        """
        input_:     (batch_size, input_dim)
        """
        if self.use_bias:
            return tf.matmul(input_, self.W) + self.b
        else:
            return tf.matmul(input_, self.W)

    def get_params(self):
        if self.use_bias:
            return [self.W, self.b]
        else:
            return [self.W]

######################################################
# 두 층의 lstm layer를 위해 사용
# 업로드 해주신 실습 자료에서 __call__의 state_를 제거하고
# tf.nn.dynamic_rnn에서 initial_state 대신 dtype을 지정함
######################################################
class PTBLM(object):

    def __init__(self, vocabulary, sequence_length, batch_size,
                 hidden_dim=650, name='PTBLM'):
        self.vocabulary = vocabulary
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.name = name

        with tf.variable_scope(self.name):
            self.embed = Embed(vocabulary, hidden_dim, 'embed')
            self.lstm0 = LSTMCell(hidden_dim)
            self.lstm1 = LSTMCell(hidden_dim)
            self.lstm0_dropout = DropoutWrapper(self.lstm0, output_keep_prob=0.5)
            self.lstm1_dropout = DropoutWrapper(self.lstm1, output_keep_prob=0.5)
            self.stack = MultiRNNCell([self.lstm0, self.lstm1])
            self.stack_dropout = MultiRNNCell([self.lstm0_dropout, self.lstm1_dropout])
            self.dense = Dense(hidden_dim, vocabulary, 'dense')
        self.params = None

    def __call__(self, input_, states_, is_train=True, reuse_=False):
        """
        input_:     (batch_size, sequence_length)
        h1:         (batch_size, sequence_length, embed_dim)
        h2:         (batch_size, sequence_length, hidden_dim)
        h3:         (batch_size * sequence_length, hidden_dim)
        h4:         (batch_size * sequence_length, vocabulary)
        """

        h1 = self.embed(input_)
        with tf.variable_scope(self.name, reuse=reuse_):
            if is_train:
                h2, s2 = tf.nn.dynamic_rnn(self.stack, h1, initial_state=states_)
            else:
                h2, s2 = tf.nn.dynamic_rnn(self.stack_dropout, h1, initial_state=states_)
        h3 = tf.reshape(h2, [-1, self.hidden_dim])
        h4 = self.dense(h3)

        self.params = tf.trainable_variables()
        # tf.contrib.rnn generates parameters after dynamic_rnn call.
        # you should first run dummy data or first batch to get all parameters.
        # or, you can make your own lstm layer.

        return h4, s2

    def perplexity(self, logits, target):
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [logits],
            [target],  # [tf.reshape(input_.targets, [-1])],
            [tf.ones([self.sequence_length], dtype=tf.float32)])
        cost = tf.reduce_sum(loss)
        perplexity = tf.exp(cost / self.sequence_length)

        return perplexity

    def zero_state(self):
        return self.stack.zero_state(self.batch_size, tf.float32)

    def save_params(self, session):
        param_dir = './params_'
        for pp in self.params:
            value = session.run(pp)
            name_string = pp.name.split('/')
            path = param_dir + '/'.join(name_string[:-1]) + '/'
            if not os.path.exists(path):
                os.makedirs(path)
            np.save(path + name_string[-1][:-2] + '.npy', value)  # remove :0

    def load_params(self, session):
        # parameter 읽어오기
        word_embedding_matrix = np.load("parameters/embedding.npy")
        lstm_0_b = np.load("parameters/lstm_0_b.npy")
        lstm_0_w = np.load("parameters/lstm_0_w.npy")
        lstm_1_b = np.load("parameters/lstm_1_b.npy")
        lstm_1_w = np.load("parameters/lstm_1_w.npy")
        softmax_b = np.load("parameters/softmax_b.npy")
        softmax_w = np.load("parameters/softmax_w.npy")

        vars = {v.name: v for v in tf.trainable_variables()}
        session.run(vars["PTBLM/embed_Embed/W:0"].assign(word_embedding_matrix))
        session.run(vars["PTBLM/dense_Dense/W:0"].assign(softmax_w))
        session.run(vars["PTBLM/dense_Dense/b:0"].assign(softmax_b))
        session.run(vars["PTBLM/rnn/multi_rnn_cell/cell_0/lstm_cell/weights:0"].assign(lstm_0_w))
        session.run(vars["PTBLM/rnn/multi_rnn_cell/cell_0/lstm_cell/biases:0"].assign(lstm_0_b))
        session.run(vars["PTBLM/rnn/multi_rnn_cell/cell_1/lstm_cell/weights:0"].assign(lstm_1_w))
        session.run(vars["PTBLM/rnn/multi_rnn_cell/cell_1/lstm_cell/biases:0"].assign(lstm_1_b))


def ptbank():
    ptb_word_to_id_lines = open("ptb_word_to_id.txt").read().splitlines()
    word_dictionary = {}
    word_size = len(ptb_word_to_id_lines)
    id_list = [0] * word_size
    for line in ptb_word_to_id_lines:
        a = line.split("\t")
        word_dictionary[a[0]] = int(a[1])
        id_list[int(a[1])] = a[0]
    return word_dictionary, id_list
