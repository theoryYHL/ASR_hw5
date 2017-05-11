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

        with tf.variable_scope(name):
            self.embed = Embed(vocabulary, hidden_dim, 'embed')
            self.lstm0 = LSTMCell(hidden_dim)
            self.lstm1 = LSTMCell(hidden_dim)
            self.lstm0_dropout = DropoutWrapper(self.lstm0, output_keep_prob=0.5)
            self.lstm1_dropout = DropoutWrapper(self.lstm1, output_keep_prob=0.5)
            self.stack = MultiRNNCell([self.lstm0, self.lstm1])
            self.stack_dropout = MultiRNNCell([self.lstm0_dropout, self.lstm1_dropout])
            self.dense = Dense(hidden_dim, vocabulary, 'dense')
        self.params = None

    def __call__(self, input_, is_train=True):
        """
        input_:     (batch_size, sequence_length)
        h1:         (batch_size, sequence_length, embed_dim)
        h2:         (batch_size, sequence_length, hidden_dim)
        h3:         (batch_size * sequence_length, hidden_dim)
        h4:         (batch_size * sequence_length, vocabulary)
        """
        h1 = self.embed(input_)
        if is_train:
            h2, s2 = tf.nn.dynamic_rnn(self.stack, h1, dtype=tf.float32)
        else:
            h2, s2 = tf.nn.dynamic_rnn(self.stack_dropout, h1, dtype=tf.float32)
        h3 = tf.reshape(h2, [-1, self.hidden_dim])
        h4 = self.dense(h3)
        self.params = tf.trainable_variables()
        # tf.contrib.rnn generates parameters after dynamic_rnn call.
        # you should first run dummy data or first batch to get all parameters.
        # or, you can make your own lstm layer.
        return h4, s2

    def zero_state(self):
        return self.c2.zero_state(self.batch_size, tf.float32)

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
        param_dir = './params_'
        for pp in self.params:
            name_string = pp.name.split('/')
            path = param_dir + '/'.join(name_string[:-1]) + '/'
            value = np.load(path + name_string[-1][:-2] + '.npy')
            session.run(pp.assign(value))

def main():

    # ptb_word_to_id.txt 읽어오기
    ptb_word_to_id_lines = open("ptb_word_to_id.txt").read().splitlines()
    # word_distionary: 단어(string) 넣으면 id(int)를 알려주는 dictionary
    # id_list: n번째 위치에 n을 id로 가지는 단어가 저장된 list
    word_dictionary = {}
    word_size = len(ptb_word_to_id_lines)
    id_list = [0] * word_size
    for line in ptb_word_to_id_lines:
        a = line.split("\t")
        word_dictionary[a[0]] = int(a[1])
        id_list[int(a[1])] = a[0]

    # test 데이터 수집
    # test_data: <eos>상관없이 수집
    # test_data_by_sentence: <eos>에 따라 문장을 구분해서 수집
    test_data = []
    test_data_by_sentence= [[]]
    sentence_order = 0
    ptb_test_index_lines = open("ptb_test_index.txt").read().splitlines()
    for line in ptb_test_index_lines:
        test_data.append(int(line))
        test_data_by_sentence[sentence_order].append(int(line))
        if int(line)==2: # <eos>가 나오면 다음 문장을 수집하기 시작
            test_data_by_sentence.append([])
            sentence_order +=1
    # 마지막 <eos>이후에는 다음 문장이 오지 않으므로 test_data_by_sentence에 마지막으로 append된 []는 제거해야함
    test_data_by_sentence.pop()
    # 가장 긴 문장의 길이 측정
    longest_sentense_size = np.amax([len(sentence) for sentence in test_data_by_sentence])

    #parameter 읽어오기
    word_embedding_matrix = np.load("parameters/embedding.npy")
    lstm_0_b = np.load("parameters/lstm_0_b.npy")
    lstm_0_w = np.load("parameters/lstm_0_w.npy")
    lstm_1_b = np.load("parameters/lstm_1_b.npy")
    lstm_1_w = np.load("parameters/lstm_1_w.npy")
    softmax_b = np.load("parameters/softmax_b.npy")
    softmax_w = np.load("parameters/softmax_w.npy")

    with tf.Graph().as_default():

        vocabulary = word_size # = 10000
        sequence_length=longest_sentense_size # = 78
        batch_size=1 # 학습을 시키지 않기 때문에 하나면 될 듯

        # input_sentence: 뉴럴넷의 input이 됨
        # PTBLM의 __call__을 참조하여 차원 구성, (batch_size, sequence_length) => 2-d
        # batch는 1이면 [1,문장길이] 차원의 matrix가 됨
        input_sentence = tf.placeholder(tf.int64,shape=[None,None])
        model = PTBLM(vocabulary,sequence_length,batch_size) # 현재 input은 사용하지 않는 의미 없는 값들
        h4, s2 = model(input_sentence) # h4: 마지막 layer output, s2: 중간 lstm의 state값

        # paramater 값 입력
        # vars는 이름을 입력하면 variable 자체를 가지고 옴
        ###### 이름을 알아내는 방법
        ###### for v in tf.global_variables():
        ######     print(v.name)
        # vars를 이용해 assign 함수로 적당한 parameter값을 줌
        vars = {v.name: v for v in tf.trainable_variables()}
        vars["PTBLM/embed_Embed/W:0"].assign(word_embedding_matrix)
        vars["PTBLM/dense_Dense/W:0"].assign(softmax_w)
        vars["PTBLM/dense_Dense/b:0"].assign(softmax_b)
        vars["rnn/multi_rnn_cell/cell_0/lstm_cell/weights:0"].assign(lstm_0_w)
        vars["rnn/multi_rnn_cell/cell_0/lstm_cell/biases:0"].assign(lstm_0_b)
        vars["rnn/multi_rnn_cell/cell_1/lstm_cell/weights:0"].assign(lstm_1_w)
        vars["rnn/multi_rnn_cell/cell_1/lstm_cell/biases:0"].assign(lstm_1_b)

        # run tensorflow session
        with tf.Session() as sess:
            # 초기화
            sess.run(tf.global_variables_initializer())
            # feed_dict을 이용해서 input 문장으로 첫 번째 문장을 넣고 마지막 h4 layer의 output값을 받아봄
            output = sess.run([h4],feed_dict={input_sentence: [test_data_by_sentence[0]]})
            # output은 [1,7,10000] = [batchsize, 문장 길이, 단어 개수] 차원의 tensor가 됨
            


if __name__ == '__main__':
    main()