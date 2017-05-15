from PTBLM import PTBLM
from PTBLM import ptbank
import tensorflow as tf
import numpy as np

word_dictionary, id_list = ptbank()

test_data = []
ptb_test_index_lines = open("ptb_test_index.txt").read().splitlines()
for line in ptb_test_index_lines:
    test_data.append(int(line))

vocabulary = 10000
# test_data = test_data[:1000]
sequence_length = len(test_data) - 1
batch_size = 1

model = PTBLM(vocabulary, sequence_length, batch_size)
input_sentence = tf.placeholder(tf.int32, shape=[None, None])  # [batch_size, sequence_length]
initial_states = tf.placeholder(tf.float32, shape=[2, 2, batch_size, 650])
logits, final_state = model(input_sentence, initial_states, reuse_=False)

# run tensorflow session
with tf.Session() as sess:
    # 초기화
    sess.run(tf.global_variables_initializer())
    model.load_params(sess)
    init_state = np.zeros((2, 2, batch_size, 650))
    perplexity = model.perplexity(logits, test_data[1:])
    feed_dict = {input_sentence: [test_data[:-1]], initial_states: init_state}
    perplexity_out, prev_states = sess.run([perplexity, final_state], feed_dict)
    print("perplexity : {}".format(perplexity_out))
