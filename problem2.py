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

input_sentence = tf.placeholder(tf.int32, shape=[None, None]) # [batch_size, sequence_length]

model = PTBLM(vocabulary, sequence_length, batch_size)
initial_states = model.zero_state()
model(input_sentence, initial_states, reuse_=False)  # dummy call

# run tensorflow session
with tf.Session() as sess:
    # 초기화
    sess.run(tf.global_variables_initializer())
    model.load_params(sess)
    logits, final_state = model(input_sentence, initial_states, reuse_=True)  # reset states.
    perplexity = model.perplexity(logits, test_data[1:])
    perplexity_out = sess.run(perplexity, feed_dict={input_sentence: [test_data[:-1]]})
    print("perplexity : {}".format(perplexity_out))