from PTBLM import PTBLM
from PTBLM import ptbank
import tensorflow as tf
import numpy as np

word_dictionary, id_list = ptbank()

vocabulary = 10000
batch_size = 1
sequence_length = 1000

input_sentence = tf.placeholder(tf.int32, shape=[None, None]) # [batch_size, sequence_length]
target_h = tf.placeholder(tf.int32, shape=[None, None])
initial_states = tf.placeholder(tf.float32, shape=[2, 2, batch_size, 650])

model = PTBLM(vocabulary, sequence_length, batch_size)
logits, final_state = model(input_sentence, initial_states)
init_states = np.zeros((2, 2, batch_size, 650))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    model.load_params(sess)

    # problem 3-1
    print("\nproblem 3-1")
    pr2_sentence_raw = ['i have a cat in my home', 'i have a cat in my house', 'i have a cat in me house']
    pr2_word_ids = []
    for sentence in pr2_sentence_raw:
        words = sentence.split()
        word_ids = []
        for word in words:
            word_ids.append(word_dictionary[word])
        pr2_word_ids.append(word_ids)

    prob = tf.nn.softmax(logits)
    softmax_probs = list()
    for pr2_word_id in pr2_word_ids:
        softmax_probs.append(sess.run(prob, feed_dict={input_sentence: [pr2_word_id], initial_states: init_states}))

    unigram_i = 0.00135759
    logLikelihoods = []
    for word_ids, softmax_prob in zip(pr2_word_ids, softmax_probs):
        likelihood = []
        target_word_ids = word_ids[1:]
        for idx, word_id in enumerate(target_word_ids):
            likelihood.append(softmax_prob[idx, word_id])
        logLikelihood = np.sum(np.log(likelihood)) + np.log(unigram_i)
        logLikelihoods.append(logLikelihood)

    for words, logLikelihood in zip(pr2_sentence_raw, logLikelihoods):
        print("loglikelihood  of '{}' : {}".format(words, logLikelihood))

    # problem 3-2
    print("\nproblem 3-2")
    seed_words = ["some", "where"]
    for seed_word in seed_words:
        input_word = [word_dictionary[seed_word]]
        generated_words = ["<" + seed_word + ">"]

        logit_outs, prev_state = sess.run([logits, final_state], feed_dict={input_sentence: [input_word], initial_states: init_states})
        next_word = np.argmax(logit_outs, axis=1)
        generated_words.append(id_list[next_word[0]])
        while next_word != 2:
            logit_outs, prev_state = sess.run([logits, final_state], feed_dict={input_sentence: [next_word], initial_states: prev_state})
            next_word = np.argmax(logit_outs, axis=1)
            generated_words.append(id_list[next_word[0]])
        print(" ".join(generated_words))

    # problem 3-3
    print("\nproblem 3-3")
    seed_words = ["some", "where"]
    for seed_word in seed_words:
        input_word = [word_dictionary[seed_word]]
        generated_words1 = ["<" + seed_word + ">"]
        generated_words2 = ["<" + seed_word + ">"]

        prob = tf.log(tf.nn.softmax(logits))
        prob_out, prev_state = sess.run([prob, final_state], feed_dict={input_sentence: [input_word], initial_states: init_states})
        top2words = np.argsort(prob_out, axis=1)[0, -2:]
        top2probs = prob_out[0, top2words]
        generated_words1.append(id_list[top2words[0]])
        generated_words2.append(id_list[top2words[1]])
        prev_state1 = prev_state2 = prev_state

        # beam search
        while 2 not in top2words:
            prob = tf.log(tf.nn.softmax(logits))
            feed_dict = {input_sentence: [[top2words[0]]], initial_states: prev_state1}
            prob_out1, prev_state1 = sess.run([prob, final_state], feed_dict)
            prob_out1 += top2probs[0]
            top2from1 = np.sort(prob_out1, axis=1)[0, -2:]

            feed_dict = {input_sentence: [[top2words[1]]], initial_states: prev_state2}
            prob_out2, prev_state2 = sess.run([prob, final_state], feed_dict)
            prob_out2 += top2probs[1]
            top2from2 = np.sort(prob_out2, axis=1)[0, -2:]

            if top2from1[0] > top2from2[1]:
                top2words = np.argsort(prob_out1, axis=1)[0, -2:]
                top2probs = prob_out1[0, top2words]
                prev_state2 = prev_state1
                generated_words2 = generated_words1.copy()
                generated_words1.append(id_list[top2words[0]])
                generated_words2.append(id_list[top2words[1]])
            elif top2from2[0] > top2from1[1]:
                top2words = np.argsort(prob_out2, axis=1)[0, -2:]
                top2probs = prob_out2[0, top2words]
                prev_state1 = prev_state2
                generated_words1 = generated_words2.copy()
                generated_words1.append(id_list[top2words[0]])
                generated_words2.append(id_list[top2words[1]])
            else:
                top2words = np.concatenate([np.argmax(prob_out1, axis=1), np.argmax(prob_out2, axis=1)])
                top2probs = [prob_out1[0, top2words[0]], prob_out2[0, top2words[1]]]
                generated_words1.append(id_list[top2words[0]])
                generated_words2.append(id_list[top2words[1]])

        # greedy search
        if top2words[0] != 2:
            next_word = [top2words[0]]
            while next_word != 2:
                prob = tf.log(tf.nn.softmax(logits))
                feed_dict = {input_sentence: [next_word], initial_states: prev_state1}
                prob_out, prev_state1 = sess.run([prob, final_state], feed_dict)
                next_word = np.argmax(prob_out, axis=1)
                generated_words1.append(id_list[next_word[0]])
        elif top2words[1] != 2:
            next_word = [top2words[1]]
            while next_word != 2:
                prob = tf.log(tf.nn.softmax(logits))
                feed_dict = {input_sentence: [next_word], initial_states: prev_state2}
                prob_out, prev_state2 = sess.run([prob, final_state], feed_dict)
                next_word = np.argmax(prob_out, axis=1)
                generated_words2.append(id_list[next_word[0]])
        print(" ".join(generated_words1))
        print(" ".join(generated_words2))
