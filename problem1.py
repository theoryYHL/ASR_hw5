import numpy as np
import scipy.spatial

# embedding.npy와 ptb_word_to_id.txt 읽어오기
word_embedding_matrix = np.load("parameters/embedding.npy")
ptb_word_to_id_lines = open("ptb_word_to_id.txt").read().splitlines()
# word_distionary: 단어(string) 넣으면 id(int)를 알려주는 dictionary
# id_list: n번째 위치에 n을 id로 가지는 단어가 저장된 list
word_dictionary = {}
word_size = len(ptb_word_to_id_lines)
id_list = [0]*word_size
for line in ptb_word_to_id_lines:
    a = line.split("\t")
    word_dictionary[a[0]] = int(a[1])
    id_list[int(a[1])] = a[0]


# 1-1 가장 가까운 단어 찾기
words_list = ["the","discount","crazy","birthday","just"]
for word in words_list:
    id = word_dictionary[word]
    # 모든 단어와의 cosine distance를 측정
    cosine_distance_from_all_words = np.zeros((word_size))
    for i in range(word_size):
        cosine_distance_from_all_words[i] = \
            scipy.spatial.distance.cosine(word_embedding_matrix[i], word_embedding_matrix[id])
    cosine_distance_from_all_words[id] = 2 # 자기 자신과의 거리가 가장 가까우므로 제외하기 위함
    nearest_word_id = np.argmin(cosine_distance_from_all_words)
    nearest_word = id_list[nearest_word_id]
    print("'{}' nearest to '{}' with distance {}".format(nearest_word, word, cosine_distance_from_all_words[nearest_word_id]))

# 1-2 답에 가까운 단어 찾기
words_list = [["king","male","female"],["breakfast","morning","evening"]]
for words in words_list:
    ids = [word_dictionary[word] for word in words]
    calculated_vector = word_embedding_matrix[ids[0]] - word_embedding_matrix[ids[1]] + word_embedding_matrix[ids[2]]
    # 모든 단어와의 cosine distance를 측정
    cosine_distance_from_all_words = np.zeros((word_size))
    for i in range(word_size):
        cosine_distance_from_all_words[i] = \
            scipy.spatial.distance.cosine(word_embedding_matrix[i], calculated_vector)
    nearest_word_id = np.argmin(cosine_distance_from_all_words)
    nearest_word = id_list[nearest_word_id]
    print("'{}' - '{}' + '{}' = {}".format(words[0], words[1], words[2], nearest_word))


