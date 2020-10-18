# -- coding=utf-8 --

import numpy as np
from collections import defaultdict
setting = {
    'window_size': 2,
    'embedding_dim': 10,
    'epochs': 50,
    'learning_rate': 0.01
}


class word2vec():
    def __init__(self, setting):
        self.embedding_dim = setting['embedding_dim']
        self.lr = setting['lr']
        self.epochs = setting['epochs']
        self.window = setting['window_size']

    def generate_training_data(self, corpus):
        word_counts = defaultdict(int)
        for row in corpus:
            for word in row:
                word_counts[word] += 1
        self.v_count = len(word_counts.keys())

        self.words_list = list(word_counts.keys())

        self.word_index = dict((word, i) for i, word in enumerate(self.words_list))

        self.index_word = dict((i, word) for i, word in enumerate(self.words_list))


        training_data = []

        for sentence in corpus:
            sent_len = len(sentence)

            for i, word in enumerate(sentence):

                w_target = self.word2onehot(sentence[i])

                w_context = []

                for j in range(i-self.window, i+self.window+1):
                    if j != i and j < sent_len and j >= 0:
                        w_context.append(self.word2onehot(sentence[j]))
                
                training_data.append([w_target, w_context])

        return np.array(training_data)

    def word2onehot(self, word):
        # word_vec = [0 for in in range(0, self.v_count)]
        word_vec = np.zeros(self.v_count)

        word_index = self.word_index[word]

        word_vec[word_index] = 1

        return word_vec

    def train(self, training_data):
        self.w1 = np.random.uniform(-1, 1, (self.v_count, self.embedding_dim))
        self.w2 = np.random.uniform(-1, 1, (self.embedding_dim, self.v_count))

        for i in range(self.epochs):
            self.loss = 0

            for w_t, w_c in training_data:
                y_pred, h, u = 
    

