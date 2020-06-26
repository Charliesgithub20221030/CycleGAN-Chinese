import gensim
import logging
import os
import json


class MySentences(object):
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        for line in open(self.filename):
            sent = line.split(' +++$+++ ')[1].split()
            sent = ['__BOS__'] + sent
            for _ in range(128-len(sent)):
                sent = sent + ['__EOS__']
            yield sent


sentences = MySentences('data/feature_ptt_spliit.txt')
model = gensim.models.Word2Vec(
    sentences, size=200, window=5, min_count=1, workers=7)
model.save('./word2vec_model/model_ptt')

i = 0
word_id_dict = dict()
for word in model.wv.vocab:
    word_id_dict[word] = i
    i += 1
fp = open('./word2vec_model/dictionary_ptt_word2vec.json', 'w')
json.dump(word_id_dict, fp)
