# -*- coding: utf-8 -*-
__author__ = "the-Quert (github.com/the-quert)"
import warnings
warnings.filterwarnings(action = 'ignore', category = UserWarning, module = 'gensim')
from gensim.models import word2vec


# Traing model with gensim
class Train(object):

	def __init__(self):
		pass

	def train(self):
		print("Training...")
		# Load file
		sentence = word2vec.Text8Corpus("/content/drive/My Drive/LAB/word2vec/segmentation.txt")
		# Setting degree and Produce Model(Train)
		# model = word2vec.Word2Vec(sentence, size = 300, window = 10, min_count = 5, workers = 4, sg = 1)
		model = word2vec.Word2Vec(sentence, size = 150, window = 10, min_count = 5, workers = 4, sg = 1)
		# Save model 
		model.wv.save_word2vec_format(u"wiki300.model.bin", binary = True)
		print("Training process is done. Model is stored.")

if __name__ == "__main__":
	t = Train()
	t.train()


'''
https://radimrehurek.com/gensim/models/word2vec.html

'''