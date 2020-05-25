# coding=utf-8
__author__ = "the-Quert (github.com/the-quert)"

import jieba
import logging
from hanziconv import Hanziconv

class Segmentation(object):
	def __init__(self):
		# Add Stream Handler
		logging.basicConfig(format = "%(asctime)s, %(levelname)s, %(message)s", level = logging.INFO)
		self.stopwordset = set()

	def set_stopwords(self):
		with open("/content/drive/My Drive/LAB/word2vec/stopwords.txt", "r", encoding="utf-8") as stopwords:
			for stopword in stopwords:
				self.stopwordset.add(stopword.strip('\n'))
		print("Stopwords set is done.")

	def simplified_to_traditional(self):
		logging.info("Loading CN2TW...")
		traditional = open("traditional.txt", "w", encoding = "utf-8")
		with open("/content/drive/My Drive/LAB/word2vec/wiki_text.txt", "r", encoding = "utf-8") as simplified:
			for s in simplified:
				traditional.write(Hanziconv.toTraditional(s))
			print("CS2TW is done.")
		traditional.close()

	def segmentation(self):
		logging.INFO("Loading...Jieba segmentation, and reject stopwords.")
		segmentation = open("segmentation.txt", "w", encoding = "utf-8", errors = "ignore")
		with open("traditional.txt", "r", encoding = "utf-8", errors = "ignore") as Corpus:
			for sentence in Corpus:
				sentence = sentence.strip("\n")
				after_cut = jieba.cut(sentence, cut_all = False)
				for term in after_cut:
					if term not in self.stopwordset:
						segmentation.write(term + " ")
		print("Jieba segmentaion and stopwrods rejection is done.")
		segmentation.close()

if __name__ == "__main__":
	segmentation = Segmentation()
	segmentation.set_stopwords()
	segmentation.simplified_to_traditional()
	segmentation.segmentation()