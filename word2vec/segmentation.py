# -*- coding: utf-8 -*-
__author__ = "the-Quert (github.com/the-Quert)"
import jieba
import logging
from hanziconv import HanziConv

class Segmentation(object):

	def __init__(self):
		# 用默認 Formatter 為日誌系統建立一個 StreamHandler ，設置基礎配置並加到 root logger 中
		logging.basicConfig(format = "%(asctime)s : %(levelname)s : %(message)s", level = logging.INFO)
		self.stopwordset = set()
		
	# Load stopwords.txt, and write into Stopwords Set
	def set_stopword(self):
		with open("/content/drive/My Drive/LAB/word2vec/stopwords.txt", "r", encoding = "utf-8") as stopwords:
			for stopword in stopwords:
				self.stopwordset.add(stopword.strip('\n'))
		#print(self.stopwordset)
		print("Stopword Set is stored.")
	
	# CN2TW, output traditional.txt
	def simplified_to_traditional(self):
		logging.info("Loading...CN2TW")
		traditional = open("traditional.txt", "w", encoding = "utf-8")
		with open("/content/drive/My Drive/LAB/word2vec/wiki_text.txt", "r", encoding = "utf-8") as simplified:
			for s in simplified:
				traditional.write(HanziConv.toTraditional(s))
		print("CN2TW is done.")
		traditional.close()
	
	# Segmentation and reject stopwords. Output segmentation.txt
	def segmentation(self):
		logging.info("Loading..(jieba segmentation，and reject stopwords)")
		segmentation = open("segmentation.txt", "w", encoding = "utf-8")
		with open("traditional.txt", "r", encoding = "utf-8") as Corpus:
			for sentence in Corpus:
				sentence = sentence.strip("\n")
				pos = jieba.cut(sentence, cut_all = False)
				for term in pos:
					if term not in self.stopwordset:
						segmentation.write(term + " ")
		print("Jieba segmentation and stopwords process is done.")
		segmentation.close()

if __name__ == "__main__":
	segmentation = Segmentation()
	segmentation.set_stopword()
	segmentation.simplified_to_traditional()
	segmentation.segmentation()