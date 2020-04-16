# -*- coding: utf-8 -*-
__author__ = "the-Quert (github.com/the-quert)"
import warnings
warnings.filterwarnings(action = 'ignore', category = UserWarning, module = 'gensim')
from gensim.models.keyedvectors import KeyedVectors
'''
from wiki_to_txt import Wiki_to_txt
from segmentation import Segmentation
from train import Train
'''

# Load model, and implementation.
def main():
	
	'''
	wiki_to_txt = Wiki_to_txt()
	# 
	wiki_to_txt.set_wiki_to_txt("zhwiki-latest-pages-articles.xml.bz2")
	segmentation = Segmentation()
	
	segmentation.set_stopword()
	
	segmentation.simplified_to_traditional()
	
	segmentation.segmentation()
	t = Train()
	
	t.train()
	'''
	# Reference: https://radimrehurek.com/gensim/models/word2vec.html
	word_vectors = KeyedVectors.load_word2vec_format("wiki300.model.bin", binary = True)
	print("\n1.輸入一個詞會找出前5名相似")
	print("2.輸入兩個詞算出相似度")
	print("3.輸入三個詞a之於b,如b之於c")
	while True:
		try:
			query = input("\n輸入格式( Ex: a,b,....最多三個詞)\n")
			query_list = query.split(",")
			if len(query_list) == 1:
				print("相似詞前 5 排序")
				res = word_vectors.most_similar(query_list[0], topn = 5)
				for item in res:
					print(item[0] + "," + str(item[1]))
			elif len(query_list) == 2:
				print("兩個詞 Cosine 相似度")
				res = word_vectors.similarity(query_list[0], query_list[1])
				print(res)
			else:
				print("%s之於%s，如%s之於" % (query_list[0], query_list[1], query_list[2]))
				res = word_vectors.most_similar(positive = [query_list[0], query_list[1]], negative = [query_list[2]], topn = 5)
				for item in res:
					print(item[0] + "," + str(item[1]))
		except Exception as e:
			print("Error:" + repr(e))

if __name__ == "__main__":
    main()
