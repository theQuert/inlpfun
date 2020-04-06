import logging

from gensim.models import word2vec

def main():

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = word2vec.LineSentence("wiki_seg.txt")
    model = word2vec.Word2Vec(sentences, size=250)

    model.save("word2vec.model")

    # model = word2vec.Word2Vec.load("model_name")

if __name__ == "__main__":
    main()


  '''

sentences: Sentences in training set.
size: Dimensions of word vectors.
alpha: Learning rate.
sg: sg=1, use SG. sg=0, use CBOW.
window:  Window size.
workers: Numbers of threads, 4 is maximum for most devices.
min_count: If word counting time is less than this value, it's not included in training set.
  '''