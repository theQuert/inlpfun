#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 11:21:44 2020

@author: quert
"""

import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
poses = {'n': 'noun', 'v': 'verb', 's': 'adj (s)', 'a': 'adj', 'r': 'adv'}
for synset in wn.synsets("good"):
    print("{}: {}".format(poses[synset.pos()],
          ", ".join([l.name() for l in synset.lemmas()])))


wn.synsets('motorcar')
wn.synset('car.n.01').lemma_names()

wn.synset('car.n.01').examples()

wn.synset('car.n.01').definition()

# lemma: Combination in a phrase.

wn.synsets('automobile')

wn.synsets('car')

for synset in wn.synsets('car'):
    print(synset.lemma_names())
    
# Hyponyms

motorcar = wn.synset('car.n.01')
types_of_motorcar = motorcar.hyponyms()
types_of_motorcar[26]
print(types_of_motorcar)

# Hypernyms

motorcar.hypernyms()
paths = motorcar.hypernym_paths()
[synset.name for synset in paths[0]]

panda = wn.synset('panda.n.01')
hyper = lambda s: s.hypernyms()
list(panda.closure(hyper))            


from nltk.corpus import wordnet as wn
poses = {'n': 'noun', 'v': 'verb', 's': 'adj (s)', 'a': 'adj', 'r': 'adv'}
for synset in wn.synsets("good"):
    print("[}: {}".format(poses[synset.pos()]))
    
