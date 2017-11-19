from __future__ import division
import numpy as np
import re
import codecs
#import editdistance
from gensim.models import KeyedVectors
from collections import defaultdict
import string

def get_embedding(count_matrix, target_word, model):
    total = count_matrix[target_word]['__TOTAL__']
    model_dimension = model.vector_size
    source_embedding = np.zeros((model_dimension,))
    if target_word in count_matrix:
        source_words_dict = count_matrix[target_word]
        for source_word in source_words_dict:
            if source_word not in model: continue
            count = source_words_dict[source_word]
            source_embedding += count*model[source_word] 

    source_embedding = source_embedding/total
    return model.wv.similar_by_vector(source_embedding, topn=1)[0][0]


def generate_count_matrix():
    
    hindi_english_corpus = "/home/ninad/Desktop/NlpProj/data/HindEnCorp_parallel/HindEnCorp 0.5/hindencorp05.plaintext"
    word_align = "/home/ninad/Desktop/NlpProj/code/gh_codes/Cross-Lingual-POS-Tagging/forward.align"
    f = codecs.open(hindi_english_corpus, 'r', 'utf-8').read().split("\n")
    word_align = codecs.open(word_align, 'r', 'utf-8').read().split("\n")


    count_matrix = defaultdict(lambda: defaultdict(int))
    for index, sentence in enumerate(f):
        sentence = sentence.split('\t')
        if len(sentence)!=5: continue
        english_words = sentence[3].split()
        hindi_words = sentence[4].split()
        alignments = word_align[index].split()
        for idx in alignments:
            target_idx, source_idx = idx.split('-')
            source_idx = int(source_idx)
            target_idx = int(target_idx)
            count_matrix[hindi_words[source_idx]][english_words[target_idx]]+=1
            count_matrix[hindi_words[source_idx]]['__TOTAL__']+=1
    return count_matrix
        

model1 = KeyedVectors.load_word2vec_format('/home/ninad/Desktop/NlpProj/data/embedding/GoogleNews-vectors-negative300-SLIM.bin',binary=True)

text_file = open("/home/ninad/Desktop/NlpProj/code/baseline/parsed_test_hi.txt", "r")                         
lines = text_file.readlines()

ctMatrix = generate_count_matrix()
enWordList = []

for l in lines:
    if(l != '\n'):        
        hw = l.split()[0]
        ew = get_embedding(ctMatrix, hw, model1)
        enWordList.append(ew)
    else:
        enWordList.append('\n')

import pickle
with open('/home/ninad/Desktop/NlpProj/code/cmatrix/enWordList.pkl', 'wb') as wfile:
    pickle.dump(enWordList, wfile)

nidxs = [] #new line indices
for i in range(len(enWordList)):
    if(enWordList[i]=='\n'):
        nidxs.append(i)        

import nltk
i1 = 0
tagged_list = []

for j in range(len(nidxs)):
    i2 = nidxs[j]
    esent = enWordList[i1:i2]
    tagged_list.append(nltk.pos_tag(esent))
    tagged_list.append('\n')
    i1=i2+1

with open('/home/ninad/Desktop/NlpProj/code/cmatrix/enTagList.pkl', 'wb') as wfile:
    pickle.dump(tagged_list, wfile)
















