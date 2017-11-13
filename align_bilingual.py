from __future__ import division
import numpy as np
import re
import codecs
import editdistance
from gensim.models import KeyedVectors
from collections import defaultdict
import string

def get_embedding(count_matrix, target_word, model):
    #total = count_matrix[target_word]['__TOTAL__']
    source_embedding = np.zeros((300,))
    if target_word in count_matrix:
        source_words_dict = count_matrix[target_word]
        for source_word in source_words_dict:
            if source_word not in model: continue
            count = source_words_dict[source_word]
            source_embedding += count*model[source_word] 
    source_embedding = source_embedding/total
    return model.wv.similar_by_vector(source_embedding, topn=10)


hindi_english_corpus = "/Users/shreesh/Academics/CS585/Project/HindEnCorp 0.5/hindencorp05.plaintext"
f = codecs.open(hindi_english_corpus, 'r', 'utf-8').read().split("\n")
model = KeyedVectors.load_word2vec_format('/Users/shreesh/Downloads/GoogleNews-vectors-negative300-SLIM.bin',binary=True)

count_matrix = defaultdict(lambda: defaultdict(int))
for sentence in f:
    sentence =  sentence.split('\t')
    if len(sentence)!=5: continue
    english_words = sentence[3].split()
    hindi_words = sentence[4].split()
    for hi_word in hindi_words:
        search = re.search('[a-zA-Z0-9]', hi_word)
        if search:continue
        for en_word in english_words:
            count_matrix[hi_word.strip()][en_word.strip(string.punctuation).lower()]+=1
            #count_matrix[hi_word.strip()]['__TOTAL__'] +=1            
            
get_embedding(count_matrix,target_word,model)