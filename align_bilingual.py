from __future__ import division
import numpy as np
import re
import codecs
import editdistance
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
    return model.wv.similar_by_vector(source_embedding, topn=1)


def generate_count_matrix():
    model = KeyedVectors.load_word2vec_format('/Users/shreesh/Downloads/GoogleNews-vectors-negative300-SLIM.bin',binary=True)

    hindi_english_corpus = "/Users/shreesh/Academics/CS585/Project/HindEnCorp 0.5/hindencorp05.plaintext"
    word_align = "/Users/shreesh/Academics/CS585/Project/fast_align-master/build/forward.align"
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
        
#get_embedding(count_matrix,target_word,model)



