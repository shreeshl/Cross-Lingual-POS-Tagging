from __future__ import division
import codecs
from collections import defaultdict

def get_data(corpus):
    data = []
    for item in corpus:
        temp = []
        for labels in item.split("\n"):
            if(len(labels)==0): continue
            word, tag = labels.split("\t")
            temp.append((word, tag))
        data.append(temp)
    X = [extract_features(sent) for sent in data]
    y = [get_labels(sent) for sent in data]
    return X,y

data_hi = codecs.open("parsed_train_hi.txt", 'r', 'utf-8').read().split("\n\n") + codecs.open("parsed_test_hi.txt", 'r', 'utf-8').read().split("\n\n")
data_ta = codecs.open("parsed_train_ta.txt", 'r', 'utf-8').read().split("\n\n") + codecs.open("parsed_test_ta.txt", 'r', 'utf-8').read().split("\n\n")

pos_types_hi = defaultdict(set)
pos_types_ta = defaultdict(set)

for i,sentence in enumerate(data_hi):
    sentence = sentence.split('\n')
    for word in sentence:
        word = word.split('\t')
        if(len(word)!=2):continue
        pos_types_hi[word[0]].add(word[1])

for i,sentence in enumerate(data_ta):
    sentence = sentence.split('\n')
    for word in sentence:
        word = word.split('\t')
        if(len(word)!=2):continue
        pos_types_ta[word[0]].add(word[1])

pos_per_word_hi = defaultdict(float)
pos_per_word_ta = defaultdict(float)

total_hi = 0
total_ta = 0

for word in pos_types_hi:
    pos_per_word_hi[len(pos_types_hi[word])]+=1
    total_hi+=1

for word in pos_types_ta:
    pos_per_word_ta[len(pos_types_ta[word])]+=1
    total_ta+=1

print "HINDI ANALYSIS"
print "====================="
for i in pos_per_word_hi:
    print "Percentage of tags with %s different tags is %.4f (exact : %d )"%(i,pos_per_word_hi[i]/total_hi,pos_per_word_hi[i])

print "\n"
print "TAMIL ANALYSIS"
print "====================="
for i in pos_per_word_ta:
    print "Percentage of tags with %s different tags is %.4f (exact : %d )"%(i,pos_per_word_ta[i]/total_ta,pos_per_word_ta[i])