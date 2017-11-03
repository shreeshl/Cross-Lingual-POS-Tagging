from __future__ import division
import codecs
import pycrfsuite
from sklearn.model_selection import train_test_split
import numpy as np
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics

def word2features(doc, i):
    word = doc[i][0]
    postag = doc[i][1]

    # Common features for all words
    features = [
        'bias=1',
        postag+'=1',
        'word='+word
        # 'word[-3:]=' + word[-3:],
        # 'word[-2:]=' + word[-2:],
        # 'word.isdigit=%s' % word.isdigit(),
    ]

    # Features for words that are not
    # at the beginning of a document
    if i > 0:
        word1 = doc[i-1][0]
        postag1 = doc[i-1][1]
        # features.extend([
            #  '-1:word.isdigit=%s' % word1.isdigit(),
        #  ])
    else:
        # Indicate that it is the 'beginning of a document'
        features.append('BOS')

    # Features for words that are not
    # at the end of a document
    if i < len(doc)-1:
        word1 = doc[i+1][0]
        postag1 = doc[i+1][1]
        # features.extend([
        #      '+1:word.isdigit=%s' % word1.isdigit(),
        #  ])
    else:
        # Indicate that it is the 'end of a document'
        features.append('EOS')

    return features

def extract_features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def get_labels(sent):
    return [label for (token, label) in sent]

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

train_data = codecs.open("parsed_train.txt", 'r', 'utf-8').read().split("\n\n")
test_data = codecs.open("parsed_test.txt", 'r', 'utf-8').read().split("\n\n")

X_train, y_train = get_data(train_data)
X_test, y_test = get_data(test_data)


trainer = pycrfsuite.Trainer(verbose=True)
for xseq, yseq in zip(X_train, y_train):
    trainer.append(xseq, yseq)

trainer.set_params({
    # coefficient for L1 penalty
    'c1': 0.1,
    # coefficient for L2 penalty
    'c2': 0.01,  
    # maximum number of iterations
    'max_iterations': 200,
    # whether to include transitions that
    # are possible, but not observed
    'feature.possible_transitions': True
})
trainer.train('crf.model')
tagger = pycrfsuite.Tagger()
tagger.open('crf.model')
y_pred_train = [tagger.tag(xseq) for xseq in X_train]
y_pred = [tagger.tag(xseq) for xseq in X_test]

y_pred = y_pred
Y_test = y_test
total = 0
for i in range(len(y_pred)):
    if(y_pred[i]==y_test[i]): total+=1

total = total/len(y_pred)

sorted_labels = sorted(
    tagger.labels(),
    key=lambda name: (name[1:], name[0])
)

print "Train Statistics"
print metrics.flat_classification_report(
    y_train, y_pred_train, labels=sorted_labels, digits=3
)

print "Test Statistics"
print metrics.flat_classification_report(
    y_test, y_pred, labels=sorted_labels, digits=3
)