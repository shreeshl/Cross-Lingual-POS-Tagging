import cPickle as p
import numpy as np
import keras
import nltk
from keras.layers import Embedding, LSTM, Dense, Dropout, concatenate, Input
from keras.models import Model
import itertools
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
vocabulary_size = 4000
unknown_token = 'UNKNOWN_TOKEN'
np.random.seed(0)
fname = ["clean_hin_sports_set1.txt.pkl","clean_hin_agriculture_set1.txt.pkl","clean_hin_philosophy_set1.txt.pkl","clean_hin_economy_set1.txt.pkl",
        "clean_hin_politics and public administration_set2.txt.pkl","clean_hin_entertainment_set1.txt.pkl","clean_hin_religion_set1.txt.pkl","clean_hin_literature_set1.txt.pkl"]

fname2 = [
"Entags_clean_hin_sports_set1.txt.pkl",
"Entags_clean_hin_agriculture_set1.txt.pkl",
"Entags_clean_hin_philosophy_set1.txt.pkl",
"Entags_clean_hin_economy_set1.txt.pkl",
"Entags_clean_hin_politics and public administration_set2.txt.pkl",
"Entags_clean_hin_entertainment_set1.txt.pkl",
"Entags_clean_hin_religion_set1.txt.pkl",
"Entags_clean_hin_literature_set1.txt.pkl"]
     
fname2idx = {} 
for idx,i in enumerate(fname):        
    fname2idx[i] = idx
    idx+=1   

def parse_sentence(sentence):
    sent = ""  
    pos = []   
    for i in sentence.split(' '):     
        i = i.strip().split("\t")     
        if len(i)!=2: continue        
        sent += i[0] + " "   
        pos.append(i[1].encode('ascii','ignore')) 
    return sent.strip(), pos 
  
  
# t = Tokenizer(num_words=1000,split = " ")    
dataset, y, postags = [], [], []        
for idx,i in enumerate(fname):
    f, m2 = p.load(open(i,'r')), p.load(open(fname2[idx],'r'))
    f2 = []
    for k in m2:
        if(len(k)==1) :f2.append(k)
        else: f2.append(k[0]+'\t'+k[1])
    f, f2 = ' '.join(f).split('\n'), ' '.join(f2).split('\n')
    for item, item2 in zip(f,f2):
        sent, pos = parse_sentence(item)
        sent2, pos2 = parse_sentence(item2)
        dataset.append(nltk.word_tokenize(sent))      
        postags.append(pos2)
        y.append(fname2idx[i])
  
pos2idx = {}        
idx = 0       
for l in postags:
    for i in l:      
        if i not in pos2idx:
            pos2idx[i] = idx
            idx+=1

word_freq = nltk.FreqDist(itertools.chain(*dataset))       
print "Found %d unique words tokens." % len(word_freq.items())      
  
vocab = word_freq.most_common(vocabulary_size) 
index_to_word = [x[0] for x in vocab] 
index_to_word.append(unknown_token)   
word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])   
  
for i, sent in enumerate(dataset):    
    dataset[i] = [w if w in word_to_index else unknown_token for w in sent]      
  
X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in dataset])
  
unique_pos = len(pos2idx.keys())
postag_hist = []
for postag_list in postags:
    m = [0]*unique_pos
    for pos in postag_list:
        m[pos2idx[pos]]+=1
    postag_hist.append(m)

postag_hist = np.array(postag_hist)
data = pad_sequences(X_train, maxlen=20)      


idxs = np.arange(len(data))
np.random.shuffle(idxs)
y = np.array(y)[idxs]
X_train = X_train[idxs]
data = data[idxs]
postag_hist = postag_hist[idxs]



main_input = Input(shape = (20,))     
auxiliary_input = Input(shape=(unique_pos,), name='aux_input')      

x = Embedding(4001,64, input_length=20)(main_input)        
x = LSTM(64, dropout=0.2, recurrent_dropout=0.2)(x)        
main_output1 = Dense(8, activation='sigmoid')(x)
model1 = Model(inputs = main_input, outputs=main_output1)
model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  
model1.fit(data[:6000], to_categorical(y)[:6000], epochs=20)

x1 = Embedding(4001,64, input_length=20)(main_input)        
x1 = LSTM(64, dropout=0.2, recurrent_dropout=0.2)(x1)        
x1 = concatenate([x1,auxiliary_input])
main_output2 = Dense(8, activation='sigmoid')(x1)
model2 = Model(inputs = [main_input,auxiliary_input], outputs=main_output2)      
model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  
model2.fit([data[:6000],postag_hist[:6000]], to_categorical(y)[:6000], epochs=15)        

model1.test_on_batch(data[6000:8000],to_categorical(y)[6000:8000])
model2.test_on_batch([data[6000:8000],postag_hist[6000:8000]],to_categorical(y)[6000:8000])

