# Script to get various model statistics

import numpy as np
import pickle

ROOT = '/home/ninad/Desktop/NlpProj/'

en_vocab = {'ADJ', 'ADP', 'ADV', 'CONJ', 'DET', 'NOUN', 'NUM', 'PART', 'PRON', 
            '.','VERB', 'X'}

enInFile = ROOT + 'code/cmatrix/enTagList.pkl'
with open(enInFile, 'rb') as enFile:
    entags = pickle.load(enFile)

hiInFile = ROOT + 'code/cmatrix/hiTagList.pkl'
with open(hiInFile, 'rb') as hiFile:
    hitags = pickle.load(hiFile)

count1=0
count2=0
for i in range(len(hitags)):
    if(hitags[i]!='\n'):
        count2+=1
        hw = hitags[i][-1]
        ew = entags[i][-1]

        if(hw == 'AUX'):
            hw='VERB'
        elif(hw=='CCONJ' or hw=='SCONJ'):
            hw='CONJ'
        elif(hw=='PROPN'):
            hw='NOUN'
        elif(hw=='PUNCT'):
            hw='.'
        
        if(hw == ew):
            count1+=1


        #~ else:
            #~ print(hitags[i],entags[i])
            #~ print(hw == ew)

acc = count1/count2
print("ACCURACY: ", acc)



def prec_recall(tag, hitags, entags):
    tp=0
    fp=0
    tn=0
    fn=0
    supp=0
    for i in range(len(hitags)):
        if(hitags[i]!='\n'):
            hw = hitags[i][-1]
            ew = entags[i][-1]
            if(hw == 'AUX'):
                hw='VERB'
            elif(hw=='CCONJ' or hw=='SCONJ'):
                hw='CONJ'
            elif(hw=='PROPN'):
                hw='NOUN'
            elif(hw=='PUNCT'):
                hw='.'        

            if(hw == tag):
                if(hw == ew):
                    tp+=1
                elif(hw != ew):
                    fn+=1
            elif(hw != tag):
                if(ew == tag):
                    fp+=1
                elif(ew != tag):
                    tn+=1

            if(ew==tag): #support
                supp+=1

            
    eps = 1e-8
    prec = (tp+eps)/(tp+fp+eps)
    recl = (tp+eps)/(tp+fn+eps)
    f1score = (2*prec*recl)/(prec+recl)    
    return (prec, recl,f1score,supp)        

tot_supp = 0
for tag in en_vocab:
    tup = prec_recall(tag, hitags, entags)
    print(tag, tup)
    tot_supp += tup[-1]

print('Sum of all supports: ', tot_supp)

