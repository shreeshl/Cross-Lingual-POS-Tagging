import codecs

hindi_english_corpus = "/Users/shreesh/Academics/CS585/Project/HindEnCorp 0.5/hindencorp05.plaintext"
f = codecs.open(hindi_english_corpus, 'r', 'utf-8').read().split("\n")
g = codecs.open('fast_align_format_hi.txt', 'w', 'utf-8')
for sentence in f:
    sentence = sentence.split('\t')
    if len(sentence)!=5: continue
    g.write(sentence[3]+ " ||| " + sentence[4]+"\n")
g.close()
