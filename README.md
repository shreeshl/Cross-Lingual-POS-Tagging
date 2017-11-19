# Cross-Lingual-POS-Tagging

Project files will go here.

Detailed documentation will be up shortly!


## Notes

- cmatrix_results contains the results of tagging from the count_matrix method. It
has three files:

- The file enWordList.pkl is the pickled file for the raw word level translation
obtianed for each hindi word.

- The file enTagList.pkl is pickled file for sentences and their corresponding
POS tags. Since there were 1660 sentences (counted using number of newline chars)
, this list has 3320 entries. The even numbered indices (starting from 0) have a 
sentence with corresponding POS tags. The odd numbered indices are just the newline
chars (for keeping the separation similar to source hindi POS tagged file)

- The code alingn_bilingual.py contains the function definitions along with the 
code for tagging (using nltk pos tagger) and saving the tagged lists.

- **Imp** the output vocab for hindi is different: OUTPUT_VOCAB = {'ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'NOUN', 'NUM',
                'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'VERB', 'X'}
