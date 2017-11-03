import codecs
train_file = '/Users/shreesh/Academics/CS585/Project/Universal Dependencies 2.0/ud-treebanks-conll2017/UD_Hindi/hi-ud-train.conllu'
test_file = '/Users/shreesh/Academics/CS585/Project/Universal Dependencies 2.0/ud-treebanks-conll2017/UD_Hindi/hi-ud-dev.conllu'

train_file_ta = '/Users/shreesh/Academics/CS585/Project/Universal Dependencies 2.0/ud-treebanks-conll2017/UD_Tamil/ta-ud-train.conllu'
test_file_ta = '/Users/shreesh/Academics/CS585/Project/Universal Dependencies 2.0/ud-treebanks-conll2017/UD_Tamil/ta-ud-dev.conllu'

def save_data(file_name, file_location):
    data = codecs.open(file_location, 'r', 'utf-8').read().split("\n\n")
    clean_file = codecs.open(file_name, 'w', 'utf-8')
    for item in data:
        item = item.split("\n")
        for sentence in item[2:]:
            sentence = sentence.split("\t")
            word, tag = sentence[1], sentence[3]
            clean_file.write(word+"\t"+tag+"\n")
        clean_file.write("\n")
    clean_file.close()

def save_data_tamil(file_name, file_location):
    data = codecs.open(file_location, 'r', 'utf-8').read().split("\n\n")
    clean_file = codecs.open(file_name, 'w', 'utf-8')
    for item in data:
        item = item.split("\n")
        for sentence in item[4:]:
            sentence = sentence.split("\t")
            word, tag = sentence[1], sentence[3]
            clean_file.write(word+"\t"+tag+"\n")
        clean_file.write("\n")
    clean_file.close()

#save_data("parsed_train.txt", train_file)
#save_data("parsed_test.txt", test_file)
save_data_tamil("parsed_train.txt", train_file_ta)
save_data_tamil("parsed_test.txt", test_file_ta)