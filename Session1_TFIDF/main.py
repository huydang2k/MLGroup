# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.




from os import listdir
from os.path import isfile
import re
import numpy as np

# Press the green button in the gutter to run the script.
def get_tf_idf(data_path):
    with open('../datasets/20news-bydate/words_idfs.txt') as f:
        words_idfs = [(line.split('<fff>')[0],float(line.split('<fff>')[1])) for line in f.read().splitlines()]
        word_IDs = dict([(word,index) for index,(word,idf) in enumerate(words_idfs)])
        idfs = dict(words_idfs)
    with open(data_path) as f:
        documents = [(int(line.split('<fff>')[0]),int(line.split('<fff>')[1]),line.split('<fff>')[2]) for line in f.read().splitlines()]
    data_td_idf = []
    for document in documents:
        label,doc_id,text = document
        words = [word for word in text.split() if word in idfs]
        word_set = list(set(words))
        max_term_freq = max([words.count(word) for word in word_set])
        words_tfidfs = []
        sum_squares = 0.0
        for word in word_set:
            term_freq = words.count(word)
            tf_idf_value = term_freq*1./max_term_freq*idfs[word]
            words_tfidfs.append((word_IDs[word],tf_idf_value))
            sum_squares += tf_idf_value ** 2
        words_tfidfs_normalized = [str(index) + ':' + str(tf_idf_value/np.sqrt(sum_squares)) for index, tf_idf_value in words_tfidfs]
        sparse_rep = ' '.join(words_tfidfs_normalized)
        data_td_idf.append((label,doc_id,sparse_rep))
    return data_td_idf
#build vocab
def generate_vocabulary(data_path):
    from collections import defaultdict
    def compute_idf(df, corpus_size):
        assert df > 0
        return np.log10(corpus_size * 1. / df)

    with open(data_path) as f:
        lines = f.read().splitlines()
    doc_count = defaultdict(int)
    corpus_size = len(lines)
    for line in lines:
        features = line.split('<fff>')
        text = features[-1]
        words = list(set(text.split()))
        for word in words:
            doc_count[word] += 1
    word_idfs = [(word, compute_idf(document_frequency, corpus_size)) for word, document_frequency in
                 zip(doc_count.keys(), doc_count.values()) if document_frequency > 10 and not word.isdigit()]
    word_idfs.sort(key=lambda tup: - tup[1])
    print("vocab size: {}".format(len(word_idfs)))
    with open('../datasets/20news-bydate/words_idfs.txt', 'w') as f:
        f.write('\n'.join([word + '<fff>' + str(idf) for word, idf in word_idfs]))
def collect_data(parent_dir, newsgroup_list):
    data = []
    for group_id, newsgroup in enumerate(newsgroup_list):
        label = group_id
        dir_path = parent_dir + '/' + newsgroup + '/'
        files = [(filename, dir_path + filename) for filename in listdir(dir_path) if isfile(dir_path + filename)]
        files.sort()
        for filename, filepath in files:
            with open(filepath) as f:
                text = f.read().lower()
                words = [stemmer.stem(word) for word in re.split('\W+', text) if word not in stop_words]
                content = ' '.join(words)
                assert len(content.splitlines()) == 1
                data.append(str(label) + '<fff>' + filename + '<fff>' + content)
    return data


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # gather 20 groups name
    path = '../datasets/20news-bydate/'
    dirs = [path + dir_name + '/' for dir_name in listdir(path) if not isfile(path + dir_name)]
    train_dir, test_dir = (dirs[0], dirs[1]) if 'train' in dirs[0] else (dirs[1], dirs[0])
    list_newsgroup = [newsgroup for newsgroup in listdir(train_dir)]
    list_newsgroup.sort()

    with open('../datasets/20news-bydate/stop_words.txt') as f:
        stop_words = f.read().splitlines()
    from nltk.stem.porter import PorterStemmer

    stemmer = PorterStemmer()

    # collect data
    train_data = collect_data(parent_dir=train_dir, newsgroup_list=list_newsgroup)
    test_data = collect_data(parent_dir=test_dir, newsgroup_list=list_newsgroup)

    full_data = train_data + test_data
    with open('../datasets/20news-bydate/20news-train-processed.txt', 'w') as f:
        f.write('\n'.join(train_data))
    with open('../datasets/20news-bydate/20news-test-processed.txt', 'w') as f:
        f.write('\n'.join(test_data))
    with open('../datasets/20news-bydate/20news-full-processed.txt', 'w') as f:
        f.write('\n'.join(full_data))

    # build idf data
    generate_vocabulary('../datasets/20news-bydate/20news-full-processed.txt')

    a = get_tf_idf('../datasets/20news-bydate/20news-train-processed.txt')
    with open('../datasets/20news-bydate/20news-train-tf-idf.txt', 'w') as f:
        f.write('\n'.join([str(cur[0]) + '<fff>' + str(cur[1]) + '<fff>' + str(cur[2]) for cur in a]))

    b = get_tf_idf('../datasets/20news-bydate/20news-test-processed.txt')
    with open('../datasets/20news-bydate/20news-test-tf-idf.txt', 'w') as f:
        f.write('\n'.join([str(cur[0]) + '<fff>' + str(cur[1]) + '<fff>' + str(cur[2]) for cur in b]))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
