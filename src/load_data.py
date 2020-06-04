import numpy as np
import re
from gensim.models import KeyedVectors


def load_MR_data():
    """
    Returns:
        data - tokenized, padded, indices of MR data samples as 2D list w/dimension [n_samples, max_sentence_length]
        labels - 1D list of integer labels in range [0, n_classes)
        max_sen_len - the maximum length sample in the dataset after word-tokenization
    """

    data, labels = load_data_and_labels('data/MR/rt-polarity.pos', 'data/MR/rt-polarity.neg')

    data, max_sen_len = tokenize(data)
    data = pad(data, max_sen_len)
    data, weights = get_indices(data)

    n_classes = len(np.unique(labels))

    return data, labels, max_sen_len, n_classes, weights


def load_subj_data(max_length=40):
    """
        Label Convention: subjective/quote = 0, objective/plot = 1
    """

    path1 = 'data/Subj/plot.tok.gt9.5000'
    path2 = 'data/Subj/quote.tok.gt9.5000'

    # Load data from files
    objective_examples = list(open(path1, "r", encoding='ISO-8859-1').readlines())
    subjective_examples = list(open(path2, "r", encoding="ISO-8859-1").readlines())
    # Split by words
    data = [clean_str(s.strip()) for s in objective_examples + subjective_examples]
    labels = [1 for _ in objective_examples] + [0 for _ in subjective_examples]

    data, max_sen_len = tokenize(data)
    data = [s[:max_length] for s in data]
    data = pad(data, max_sen_len)
    data = get_indices(data)

    n_classes = len(np.unique(labels))

    return data, labels, max_sen_len, n_classes


def load_TREC_data():
    """
        Label Convention: Description = 0, Entity = 1, Abbreviation = 2, Human = 3, Number = 4, Location = 5"
    """

    ### load TRAIN FILE ###
    path = "data/TREC/TrainTREC.txt"

    line_list = list(open(path, "r").readlines())
    label_ids = {"DESC:": 0, "ENTY:": 1, "ABBR:": 2, "HUM:": 3, "NUM:": 4, "LOC:": 5}
    train_data = []
    train_labels = []

    for line in line_list:
        for label in label_ids:
            if label in line:
                # Append label number to labels
                train_labels.append(label_ids[label])
                remove = ""
                for i in line:
                    if i == " ":
                        break
                    remove += i
                # Remove the label from the front of the string
                train_data.append(line.replace(remove + " ", "")[:-1])

    ### load TEST FILE ###
    path = "data/TREC/TestTREC.txt"

    line_list = list(open(path, "r").readlines())
    label_ids = {"DESC:": 0, "ENTY:": 1, "ABBR:": 2, "HUM:": 3, "NUM:": 4, "LOC:": 5}
    test_data = []
    test_labels = []

    for line in line_list:
        for label in label_ids:
            if label in line:
                # Append label number to labels
                test_labels.append(label_ids[label])
                remove = ""
                for i in line:
                    if i == " ":
                        break
                    remove += i
                # Remove the label from the front of the string
                test_data.append(line.replace(remove + " ", "")[:-1])

    train_data = [clean_str(s.strip()) for s in train_data]
    test_data = [clean_str(s.strip()) for s in test_data]
    train_data, train_max_sen_len = tokenize(train_data)
    test_data, test_max_sen_len = tokenize(test_data)

    max_sen_len = max(train_max_sen_len, test_max_sen_len)

    train_data = pad(train_data, max_sen_len)
    test_data = pad(test_data, max_sen_len)
    train_data = get_indices(train_data)
    test_data = get_indices(test_data)

    n_classes = len(np.unique(train_labels))

    return train_data, train_labels, test_data, test_labels, max_sen_len, n_classes


def tokenize(data):
    data = [sentence.split(' ') for sentence in data]
    max_sentence = len(max(data, key=len))
    return data, max_sentence


def pad(data, max_sen_len):
    for i in range(len(data)):
        sentence = data[i]
        if len(sentence) < max_sen_len:
            diff = max_sen_len - len(sentence)
            sentence.extend(['<pad>' for _ in range(diff)])
        data[i] = sentence
    return data


def get_indices(data):

    goog_w2v_path = 'data/embedding_models/GoogleNews-vectors-negative300.bin'
    model = KeyedVectors.load_word2vec_format(goog_w2v_path, binary=True)
    vocab_dict = {}
    weights = [model['pad'], model['unk']]
    for i, sentence in enumerate(data):
        for j, token in enumerate(sentence):

            if token in vocab_dict:
                sentence[j] = vocab_dict[token]
            else:
                if token in model.vocab:
                    weights.append(model[token])
                    vocab_dict[token] = len(weights) - 1
                    sentence[j] = len(weights) - 1
                elif token == '<pad>':
                    sentence[j] = 0
                else:
                    sentence[j] = 1
        data[i] = sentence

    return data, weights


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r", encoding='utf-8').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r", encoding='utf-8').readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [1 for _ in positive_examples]
    negative_labels = [0 for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


# data, labels, max_sen_len = load_TREC_data(set='train')
# print(np.shape(data), np.shape(labels))
# print(data[0:5])
# print(labels[0:5])
