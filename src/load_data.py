import data_helpers
from gensim.models import KeyedVectors


def load_MR_data():
    '''
    Returns:
        data - tokenized, padded, indices of dataset as 2D list w/dimension [n_samples, max_sentence_length]
        labels - 1D list of integer labels in range [0, n_classes)
        max_sen_len - the maximum length sample in the dataset after word-tokenization
    '''

    data, labels = data_helpers.load_data_and_labels('data/MR/rt-polarity.pos', 'data/MR/rt-polarity.neg')

    data = tokenize(data)
    max_sen_len = len(max(data, key=len))

    data = pad(data, max_sen_len)
    data = get_indices(data)

    return data, labels, max_sen_len


def tokenize(data):
    data = [sentence.split(' ') for sentence in data]
    return data


def pad(data, max_sen_len):
    for i in range(len(data)):
        sentence = data[i]
        if len(sentence) < max_sen_len:
            diff = max_sen_len - len(sentence)
            sentence.extend(['<pad>' for x in range(diff)])
        data[i] = sentence
    return data


def get_indices(data):
    # Load embedding model
    goog_w2v_path = 'data/embedding_models/GoogleNews-vectors-negative300.bin'
    model = KeyedVectors.load_word2vec_format(goog_w2v_path, binary=True)

    for i, sentence in enumerate(data):
        for j, token in enumerate(sentence):
            if token in model.vocab:
                sentence[j] = model.vocab[token].index
            elif token == '<pad>':
                sentence[j] = 1000000
            else:
                sentence[j] = 2000000
        data[i] = sentence
    return data
