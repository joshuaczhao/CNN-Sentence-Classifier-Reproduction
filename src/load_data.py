import data_helpers
from gensim.models import KeyedVectors


def load_MR_data():

    data, labels = data_helpers.load_data_and_labels('data/MR/rt-polarity.pos', 'data/MR/rt-polarity.neg')
    print('Data file loaded')

    # Tokenize
    data = [sentence.split(' ') for sentence in data]
    max_sen_len = len(max(data, key=len))
    print('Tokenized')

    # Load embedding model
    goog_w2v_path = 'data/embedding_models/GoogleNews-vectors-negative300.bin'
    model = KeyedVectors.load_word2vec_format(goog_w2v_path, binary=True)
    print("Embeddings model loaded. vocabulary size = ", len(model.vocab), 'vocab shape = ', model.vectors.shape)

    # Pad lines, Sub out-of-vocab words, Get model indices
    data = pad(data)
    for i, sentence in enumerate(data):

        # sub unknown tokens
        for j, token in enumerate(sentence):
            if token in model.vocab:
                sentence[j] = model.vocab[token].index
            elif token == '<pad>':
                sentence[j] = 1000000
            else:
                sentence[j] = 2000000

        data[i] = sentence

        if i % 1000 == 0:
            print(f'{i}/{len(data)}')

    print('Preprocessing complete')

    return data, labels, max_sen_len

def pad(data, max_sen_len):
    for i in range(len(data)):
        sentence = data[i]
        if len(sentence) < max_sen_len:
            diff = max_sen_len - len(sentence)
            sentence.extend(['<pad>' for x in range(diff)])
        data[i] = sentence
    return data