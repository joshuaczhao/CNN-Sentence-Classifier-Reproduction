import data_helpers
from gensim.models import KeyedVectors


x_text, y = data_helpers.load_data_and_labels('data/MR/rt-polarity.pos', 'data/MR/rt-polarity.neg')

data = x_text[0:5]


# Tokenize
data = [sentence.split(' ') for sentence in data]
max_sen_len = len(max(data, key=len))

# Load embedding model
goog_w2v_path = 'data/embedding_models/GoogleNews-vectors-negative300.bin'
glove_path = 'data/embedding_models/glove.twitter.27B/converted_25d.txt'
model = KeyedVectors.load_word2vec_format(goog_w2v_path, binary=True)
print("Embeddings model loaded. vocabulary size = ", len(model.vocab))


for i, sentence in enumerate(data):
    print(sentence)
    new_sentence = sentence
    # pad
    if len(new_sentence) < max_sen_len:
        diff = max_sen_len - len(sentence)
        new_sentence.extend(['<pad>' for x in range(diff)])

    # sub unknown tokens
    for j, token in enumerate(new_sentence):

        # if token in ['the']:
        if True:
            new_sentence[j] = model.vocab[token].index
        else:
            new_sentence[j] = '<unk>'


    data[i] = new_sentence
    print(new_sentence)

