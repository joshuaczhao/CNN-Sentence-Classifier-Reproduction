import gensim
from gensim.test.utils import datapath
from gensim.models import KeyedVectors
from gensim.models.keyedvectors import KeyedVectors
import gensim.downloader as api
import torch
import data_helpers

x_text, y = data_helpers.load_data_and_labels('data/rt-polaritydata/rt-polarity.pos', 'data/rt-polaritydata/rt-polarity.neg')
print(len(x_text), len(y))
print(x_text[0])
print(y[0])


# path = 'data/GoogleNews-vectors-negative300.bin'
path = 'data/glove.twitter.27B/converted_25d.txt'

# from gensim.scripts.glove2word2vec import glove2word2vec
# glove2word2vec(glove_input_file=path, word2vec_output_file=path2)
model = KeyedVectors.load_word2vec_format(path)
print("Vocabulary size = ", len(model.vocab))


sentence = "Hello there my Friend  Bob."
cleaned_tokens = data_helpers.clean_str(sentence).split()
print(type(clean), clean)

vector = model[sentence]
print(vector.shape, vector)

# weights = torch.FloatTensor(model.vectors)
# embedding = torch.nn.Embedding.from_pretrained(weights)
# indices = [model.vocab[word].index for word in sentence]
# input = torch.LongTensor(indices)
# print(input)
# print(embedding(input))




