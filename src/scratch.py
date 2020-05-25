import gensim
from gensim.test.utils import datapath
from gensim.models import KeyedVectors
from gensim.models.keyedvectors import KeyedVectors
import gensim.downloader as api
import torch
import data_helpers
import os
import numpy as np

import torchtext
import torchtext.data as data
from torchtext.vocab import GloVe
import torchtext.datasets as datasets
from torchtext.data import get_tokenizer

import model


x_text, y = data_helpers.load_data_and_labels('data/MR/rt-polarity.pos', 'data/MR/rt-polarity.neg')
# print(len(x_text), len(y))

review_data = x_text[0:2]
tokenizer = get_tokenizer("basic_english")
review_data = [tokenizer(line) for line in review_data]
labels = y[0:10]

TEXT = data.Field(sequential=True, use_vocab=True, fix_length=None, lower=True)
glove_path = 'data/embedding_models/glove.twitter.27B/glove.twitter.27B.50d.txt'
path = 'data/embedding_models/GoogleNews-vectors-negative300.bin'
w2v_path = 'data/embedding_models/GoogleNews-w2v_format.txt'
vectors = torchtext.vocab.Vectors(name=w2v_path, cache=None)
# vectors = torchtext.vocab.GloVe(name='6B', dim=50)
TEXT.build_vocab(review_data, max_size=None, min_freq=1, vectors=vectors)
processsed_data = TEXT.process(review_data)
print(processsed_data)
print("Output data shape", processsed_data.shape)
print("Vocab tensor shape", TEXT.vocab.vectors.size())
print(len(vectors.stoi.keys()))
print(TEXT.vocab.stoi)


embedding = torch.nn.Embedding.from_pretrained(TEXT.vocab.vectors)

# input = [batch_size, in_channels, max_sentence_len, embedding_dim]
# x = processsed_data
input = torch.LongTensor([[a for a in range(40)], [a for a in range(40)]])
cnn = model.Net(E_DIMS=25)
out = cnn.forward(input)
print(out)

# cleaned_tokens = [data_helpers.clean_str(line).split() for line in data]
# print(np.shape(cleaned_tokens))
# print(cleaned_tokens[0])
#
#
# ''' # Convert glove embedding file to gensim compatible
# from gensim.scripts.glove2word2vec import glove2word2vec
# # glove2word2vec(glove_input_file=path, word2vec_output_file=path2)
# '''
#
# Load embedding model
# path = 'data/embedding_models/GoogleNews-vectors-negative300.bin'
# save_path = 'data/embedding_models/GoogleNews-w2v_format.txt'
# # path = 'data/embedding_models/glove.twitter.27B/converted_25d.txt'
# model = KeyedVectors.load_word2vec_format(save_path, binary=True)
#
# print("Embeddings model loaded. vocabulary size = ", len(model.vocab))
#
# model.wv.save_word2vec_format(save_path)
#
#
# # print(cleaned_tokens[1])
# # print(model[cleaned_tokens[1]])
# # indices = [model[line] for line in cleaned_tokens]
# # print(indices.shape, indices[0])
#
#
# weights = torch.FloatTensor(model.vectors)
# embedding = torch.nn.Embedding.from_pretrained(weights)
# indices = [model.vocab[word].index for word in cleaned_tokens[1]]
# print(indices)
# output = embedding(torch.LongTensor(indices))
# print(output)




