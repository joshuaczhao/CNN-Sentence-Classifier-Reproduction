import torch
import torch.nn as nn
import torch.nn.functional as F

from gensim.models import KeyedVectors

class Net(nn.Module):
    def __init__(self, EMBEDDINGS, BATCH_SIZE, M_TYPE='NOT_STATIC', N_KERN=100, E_DIMS=300, E_NUMB=662109, DROP=0, C_SIZE=2, TRAIN=True, M_SENT_LEN=40):
        super(Net, self).__init__()

        # Need to update the way arguments are passed into this, C_SIZE, M_SENT_LEN are incorrect values
        self.EMBEDDINGS = EMBEDDINGS
        self.BATCH_SIZE = BATCH_SIZE
        self.M_TYPE = M_TYPE
        self.E_DIMS = E_DIMS
        self.E_NUMB = E_NUMB
        self.N_KERN = N_KERN
        self.DROP   = DROP
        self.C_SIZE = C_SIZE
        self.TRAIN  = TRAIN
        self.M_S_L  = M_SENT_LEN
        self.NUM_INPUT_C = 1

        # path = 'data/embedding_models/GoogleNews-vectors-negative300.bin'
        # embedding_model = KeyedVectors.load_word2vec_format(path, binary=True)
        # weights = torch.FloatTensor(embedding_model.vectors)

        vocab_size = len(EMBEDDINGS)
        print(f'VOCAB SIZE = {vocab_size}')
        weights = torch.FloatTensor(EMBEDDINGS)

        if M_TYPE == 'RANDOM':
            self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=E_DIMS, padding_idx=0)
        elif M_TYPE == 'NOT_STATIC':
            self.embedding = nn.Embedding.from_pretrained(weights, freeze=False, padding_idx=0)
        elif M_TYPE == 'STATIC':
            self.embedding = nn.Embedding.from_pretrained(weights, freeze=True, padding_idx=0)
        elif M_TYPE == 'MULTI':
            self.embedding = nn.Embedding.from_pretrained(weights, freeze=False, padding_idx=0)
            self.embedding_static = nn.Embedding.from_pretrained(weights, freeze=True, padding_idx=0)
            self.NUM_INPUT_C = 2
        else:
            print('Invalid M_TYPE')
            return

        '''
        # For testing
        self.embedding = nn.Embedding(E_DIMS, E_NUMB)
        '''
        # nn.Conv2d(n_inp, n_outp, kern_size, stride)
        self.conv1 = nn.Conv2d(self.NUM_INPUT_C, N_KERN, (3, E_DIMS), 1)
        self.conv2 = nn.Conv2d(self.NUM_INPUT_C, N_KERN, (4, E_DIMS), 1)
        self.conv3 = nn.Conv2d(self.NUM_INPUT_C, N_KERN, (5, E_DIMS), 1)
        self.dropout = nn.Dropout(DROP)
        self.fc1   = nn.Linear(3*N_KERN, C_SIZE)       
        
    def forward(self, x):

        
        if self.M_TYPE == 'MULTI':
            x_s = self.embedding_static(x)
            x_s = x_s.view((-1, 1, self.M_S_L, self.E_DIMS))
            x = self.embedding(x)
            x = x.view((-1, 1, self.M_S_L, self.E_DIMS))
            x = torch.cat((x_s, x), dim=1) # Concatenate along "Cin" dimension
        else:
            x = self.embedding(x)
            x = x.view((-1, 1, self.M_S_L, self.E_DIMS))

        '''
        if self.TYPE == 'STATIC':
            x = torch.autograd.Variable(x)
        '''

        x_f1 = F.max_pool1d(F.relu(self.conv1(x)).squeeze(3), self.M_S_L - 3)
        x_f2 = F.max_pool1d(F.relu(self.conv2(x)).squeeze(3), self.M_S_L - 3)
        x_f3 = F.max_pool1d(F.relu(self.conv3(x)).squeeze(3), self.M_S_L - 4)
        x = torch.cat((x_f1, x_f2, x_f3), 1)
        
        x = self.dropout(x)
        x = self.fc1(x.squeeze(2))
        
        return x
