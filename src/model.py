import torch
import torch.nn as nn
import torch.nn.functional as F

from gensim.models import KeyedVectors

class Net(nn.Module):
    def __init__(self, BATCH_SIZE, M_TYPE='NOT_STATIC', N_KERN=100, E_DIMS=300, E_NUMB=662109, DROP=0, C_SIZE=2, TRAIN=True, M_SENT_LEN=40):
        super(Net, self).__init__()

        # Need to update the way arguments are passed into this, C_SIZE, M_SENT_LEN are incorrect values
        self.BATCH_SIZE = BATCH_SIZE
        self.M_TYPE = M_TYPE
        self.E_DIMS = E_DIMS
        self.E_NUMB = E_NUMB
        self.N_KERN = N_KERN
        self.DROP   = DROP
        self.C_SIZE = C_SIZE
        self.TRAIN  = TRAIN
        self.M_S_L  = M_SENT_LEN

        path = 'data/embedding_models/GoogleNews-vectors-negative300.bin'
        embedding_model = KeyedVectors.load_word2vec_format(path, binary=True)
        weights = torch.FloatTensor(embedding_model.vectors)

        if M_TYPE == 'RANDOM':
            vocabulary_size = int(3e6)
            self.embedding = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=E_DIMS)
        elif M_TYPE == 'NOT_STATIC':
            self.embedding = nn.Embedding.from_pretrained(weights)
        elif M_TYPE == 'STATIC':
            self.embedding = nn.Embedding.from_pretrained(weights)
            self.embedding.weight.requires_grad = False  
        else:
            print('Invalid M_TYPE')
            return

        '''
        # For testing
        self.embedding = nn.Embedding(E_DIMS, E_NUMB)
        '''
        # nn.Conv2d(n_inp, n_outp, kern_size, stride)
        self.conv1 = nn.Conv2d(1, N_KERN, (3, E_DIMS), 1)
        self.conv2 = nn.Conv2d(1, N_KERN, (4, E_DIMS), 1)
        self.conv3 = nn.Conv2d(1, N_KERN, (5, E_DIMS), 1)
        self.dropout = nn.Dropout(DROP)
        self.fc1   = nn.Linear(3*N_KERN, C_SIZE)       
        
    def forward(self, x):

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
        
        # x = F.dropout(x, p=self.DROP, train=self.TRAIN)
        x = self.dropout(x)
        x = self.fc1(x.squeeze(2))
        
        return x
