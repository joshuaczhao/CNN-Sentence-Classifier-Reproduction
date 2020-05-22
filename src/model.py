import torch
import torch.nn as nn
import torch.nn.functional as F

from gensim.models import KeyedVectors

class Net(nn.Module):
    def __init__(self, M_TYPE='STATIC', N_KERN=100, E_DIMS=300, E_NUMB=662109, DROP=0, C_SIZE=5, TRAIN=True, M_SENT_LEN=5):
        super(Net, self).__init__()

        # Defaults for Google Word2Vec

        # Need to update the way arguments are passed into this, C_NUM, M_SENT_LEN are incorrect values
        self.M_TYPE = M_TYPE
        self.E_DIMS = E_DIMS
        self.E_NUMB = E_NUMB
        self.N_KERN = N_KERN
        self.DROP   = DROP
        self.C_SIZE = C_SIZE
        self.TRAIN  = TRAIN
        self.M_S_L  = M_SENT_LEN


        # Don't download if just testing the model, embeddings take up lots of space
        path = 'data/embedding_models/GoogleNews-vectors-negative300.bin'
        # path = 'data/embedding_models/glove.twitter.27B/converted_25d.txt'
        model = KeyedVectors.load_word2vec_format(path)
        weights = torch.FloatTensor(model.vectors)
        self.embedding = nn.Embedding.from_pretrianed(weights)
        
        if TYPE == 'STATIC':
            self.embedding.weight.requires_grad = False  

        '''
        # For testing
        self.embedding = nn.Embedding(E_DIMS, E_NUMB)
        '''
        # nn.Conv2d(n_inp, n_outp, kern_size, stride)
        self.conv1 = nn.Conv2d(1, N_KERN, (3, E_DIMS), E_DIMS)
        self.conv2 = nn.Conv2d(1, N_KERN, (4, E_DIMS), E_DIMS)
        self.conv3 = nn.Conv2d(1, N_KERN, (5, E_DIMS), E_DIMS)
        self.fc1   = nn.Linear(3*N_KERN, C_SIZE)       
        
    def forward(self, x):

        x = self.embedding(x).view((1 , -1))
        '''
        if self.TYPE == 'STATIC':
            x = torch.autograd.Variable(x)
        '''
        
        x_f1 = F.max_pool1d(F.relu(self.conv1(x)), self.M_S_L - 4)
        x_f2 = F.max_pool1d(F.relu(self.conv1(x)), self.M_S_L - 5)
        x_f3 = F.max_pool1d(F.relu(self.conv1(x)), self.M_S_L - 6)   
        x = torch.cat((x_f1, x_f2, x_f3), 1)
        
        x = F.dropout(x, self.DROP, self.TRAIN)        
        x - self.fc1(x)
        
        return x
