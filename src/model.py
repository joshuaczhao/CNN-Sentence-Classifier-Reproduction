import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Embedding layer
        path = 'data/embedding_models/GoogleNews-vectors-negative300.bin'
        # path = 'data/embedding_models/glove.twitter.27B/converted_25d.txt'
        model = KeyedVectors.load_word2vec_format(path)
        weights = torch.FloatTensor(model.vectors)
        self.embedding = nn.Embedding.from_pretrianed(weights)

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):

        # get word embeddings
        x = self.embedding(x)

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x