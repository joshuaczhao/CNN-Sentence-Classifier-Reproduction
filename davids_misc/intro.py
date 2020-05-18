import torch
import numpy as np


import torch.nn as nn
import torch.nn.functional as F


print('Hello World')


from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset

import data_helpers



class CustomIterableDataset(IterableDataset):

    def __init__(self, filename):
        #Store the filename in object's memory
        self.filename = filename

    def __iter__(self):
        #Create an iterator
        file_itr = open(self.filename)
        return file_itr

#Create a dataset object
dataset = CustomIterableDataset('data/rt-polaritydata/rt-polarity.neg')

#Wrap it around a dataloader
dataloader = DataLoader(dataset, batch_size=64)

import gensim

model = gensim.models.KeyedVectors.load_word2vec_format('data/GoogleGoogleNews-vectors-negative300.bin', binary=True)
weights = model.wv


# Declare model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()


# count = 0
# for data in dataloader:
#
#     count += len(data)
#     outputs = net(data)
#     print(outputs)
#
# print(count)



import torch.optim as optim

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


# Training
for epoch in range(1):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs = data[0]
        labels = np.ones(len(inputs))

        print(inputs)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
#
#
# # save trained model
# PATH = './cifar_net.pth'
# torch.save(net.state_dict(), PATH)
#
#
#
# # Testing
# dataiter = iter(testloader)
# images, labels = dataiter.next()
#
# # print images
# imshow(torchvision.utils.make_grid(images))
# print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))










# if __name__ == "__main__":
#     main()
