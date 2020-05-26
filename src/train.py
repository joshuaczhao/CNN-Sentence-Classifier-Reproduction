import torch
import torch.nn as nn
import numpy as np
import time
import datetime
from sklearn.model_selection import train_test_split

import load_data
import model

start = time.time()

BATCH = 50
N_EPOCHS = 5
DROPOUT_RATE = 0.5
MODEL_TYPE = 'NON STATIC'

data, labels, max_sen_len = load_data.load_MR_data()

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, shuffle=True, stratify=labels)

x_train_tensor = torch.LongTensor(x_train)
y_train_tensor = torch.LongTensor(y_train)
x_test_tensor = torch.LongTensor(x_test)
y_test_tensor = torch.LongTensor(y_test)

train_dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH, shuffle=True)

cnn = model.Net(BATCH_SIZE=BATCH, M_TYPE=MODEL_TYPE, E_DIMS=300, M_SENT_LEN=max_sen_len, DROP=DROPOUT_RATE)

criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(cnn.parameters(), lr=0.001, momentum=0.9)
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)

name = 'outputs/' + datetime.datetime.now().strftime("%m-%d-%Y_%H_%M") + '.txt'
f = open(name, "a")

print(f'EPOCHS = {N_EPOCHS}; BATCH = {BATCH} \n', file=f)

train_loss_history = []
train_acc_history = []
valid_loss_history = []
valid_acc_history = []
for epoch in range(N_EPOCHS):  # loop over the dataset multiple times
    print(f'===== EPOCH {epoch + 1}/{N_EPOCHS} =====', file=f)
    print(f'===== EPOCH {epoch + 1}/{N_EPOCHS} =====')
    one_epoch_acc_history = []
    one_epoch_loss_history = []

    cnn.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = cnn(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # calculate mini-batch training accuracy
        predictions = torch.argmax(outputs, dim=1).detach()
        acc = (predictions == labels).sum().item() / len(labels)
        one_epoch_acc_history.append(acc)

        # print mini-batch statistics
        minibatch_loss = loss.item()
        running_loss += minibatch_loss
        one_epoch_loss_history.append(minibatch_loss)
        print_frequency = 1
        if i % print_frequency == print_frequency - 1:    # print every mini-batch
            # print('[%d, %3d] train loss: %.4f train acc: %.4f' % (epoch + 1, i + 1, running_loss / print_frequency, acc), file=f)
            print('[%d, %3d] train loss: %.4f train acc: %.4f' % (epoch + 1, i + 1, running_loss / print_frequency, acc))
            running_loss = 0.0

    train_loss = np.mean(one_epoch_loss_history)
    train_acc = np.mean(one_epoch_acc_history)
    print(f'EPOCH AVG TRAIN ACC: {train_acc:.4f}', file=f)
    print(f'EPOCH AVG TRAIN ACC: {train_acc:.4f}')

    # run validation data
    cnn.eval()
    outputs = cnn(x_test_tensor)
    valid_loss = criterion(outputs, y_test_tensor).item()
    valid_predictions = torch.argmax(outputs, dim=1).detach()
    valid_n_correct = (valid_predictions == y_test_tensor).sum().item()
    valid_acc = valid_n_correct / len(y_test_tensor)
    print(f'VALID LOSS: {valid_loss:.4f} VALID ACC: {valid_acc:.4f}', file=f)
    print(f'VALID LOSS: {valid_loss:.4f} VALID ACC: {valid_acc:.4f}')

    # save average loss, acc for this epoch
    train_loss_history.append(train_loss)
    train_acc_history.append(train_acc)
    valid_loss_history.append(valid_loss)
    valid_acc_history.append(valid_acc)

print('Finished Training', file=f)
print('Finished Training')

print('train_loss_history, train_acc_history, valid_loss_history, valid_acc_history', file=f)
print([train_loss_history, train_acc_history, valid_loss_history, valid_acc_history], file=f)

print(f'Time: {time.time() - start}', file=f)
print(f'Time: {time.time() - start}')

f.close()