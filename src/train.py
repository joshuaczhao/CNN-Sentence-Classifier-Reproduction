import torch
import torch.nn as nn
import numpy as np
import time
import datetime
from sklearn.model_selection import train_test_split
import argparse
import os

import load_data
import model

def train_model(args):

    start = time.time()

    DATASET = args.dataset.upper()
    BATCH = args.batchsize
    DROPOUT_RATE = args.dropout
    LR = args.lr
    N_EPOCHS = args.epochs
    OPTIMIZER = args.optimizer.upper()
    MODEL_TYPE = args.model.upper()
    PRINT_FREQ = args.print_freq

    if MODEL_TYPE not in ['RANDOM', 'STATIC', 'NOT_STATIC', 'MULTI']:
        print('Invalid model type')
        return

    if OPTIMIZER not in ['SGD', 'ADADELTA', 'ADAM']:
        print('Invalid optimizer input')
        return

    output_dir = f'outputs/{DATASET}/{MODEL_TYPE}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    name = f'outputs/{DATASET}/{MODEL_TYPE}/{datetime.datetime.now().strftime("%m-%d-%Y_%H%M")}_{OPTIMIZER}_{LR}.txt'
    print(f'Saving Outputs to: {name}\n')
    f = open(name, "a")

    print(f'DATASET={DATASET}; BATCH={BATCH}; LR={LR}; DROPOUT={DROPOUT_RATE}; N_EPOCHS={N_EPOCHS}; OPT={OPTIMIZER}; MODEL={MODEL_TYPE}; \n')
    print(f'DATASET={DATASET}; BATCH={BATCH}; LR={LR}; DROPOUT={DROPOUT_RATE}; N_EPOCHS={N_EPOCHS}; OPT={OPTIMIZER}; MODEL={MODEL_TYPE}; \n', file=f)

    if DATASET == 'MR':
        data, labels, max_sen_len, n_classes, weights = load_data.load_MR_data()
    elif DATASET == 'SUBJ':
        data, labels, max_sen_len, n_classes, weights = load_data.load_subj_data(max_length=40)
    elif DATASET == 'TREC':
        x_train, y_train, x_test, y_test, max_sen_len, n_classes = load_data.load_TREC_data()
    else:
        print('Invalid DATASET input')
        return

    if DATASET == 'MR' or DATASET == 'SUBJ':
        x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, shuffle=True, stratify=labels)

    print('data loaded')

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")

    x_train_tensor = torch.LongTensor(x_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    x_test_tensor = torch.LongTensor(x_test).to(device)
    y_test_tensor = torch.LongTensor(y_test).to(device)

    print('data sent to device')

    train_dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
    # trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH, shuffle=True)

    cnn = model.Net(EMBEDDINGS=weights, BATCH_SIZE=BATCH, M_TYPE=MODEL_TYPE, E_DIMS=300, M_SENT_LEN=max_sen_len, DROP=DROPOUT_RATE, C_SIZE=n_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    if OPTIMIZER == 'SGD':
        optimizer = torch.optim.SGD(cnn.parameters(), lr=LR, momentum=0.9)
    elif OPTIMIZER == 'ADADELTA':
        optimizer = torch.optim.Adadelta(cnn.parameters(), lr=LR, weight_decay=0.01)
    elif OPTIMIZER == 'ADAM':
        optimizer = torch.optim.Adam(cnn.parameters(), lr=LR, weight_decay=0.01)
    else:
        print('No optimizer specified')
        return

    train_loss_history = []
    train_acc_history = []
    valid_loss_history = []
    valid_acc_history = []
    for epoch in range(N_EPOCHS):  # loop over the dataset multiple times
        epoch_start = time.time()
        print(f'===== EPOCH {epoch + 1}/{N_EPOCHS} =====', file=f)
        print(f'===== EPOCH {epoch + 1}/{N_EPOCHS} =====')
        one_epoch_acc_history = []
        one_epoch_loss_history = []

        cnn.train()
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH, shuffle=True)

        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # inputs, labels = inputs.to(device), labels.to(device)

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
            one_epoch_loss_history.append(minibatch_loss)
            if i % PRINT_FREQ == PRINT_FREQ - 1:    # print every PRINT_FREQ mini-batches
                print('[%d, %3d] train loss: %.4f train acc: %.4f' % (epoch + 1, i + 1, minibatch_loss, acc))

        train_loss = np.mean(one_epoch_loss_history)
        train_acc = np.mean(one_epoch_acc_history)
        print(f'TRAIN LOSS: {train_loss:.4f} TRAIN ACC: {train_acc:.4f}', file=f)
        print(f'TRAIN LOSS: {train_loss:.4f} TRAIN ACC: {train_acc:.4f}')

        # run validation data
        with torch.no_grad():
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
        print(f'EPOCH TIME: {(time.time() - epoch_start):.2f}s')

    print('Finished Training', file=f)
    print('Finished Training')

    print('train_loss_history, train_acc_history, valid_loss_history, valid_acc_history', file=f)
    print(f'{train_loss_history}\n {train_acc_history}\n {valid_loss_history}\n {valid_acc_history}\n', file=f)

    total_time = time.time() - start
    mins = int(total_time / 60)
    secs = int(total_time % 60)
    print(f'Total Run Time: {mins} min {secs} seconds', file=f)
    print(f'Total Run Time: {mins} min {secs} s')

    f.close()

    return cnn


if __name__ == '__main__':

    print()

    parser = argparse.ArgumentParser(description='python train.py -dataset MR -lr 0.1 -epochs 50 -optimizer ADADELTA -batchsize 50 -dropout 0.5 -model NOT_STATIC -print_freq 25')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate [default: 0.1]')
    parser.add_argument('-epochs', type=int, default=50, help='number of epochs for train [default: 50]')
    parser.add_argument('-batchsize', type=int, default=50, help='batch size for training [default: 50]')
    parser.add_argument('-optimizer', type=str, default='ADADELTA', help='optimizer [default: ADADELTA]')
    parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
    parser.add_argument('-model', type=str, default='NOT_STATIC', help='model type from [RANDOM, STATIC, NOT_STATIC, MULTI]')
    parser.add_argument('-print_freq', type=int, default=25, help='number of mini-batches to print after [default: 25]')
    parser.add_argument('-dataset', type=str, default='MR', help='dataset from [MR, TREC, SUBJ]')

    args = parser.parse_args()
    cnn = train_model(args)
