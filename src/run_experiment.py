import train
from argparse import Namespace
import numpy as np
import datetime
import time

start = time.time()

name = f'outputs/experiments/{datetime.datetime.now().strftime("%m-%d-%Y_%H%M")}.txt'
f = open(name, 'w')

models = ['RANDOM', 'STATIC', 'NOT_STATIC', 'MULTI']
datasets = ['MR', 'SST-1', 'SST-2', 'SUBJ', 'TREC', 'CR', 'MPQA']

scores = np.zeros(shape=(len(models), len(datasets)))

for x, model_type in enumerate(models):
    for y, dataset in enumerate(datasets):

        # if dataset in ['SST-1', 'SST-2']:
        #     continue

        args = Namespace(
            lr=0.01,
            epochs=200,
            batchsize=50,
            optimizer='ADADELTA',
            dropout=0.1,
            model=model_type,
            print_freq=500,
            dataset=dataset
        )
        model, score = train.train_model(args=args)
        scores[x, y] = score

        print(scores)
        print(scores, file=f)

total_time = time.time() - start
mins = int(total_time / 60)
secs = int(total_time % 60)
print(f'Total Run Time: {mins} min {secs} seconds', file=f)
print(f'Total Run Time: {mins} min {secs} s')