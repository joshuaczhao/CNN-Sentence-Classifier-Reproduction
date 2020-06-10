import train
from argparse import Namespace
import numpy as np

models = ['RANDOM', 'STATIC', 'NOT_STATIC', 'MULTI']
datasets = ['MR', 'SST-1', 'SST-2', 'SUBJ', 'TREC', 'CR', 'MPQA']

scores = np.zeros(shape=(len(datasets), len(models)))

for x, model_type in enumerate(models):
    for y, dataset in enumerate(datasets):
        args = Namespace(
            lr=0.01,
            epochs=100,
            batchsize=50,
            optimizer='ADADELTA',
            dropout=0.1,
            model=model_type,
            print_freq=200,
            dataset=dataset
        )
        model, score = train.train_model(args=args)
        scores[y, x] = score

print(scores)