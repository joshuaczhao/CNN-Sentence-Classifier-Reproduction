import load_data
import train

import numpy as np

models = ['RANDOM', 'STATIC', 'NOT_STATIC', 'MULTI']
datasets = ['MR', 'TREC', 'SUBJ']

scores = np.zeros(shape=(len(datasets), len(models)))

for x, model_type in enumerate(models):
    for y, dataset in enumerate(datasets):
        # score = train.train_model(args=4)
        score = 5
        scores[y, x] = score

print(scores)