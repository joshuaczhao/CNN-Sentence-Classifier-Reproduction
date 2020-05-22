import data_helpers
import numpy as np
from torchtext.data.utils import get_tokenizer
from sklearn.model_selection import cross_val_score

positive_data_file = "data/rt-polaritydata/rt-polarity.pos"
negative_data_file = "data/rt-polaritydata/rt-polarity.neg"


def Preprocess(positive_data_file, negative_data_file):
    dev_sample_percentage = .1

    print("Loading data...")
    x_text, y = data_helpers.load_data_and_labels(positive_data_file, negative_data_file)

    breakpoint()
    # Build vocabulary
    tokenizer = get_tokenizer(None)
    x = tokenizer(x_text)
   # breakpoint()

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # TODO: Replace with Cross-Validation
    dev_sample_index = -1 * int(dev_sample_percentage * float(len(y)))
    x_train, x_val = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_val = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

    del x, y, x_shuffled, y_shuffled

  #  print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_val)))
#    return x_train, y_train, vocab_processor, x_val, y_val


Preprocess(positive_data_file, negative_data_file)

