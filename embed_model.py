"""Resolve a referent

Usage:
  embed_model.py
  embed_model.py (-h | --help)
  embed_model.py --version

Options:
  -h --help     Show this screen.
  --version     Show version.

"""

import re
import sys
sys.path.append('./utils/')
import random
import numpy as np
from docopt import docopt
from ext2vec import ext2vec
from utils import read_external_vectors


if __name__ == '__main__':
    args = docopt(__doc__, version='Interpretation 0.1')

img_ids, contexts = read_external_vectors("./data/55_vision_animals.txt")
vocab, interpretations = read_external_vectors("./data/55_vision_animals_mapping.bin")
print("Length of vocab:",len(vocab))
corpus = []


print("Running ext2vec...")
settings = {}
settings['n'] = len(contexts[0])                   # dimension of word embeddings
settings['window_size'] = 1         # context window +/- center word
settings['min_count'] = 1           # minimum word count
settings['epochs'] = 10           # number of training epochs
settings['neg_samp'] = 5           # number of negative words to use during training
settings['learning_rate'] = 0.001    # learning rate
#np.random.seed(0)                   # set the seed for reproducibility


# INITIALIZE E2V MODEL
e2v = ext2vec(vocab, img_ids, contexts, settings)
e2v.train(interpretations,vocab, img_ids)
e2v.pretty_print(vocab)
