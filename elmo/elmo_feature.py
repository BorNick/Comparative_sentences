import torch
import numpy as np
from string import punctuation
from allennlp.commands.elmo import ElmoEmbedder

from sklearn.base import BaseEstimator, TransformerMixin



def initialize_elmo():
 #   options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
 #   weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
    weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5"
    options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json"
    elmo = ElmoEmbedder(options_file, weight_file)
    print("Loaded")
    return elmo


class ElmoFeature(BaseEstimator, TransformerMixin):

    def __init__(self, model, n_batches):
        self.model = model
        self.n_batches = n_batches

    def fit(self, X, y):
        return self

    def transform(self, sentences):
        tokens = self.tokenize(sentences)
        embeddings = np.zeros((len(sentences), 1024))
        full_embs = embs = self.model.embed_sentences(tokens, batch_size=self.n_batches)
        for i, e in enumerate(full_embs):
            embeddings[i, :] = e[2,:,:].mean(axis=0)
        return embeddings

    def tokenize(self, sentences): 
        tokens = [] 
        for sent in sentences:
            sent = self.add_spaces_to_punct(sent.lower())
            tokens.append(sent.split())
        return tokens

    def add_spaces_to_punct(self, text):
        processed_text = []
        for letter in text:
            if letter in punctuation:
                processed_text.append(' ')
                processed_text.append(letter)
                processed_text.append(' ')
            else:
                processed_text.append(letter)
        return ("".join(processed_text))
	

    def get_feature_names(self):
        return ['Elmo_{}'.format(w) for w in range(0,1024)]

