import torch

from sklearn.base import BaseEstimator, TransformerMixin


def initialize_infersent(sentences):
    infersent = torch.load('infersent/infersent.allnli.pickle', map_location=lambda storage, loc: storage)
    print("Loaded")
    infersent.set_glove_path('infersent/glove.840B.300d.txt')
    infersent.build_vocab(sentences, tokenize=True)
    return infersent


class InfersentFeature(BaseEstimator, TransformerMixin):

    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        return self

    def transform(self, sentences):
        encode = self.model.encode(sentences, tokenize=True)
        return encode

    def get_feature_names(self):
        return ['infersent_{}'.format(w) for w in range(0,4096)]

