from torch import nn, matmul
import torch.nn.functional as F
import torch
import pandas as pd
import numpy as np
from collections import OrderedDict

class AttentionLayer(nn.Module):
    
    def __init__(self, input_dim, output_dim,
                 query_repr_size=100, value_repr_size=100):
        self.sequence_length = sequence_length
        self.output_dim = output_dim
        self.query_repr_size = query_repr_size
        self.W_q = nn.Parameter(torch.randn(sequence_length, query_repr_size))
        self.W_k = nn.Parameter(torch.randn(sequence_length, query_repr_size))
        self.W_v = nn.Parameter(torch.randn(sequence_length, value_repr_size))


    def forward(self, x):
        # assumption: x has dimensionality (n, d) where n is the number of
        # words/tokens and d is the embedding dimension
        Q = matmul(x, self.W_q)
        K = matmul(x, self.W_k)
        V = matmul(x, self.W_v)
        S = torch.matmul(Q, K.T) / torch.sqrt(torch.tensor(self.W_q.shape[1]).float())
        A = torch.softmax(S, dim=1)
        return matmul(A, V)
    
class EmbeddingLayer(nn.Module):
    """
    A simple embedding layer that looks at some data and counts the words.
    Then, a one-hot-encoding of them is produced. 
    """

    def __init__(self, data):
        self.onehot_dict = self.word_counter(data)
        self.output_size = len(self.onehot_dict.keys())

    def word_counter(self, data):
        d = {}
        for sentence in data:
            for word in sentence.split(' '):
                d[word] = 1
        words = d.keys()
        onehot_size = len(words)
        for i, word in enumerate(words):
            value = ['0'] * onehot_size
            value[i] = '1'
            value = ''.join(value)
            d[word] = value
        return d        

    def forward(self, x):
        return torch.Tensor([self.onehot_dict[xi] for xi in x.split(' ')])


data = pd.read_csv('input.csv').to_numpy()
data_preprocess = data.flatten()
onehot_length = EmbeddingLayer(data_preprocess).output_size
network = nn.Sequential(OrderedDict([
        ('embedding', EmbeddingLayer(data_preprocess)),
        ('embedding_repr', nn.Linear(onehot_length, 100)),
        ('attention1', AttentionLayer(10, 10))
    ]
    ))
