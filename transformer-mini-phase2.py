from torch import nn, matmul
import torch.nn.functional as F
import torch
import pandas as pd
import numpy as np
from collections import OrderedDict

class AttentionLayer(nn.Module):
    
    def __init__(self, input_dim,
                 query_repr_size=100, value_repr_size=100):
        super().__init__()
        self.W_q = nn.Parameter(torch.randn(input_dim, query_repr_size))
        self.W_k = nn.Parameter(torch.randn(input_dim, query_repr_size))
        self.W_v = nn.Parameter(torch.randn(input_dim, value_repr_size))


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
    A simple embedding layer that looks at some data and counts the unique words.
    Then, a one-hot-encoding of them is produced. 
    """

    def __init__(self, data):
        super().__init__()
        self.onehot_dict = self.word_counter(data)
        self.output_size = len(self.onehot_dict.keys())

    def word_counter(self, data):
        d = {}
        # add start and end of sentence tokens
        d['EOS'] = 1
        d['SOS'] = 1
        for sentence in data:
            for word in sentence.split(' '):
                d[word] = 1
        words = d.keys()
        onehot_size = len(words) + 2 # for start and end tokens
        for i, word in enumerate(words):
            value = ['0'] * onehot_size
            value[i] = '1'
            value = ''.join(value)
            d[word] = value
        return d        

    def forward(self, x):
        tokens = ['SOS'] + x.split(' ') + ['EOS']
        return torch.Tensor([list(map(int, self.onehot_dict[xi])) for xi in x.split(' ')])


def test_embedding_layer(sentence_list):
    layer = EmbeddingLayer(sentence_list)
    embedding = layer(sentence_list)
    print(embedding)

def get_vocab_dict(text_data):
    """
    Note that text_data is assumed to be (input, output) pairs of sentences
    """
    # create a "big sentence" out of all the (input, output) pairs
    sentences = text_data.flatten()
    raw_text = ' '.join(sentences)
    unique_words = set(raw_text.split(' ') + ['SOS'] + ['EOS'] + ['UNK'])
    vocab_dict = {w: i for i, w in enumerate(unique_words)}
    return vocab_dict
    
def preprocess_data(data):
    """
    Note that text_data is assumed to be (input, output) pairs of sentences
    """
    vocab_dict = get_vocab_dict(data)
    input_return = []
    output_return = []
    for example in data:
        input = example[0]
        output = example[1]
        input_idx = np.array([vocab_dict[word] for word in input.split(' ')])
        output_idx = np.array([vocab_dict[word] for word in output.split(' ')])
        input_return.append(input_idx)
        output_return.append(output_idx)
    return input_return, output_return


data = pd.read_csv('input.csv').to_numpy()
X, y = preprocess_data(data)
print(X)

#test_embedding_layer(['I wonder what happens when I do this'])

data_preprocess = data.flatten()
onehot_length = EmbeddingLayer(data_preprocess).output_size
network = nn.Sequential(OrderedDict([
        ('embedding', EmbeddingLayer(data_preprocess)),
        ('embedding_repr', nn.Linear(onehot_length, 100)),
        ('attention1', AttentionLayer(100))
    ]
    ))
