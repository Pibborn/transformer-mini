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

    def __init__(self, vocab, embedding_size=100):
        super().__init__()
        self.vocab = vocab
        self.embedding_matrix = nn.Parameter(torch.randn(len(vocab), embedding_size))

    def forward(self, x):
        # x is indexes now
        x = self.embedding_matrix[x] 
        return x


def get_vocab_dict(text_data):
    # create a "big sentence" out of all the (input, output) pairs
    sentences = text_data.flatten()
    raw_text = ' '.join(sentences).lower()  # convert to lowercase
    unique_words = set(raw_text.split(' ') + ['SOS', 'EOS', 'PAD', 'UNK'])
    vocab_dict = {w: i for i, w in enumerate(unique_words)}
    return vocab_dict

def preprocess_data(data):
    vocab_dict = get_vocab_dict(data)
    input_return = []
    output_return = []
    for example in data:
        input = example[0]
        output = example[1]
        # Add 'SOS' at the beginning and 'EOS' at the end of the sentence
        input_idx = np.array([vocab_dict['SOS']] + [vocab_dict.get(word, vocab_dict['UNK']) for word in input.lower().split(' ')] + [vocab_dict['EOS']])
        output_idx = np.array([vocab_dict['SOS']] + [vocab_dict.get(word, vocab_dict['UNK']) for word in output.lower().split(' ')] + [vocab_dict['EOS']])
        input_return.append(input_idx)
        output_return.append(output_idx)
    # Padding can be done here, but it also can be done during the batch formation stage in PyTorch
    return input_return, output_return, vocab_dict



data = pd.read_csv('input.csv').to_numpy()
X, y, vocab = preprocess_data(data)

layer = EmbeddingLayer(vocab)
input_string = 'what is the capital'
input_indices = torch.Tensor([vocab[word] for word in input_string.lower().split(' ')]).long()
emb = layer(input_indices)
print(emb.shape)
exit(1)


#test_embedding_layer(['I wonder what happens when I do this'])

data_preprocess = data.flatten()
onehot_length = EmbeddingLayer(data_preprocess).output_size
network = nn.Sequential(OrderedDict([
        ('embedding', EmbeddingLayer(data_preprocess)),
        ('embedding_repr', nn.Linear(onehot_length, 100)),
        ('attention1', AttentionLayer(100))
    ]
    ))
