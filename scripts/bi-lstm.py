import torch
import torch.nn as nn
import pickle

torch.set_num_threads(8)
print()

# temp params, will tweak later
embeddingsize = 100 
hiddensize = 10
dropoutrate = 0.2
numepochs = 10
vocabsize = 20000
pad = 1
unk = 0

class MyRNN(nn.Module):
    '''Finish building this class'''
    def __init__(self,model):
        super().__init__()
        self.name = model
        self.bidir = (model == 'BiLSTM')

        self.embed = nn.Embedding(vocabsize,embeddingsize,padding_idx=pad)
        self.rnn = nn.LSTM(embeddingsize,hiddensize,bidirectional=self.bidir)

        self.dense = nn.Linear(hiddensize * (2 if self.bidir else 1),1)
        self.dropout = nn.Dropout(dropoutrate)

    def forward(self,text,textlengths):
        embedded = self.dropout(self.embed(text))
        packedembedded = nn.utils.rnn.pack_padded_sequence(embedded,textlengths)

        hidden = torch.cat(())
        return self.dense(self.dropout(hidden))







