import torch
import torch.nn as nn
import pickle

torch.set_num_threads(4)
torch.set_num_interop_threads(4)
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
        packedOutput, hidden = self.rnn(packedEmbedded)

        output, outputLengths = nn.utils.rnn.pad_packed_sequence(packedOutput)
        if self.bidir:
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)

        return self.dense(self.dropout(hidden))

basicRNN = MyRNN(model='RNN')
biLSTM = MyRNN(model = 'BiLSTM') # Construct a BiLSTM model, as above

biLSTM.embed.weight.data.copy_(reviewVocabVectors)
biLSTM.embed.weight.data[unk] = torch.zeros(embeddingSize)
biLSTM.embed.weight.data[pad] = torch.zeros(embeddingSize)

criterion = nn.BCEWithLogitsLoss()

def batchAccuracy(preds, targets):
    roundedPreds = (preds >= 0)
    return (roundedPreds == targets).sum().item() / len(preds)

biLSTM.train()

torch.manual_seed(0)
optimizer = torch.optim.Adam(biLSTM.parameters())
for epoch in range(numEpochs):
    epochLoss = 0
    for batch in trainIterator:
        optimizer.zero_grad()
        text, textLen = batch[0]
        predictions = model(text, textLen).squeeze(1)
        loss = criterion(predictions, batch[1])
        loss.backward()
        optimizer.step()
        epochLoss += loss.item()
    print(f'Model: {biLSTM.name}, Epoch: {epoch + 1}, Train Loss: {epochLoss / len(trainIterator)}')
print()

biLSTM.eval()

with torch.no_grad():
    accuracy = 0.0
    for batch in testIterator:
        text, textLen = batch[0]
        predictions = model(text, textLen).squeeze(1)
        loss = criterion(predictions, batch[1])
        acc = batchAccuracy(predictions, batch[1])
        accuracy += acc
    print('Model: {}, Validation Accuracy: {}%'.format(biLSTM.name, accuracy / len(testIterator) * 100))
