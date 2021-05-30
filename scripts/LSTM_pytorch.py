import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from extract_feature import preprocessing

traindf = pd.read_csv('assets/train.csv')

cv = CountVectorizer(analyzer=lambda x: x)

#df with feature representations from preprocessing function of extract_feature.py
df = preprocessing(df.dropna())
df['vec'] = df.words.apply(lambda x: " ".join(x))

arr = cv.fit_transform(df['vec']).toarray()

df['vec2'] = [x for x in arr]

from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()



X_arr = scaler.fit_transform(df[['sentence_len','freq_score','aoa_score','syllable_count','Flesch_Kincaid']])
df['l'] = [x for x in X_arr]

train_sample = df.sample(20000)[['l','vec2','label']]
split_frac = 0.8
train_x = train_sample.vec2[0:int(split_frac*len(train_sample))]
t_x = train_sample.vec2[0:int(split_frac*len(train_sample))]
train_y = train_sample.label[0:int(split_frac*len(train_sample))]

valid_x = train_sample.vec2[int(split_frac*len(train_sample)):]
valid_y = train_sample.label[int(split_frac*len(train_sample)):len(train_sample)]

print(train_x.shape)
print(train_y.shape)
print(valid_x.shape)
print(valid_y.shape)

train_data = TensorDataset(torch.from_numpy(np.stack(np.array(train_x), axis = 0)), torch.from_numpy(np.array(train_y)))
valid_data = TensorDataset(torch.from_numpy(np.array(np.stack(np.array(valid_x), axis = 0))), torch.from_numpy(np.array(valid_y)))

batch_size = 100

train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)


class SentimentLSTM(nn.Module):
    """
    The RNN model that will be used to perform Sentiment analysis.
    """

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        """
        Initialize the model by setting up the layers.
        """
        super().__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        # embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                            batch_first=True) #dropout=drop_prob

        # dropout layer
        self.dropout = nn.Dropout(0.4)

        # linear and sigmoid layers
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sig = nn.Sigmoid()

    def forward(self, x, hidden):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        batch_size = x.size(0)

        # embeddings and lstm_out
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)

        # stack up lstm outputs
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        # dropout and fully-connected layer
        out = self.dropout(lstm_out)
        out = self.fc(out)
        # sigmoid function
        sig_out = self.sig(out)

        # reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1]  # get last batch of labels

        # return last sigmoid output and hidden state
        return sig_out, hidden

    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())

        return hidden

# Instantiate the model w/ hyperparams
vocab_size = 1000 # +1 for the 0 padding
output_size = 2
embedding_dim = 400
hidden_dim = 256
n_layers = 2
net = SentimentLSTM(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
print(net)

lr = 0.01

criterion = nn.BCELoss()
optimizer = torch.optim.RMSprop(net.parameters(), lr=lr)

# training params

epochs = 5  # 3-4 is approx where I noticed the validation loss stop decreasing

counter = 0
print_every = 100
clip = 5  # gradient clipping

net.train()
# train for some number of epochs
for e in range(epochs):
    # initialize hidden state
    h = net.init_hidden(batch_size)

    # batch loop
    for inputs, labels in tqdm(train_loader):
        counter += 1

        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        h = tuple([each.data for each in h])

        # zero accumulated gradients
        net.zero_grad()

        # get the output from the model
        inputs = inputs.type(torch.LongTensor)
        output, h = net(inputs, h)

        # calculate the loss and perform backprop
        loss = criterion(output.squeeze(), labels.float())
        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        nn.utils.clip_grad_norm_(net.parameters(), clip)
        optimizer.step()

        # loss stats
        if counter % print_every == 0:
            # Get validation loss
            val_h = net.init_hidden(batch_size)
            val_losses = []
            net.eval()
            for inputs, labels in valid_loader:
                # Creating new variables for the hidden state, otherwise
                # we'd backprop through the entire training history
                val_h = tuple([each.data for each in val_h])

                inputs = inputs.type(torch.LongTensor)
                output, val_h = net(inputs, val_h)
                val_loss = criterion(output.squeeze(), labels.float())

                val_losses.append(val_loss.item())

            net.train()
            print("Epoch: {}/{}...".format(e + 1, epochs),
                  "Step: {}...".format(counter),
                  "Loss: {:.6f}...".format(loss.item()), )

valid_losses = []  # track loss
num_correct = 0

# init hidden state
h = net.init_hidden(batch_size)

net.eval()
# iterate over valid data
for inputs, labels in tqdm(valid_loader):
    # Creating new variables for the hidden state, otherwise
    # we'd backprop through the entire training history
    h = tuple([each.data for each in h])

    # get predicted outputs
    inputs = inputs.type(torch.LongTensor)
    output, h = net(inputs, h)

    # calculate loss
    valid_loss = criterion(output.squeeze(), labels.float())
    valid_losses.append(valid_loss.item())

    # convert output probabilities to predicted class (0 or 1)
    pred = torch.round(output.squeeze())  # rounds to the nearest integer
    # compare predictions to true label
    correct_tensor = pred.eq(labels.float().view_as(pred))
    correct = np.squeeze(correct_tensor.numpy())
    num_correct += np.sum(correct)

# -- stats! -- ##
# avg valid loss
print("valid loss: {:.3f}".format(np.mean(valid_losses)))

# accuracy over all valid data
valid_acc = num_correct / len(valid_loader.dataset)
print("valid accuracy: {:.3f}".format(valid_acc))