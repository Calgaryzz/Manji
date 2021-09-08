# import resources
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np


# Todo: change to our data type (seq like true,false,true,true,false....)
training_data = [
    ("The cat ate the cheese".lower().split(), ["FALSE", "TRUE", "TRUE", "FALSE", "TRUE"]),
    ("She read cheese book".lower().split(), ["TRUE", "TRUE", "FALSE", "TRUE"]),
    ("The dog loves art".lower().split(), ["FALSE", "TRUE", "TRUE", "TRUE"]),
    ("The elephant answers cheese phone".lower().split(), ["FALSE", "TRUE", "TRUE", "FALSE", "TRUE"])
]

# create a dictionary that maps transitions to indices
tran2idx = {}

for sent, tags in training_data:
    for tran in sent:
        if tran not in tran2idx:
            tran2idx[tran] = len(tran2idx)

# Values
tag2idx = {"FALSE": 0, "TRUE": 1}

# Todo: Our sequential data to tensor
def prepare_sequence(seq, dictionnary):
    '''This function takes in a sequence of transition and returns a
    corresponding Tensor of numerical values (indices for each transition).'''

    idxs = [dictionnary[w] for w in seq]
    idxs = np.array(idxs)

    return torch.from_numpy(idxs)


class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, output_size):
        ''' Initialize the layers of this model.'''
        super(LSTMTagger, self).__init__()

        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, output_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        ''' At the start of training, we need to initialize a hidden state;
           there will be none because the hidden state is formed based on perviously seen data.
           So, this function defines a hidden state with all zeroes and of a specified size.'''
        # The axes dimensions are (n_layers, batch_size, hidden_dim)
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    #Todo : adapt forward pass to lattice
    def forward(self, sentence):
        # create embedded word vectors for each word in a sentence
        embeds = self.word_embeddings(sentence)

        # get the output and hidden state by passing the lstm over our word embeddings
        # the lstm takes in our embeddings and hiddent state
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden)

        # get the scores for the most likely tag for a word
        tag_outputs = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_outputs, dim=1)

        return tag_scores

# the embedding dimension defines the size of our word vectors
# for our simple vocabulary and training set, we will keep these small
EMBEDDING_DIM = 6
HIDDEN_DIM = 6

# instantiate our model
model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(tran2idx), len(tag2idx))

# define our loss and optimizer
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

test_sentence = "The cheese loves the elephant".lower().split()

# see what the scores are before training
# element [i,j] of the output is the *score* for tag j for word i.
# to check the initial accuracy of our model, we don't need to train, so we use model.eval()
inputs = prepare_sequence(test_sentence, tran2idx)
inputs = inputs
tag_scores = model(inputs)
print(tag_scores)

_, predicted_tags = torch.max(tag_scores, 1)
print('\n')
print('Predicted tags: \n',predicted_tags)

n_epochs = 300

for epoch in range(n_epochs):

    epoch_loss = 0.0

    # get all sentences and corresponding tags in the training data
    for sentence, tags in training_data:

        model.zero_grad()
        model.hidden = model.init_hidden()

        # prepare the inputs for processing by out network,
        # turn all sentences and targets into Tensors of numerical indices
        sentence_in = prepare_sequence(sentence, tran2idx)
        targets = prepare_sequence(tags, tag2idx)

        # forward pass to get tag scores
        tag_scores = model(sentence_in)

        # compute the loss, and gradients
        loss = loss_function(tag_scores, targets)
        epoch_loss += loss.item()
        loss.backward()

        # update the model parameters with optimizer.step()
        optimizer.step()

    # print out avg loss per 20 epochs
    if (epoch % 20 == 19):
        print("Epoch: %d, loss: %1.5f" % (epoch + 1, epoch_loss / len(training_data)))

test_sentence = "The cheese loves the elephant".lower().split()

# see what the scores are after training
inputs = prepare_sequence(test_sentence, tran2idx)
inputs = inputs
tag_scores = model(inputs)
print(tag_scores)

_, predicted_tags = torch.max(tag_scores, 1)
print('\n')
print('Predicted tags: \n',predicted_tags)