import torch.nn as nn
import torch

class LSTMClassifier(nn.Module):
    """
    This is the simple RNN model we will be using to perform Sentiment Analysis.
    """

    def __init__(self, embedding_dim, hidden_dim, vocab_size, n_layers=2, in_dropout=0.5, lstm_dropout=0.8):
        """
        Initialize the model by settingg up the various layers.
        """
        super(LSTMClassifier, self).__init__()

        self.n_layers=n_layers
        self.hidden_dim=hidden_dim
        self.h0 = torch.nn.Parameter(torch.randn((2*n_layers, 1, hidden_dim),requires_grad=True))
        self.c0 = torch.nn.Parameter(torch.randn((2*n_layers, 1, hidden_dim),requires_grad=True))
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(p=in_dropout)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, dropout=lstm_dropout, bidirectional=True)
        self.norm1 = nn.BatchNorm1d(2*hidden_dim)
        self.dense = nn.Linear(2*hidden_dim, hidden_dim)
        self.norm2 = nn.BatchNorm1d(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)
        self.sig = nn.Sigmoid()
        
        self.word_dict = None

    def forward(self, x):
        """
        Perform a forward pass of our model on some input.
        """
        #print('input: {}'.format(x.shape))
        x = x.t() #TF: because batch_first=False
        #print('transposed: {}'.format(x.shape))
        lengths = x[0,:]
        reviews = x[1:,:]
        #print('reviews: {}'.format(reviews.shape))
        fixed_length = reviews.shape[0]
        batch_size=reviews.shape[1]
        embeds = self.embedding(reviews)
        embeds = self.dropout(embeds)
        #print('embedded: {}'.format(embeds.shape))
        #TF: use learnable hidden state. i don't need the updated hidden because every review has his own history
        lstm_out, _ = self.lstm(embeds, (self.h0.repeat(1, batch_size, 1), 
                                         self.c0.repeat(1, batch_size, 1)))        
        #print('lstm out: {}'.format(lstm_out.shape))
        temp_out = lstm_out.view(fixed_length, batch_size, 2, self.hidden_dim) #reshape to separate the two directions
        #print('lstm out after separation: {}'.format(temp_out.shape))
        #TF: concatenate the last significative output of the right lstm (lenghts-1) with the last output of the left lstm (0)
        out = torch.cat((temp_out[-1, :, 0, :], temp_out[0, :, 1, :]), 1).squeeze()        
        #print('resulting output after lstm: {}'.format(out.shape))
        out = self.norm1(out)
        #print('after batchnorm1: {}'.format(out.shape))
        out = self.dense(out)
        #print('after fc1: {}'.format(out.shape))
        out = self.norm2(out)
        #print('after batchnorm2: {}'.format(out.shape))
        out = self.classifier(out)
        #print('after classifier: {}'.format(out.shape))
        out = out.view(batch_size, -1, 1)
        #print('predictions reshaped: {}'.format(out.shape))
        #out = out[lengths - 1, range(len(lengths))].squeeze()
        #print('prediction filtered: {}'.format(out.shape))
        out = self.sig(out).squeeze()
        #print('after sigmoid: {}'.format(out.shape))
        return out