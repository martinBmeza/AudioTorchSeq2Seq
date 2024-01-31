import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, dropout):
        """
        input_dim = n_channels (cantidad de features)
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.rnn = nn.LSTM(input_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        inputs, lens = x
        packed = pack_padded_sequence(inputs, lens, enforce_sorted=False, batch_first=True)
        output, (hidden, cell) = self.rnn(packed)
        #output, _ = pad_packed_sequence(output, batch_first=True, padding_value=1.)
        #print(f'Encoder output [Batch, Seq_len, Dim*Hidden]: {output.shape}')
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self,input_dim, output_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.rnn = nn.LSTM(input_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        output, (hidden, cell) = self.rnn(input, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder_params, decoder_params, device):
        super().__init__()
        self.encoder = Encoder(**encoder_params)
        self.decoder = Decoder(**decoder_params)
        self.device = device
        assert(self.encoder.hidden_dim == self.decoder.hidden_dim),\
                "Hidden dim of encoder and decoder must be equal"
        assert(self.encoder.n_layers == self.decoder.n_layers),\
                "Encoder and decoder must have equal number of layers"
    
    def forward(self, x, trg, seq_len, teacher):
        #  x  --> [Batch, Seq_len, Channel]
        # trg --> [Batch, Seq_len, Channel]
        # seq_len --> list of len = num_batches
        (batch_size, trg_length, channels) = trg.shape
        #tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_length, channels).to(self.device)
        hidden, cell = self.encoder((x, seq_len))
        # first input es <sos> (?
        input = trg[:, :1, :] # first input must be the sos 
        for t in range(trg_length):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[:,t:t+1] = output
            teacher_force = random.random() < teacher
            input = trg[:, t:t+1] if teacher_force else output
        return outputs
