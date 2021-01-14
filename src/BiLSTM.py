import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

torch.manual_seed(0)
np.random.seed(0)

bidirectional = True
n_lstm_layers = 1
n_hidden_dimensions = 128
dropout = 0
vocab_len = 26


class BiLSTM(nn.Module):
    """docstring for BiLSTM"""

    def __init__(self, batch_size=1):
        super(BiLSTM, self).__init__()
        self.h0 = torch.randn(n_lstm_layers * 2, batch_size, n_hidden_dimensions)
        self.c0 = torch.randn(n_lstm_layers * 2, batch_size, n_hidden_dimensions)
        self.batch_size = batch_size

        self.lstm = nn.LSTM(input_size=vocab_len, hidden_size=n_hidden_dimensions, bidirectional=bidirectional,
                            dropout=dropout, batch_first=True, num_layers=n_lstm_layers)

        num_directions = 2 if bidirectional else 1
        self.fc = nn.Linear(n_hidden_dimensions * num_directions, 5)

    def init_hidden(self):
        self.h0 = torch.randn(n_lstm_layers * 2, self.batch_size, n_hidden_dimensions)
        self.c0 = torch.randn(n_lstm_layers * 2, self.batch_size, n_hidden_dimensions)

    def forward(self, seq_batch, seq_lens):
        self.init_hidden()
        seq_batch = torch.nn.utils.rnn.pack_padded_sequence(seq_batch, seq_lens, batch_first=True, enforce_sorted=False)

        _: object
        packed_output, _ = self.lstm(seq_batch, (self.h0, self.c0))
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        # return output
        output = output[:, -1, :]
        output = self.fc(output)

        softmax_output = F.softmax(output, dim=1)

        return softmax_output
