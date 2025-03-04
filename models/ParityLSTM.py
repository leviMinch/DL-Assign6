from torch import nn
import torch

class ParityLSTM(nn.Module) :

    def __init__(self, hidden_dim=16):
        super().__init__()
        ###########################################
        #
        # Q2 TODO
        #
        ###########################################
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_dim, batch_first=True)
        self.linear = nn.Linear(in_features=hidden_dim, out_features=2)

    def forward(self, x, x_lens):
        lstm_out, (h_n, c_n) = self.lstm(x)

        batch_size = x.size(0)

        hidden = lstm_out[torch.arange(batch_size), x_lens - 1]

        out = self.linear(hidden)

        return out

