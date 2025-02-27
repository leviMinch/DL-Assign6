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
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_dim)
        self.linear = nn.Linear(in_features=16, out_features=2)

    def forward(self, x, x_lens):
        ###########################################
        #
        # Q2 TODO
        #
        ###########################################
        x = self.lstm(x)

        vals = x[torch.arange(x.shape[0]), x_lens - 1, :]

        out = self.linear(vals)

        return out

