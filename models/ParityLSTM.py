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
        self.lstm = nn.LSTM(input_size=9, hidden_size=hidden_dim, batch_first=True)
        self.linear = nn.Linear(in_features=hidden_dim, out_features=2)

    def forward(self, x, x_lens):
        print("shape of x", x.shape)
        print("shape of x_lens", x_lens.shape)
        lstm_out, (h_n, c_n) = self.lstm(x)
        print("shape of output:", lstm_out.shape)

        batch_size = x.size(0)

        hidden = lstm_out[torch.arange(batch_size), x_lens - 1]

        print("shape of hidden:", hidden.shape)
        out = self.linear(hidden)

        return out

