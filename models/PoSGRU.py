import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class PoSGRU(nn.Module) :

    def __init__(self, vocab_size=1000, embed_dim=16, hidden_dim=16, num_layers=2, output_dim=10, residual=True) :
      super().__init__()
      ###########################################
      #
      # Q4 TODO
      #
      ###########################################

      #####################
      # Add padding layer #
      #####################
      self.embed = nn.Embedding(num_embeddings=embed_dim, embedding_dim=vocab_size)
      self.embedToGru = nn.Linear(in_features=embed_dim, out_features=hidden_dim)

      #appending all the GRUS
      self.grus = nn.ModuleList()
      for _ in range(num_layers):
        self.grus.append(nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim//2, bidirectional=True))

      self.gruToGelu = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
      self.gelu = nn.Gelu()
      self.output = nn.Linear(in_features=hidden_dim, out_features=output_dim)



    def forward(self, x):
      ###########################################
      #
      # Q4 TODO
      #
      ###########################################
      x = self.embed(x)
      x = self.embedToGru(x)
      for layer in self.grus:
        x = layer(x)

      x = self.gruToGelu(x)
      x = self.gelu(x)
      x = self.output(x)

      return x