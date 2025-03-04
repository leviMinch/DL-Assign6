import torch
from torch import nn


class PoSGRU(nn.Module):
    def __init__(self, vocab_size=1000, embed_dim=16, hidden_dim=16, num_layers=2, output_dim=10, residual=True):
        super().__init__()

        self.residual = residual

        # Embedding layer with padding index
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=1)
        self.embedToGru = nn.Linear(in_features=embed_dim, out_features=hidden_dim)

        # GRU layers (bidirectional)
        self.grus = nn.ModuleList([
            nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim // 2, bidirectional=True, batch_first=True)
            for _ in range(num_layers)
        ])

        # Classification layers
        self.gruToGelu = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.gelu = nn.GELU()
        self.output = nn.Linear(in_features=hidden_dim, out_features=output_dim)

    def forward(self, x):
        print("Shape of x:", x.shape)
        x = self.embed(x)
        print("shape after embedding:", x.shape)
        x = self.embedToGru(x)

        for layer in self.grus:
            residual = x.clone()  # Prevent in-place modification issue
            x, _ = layer(x)
            if self.residual:
                x = x + residual  # Add residual connection

        x = self.gruToGelu(x)
        x = self.gelu(x)
        x = self.output(x)

        return x