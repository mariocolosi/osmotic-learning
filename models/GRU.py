# v0.1.0

import torch.nn as nn


class GRUModel(nn.Module):
    """
    GRU-based model for generating embeddings.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GRUModel, self).__init__()
        self.input_size = input_dim
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        x: [batch_size, sequence_length, input_dim]
        """
        _, h = self.gru(x)
        return self.fc(h[-1])