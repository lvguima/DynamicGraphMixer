import torch.nn as nn

from layers.DynamicGraph import LinearGraphMixing


class GraphMixer(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.mixer = LinearGraphMixing(dropout=dropout)

    def forward(self, adj, x):
        return self.mixer(adj, x)
