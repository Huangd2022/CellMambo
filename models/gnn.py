import torch.nn as nn
from torch_geometric.nn import GCNConv

class GNN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = GCNConv(in_dim, out_dim)

    def forward(self, x, edge_index, edge_weight=None):
        return self.conv(x, edge_index, edge_weight)

