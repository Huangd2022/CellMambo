import torch
import torch.nn as nn
from torch_geometric.nn import dense_diff_pool
from torch_geometric.utils import dense_to_sparse

from models.gnn import GNN
from models.decoder import Decoder

class DiffPoolAutoencoder(nn.Module):
    def __init__(self, in_dim=1, embed_dim=16, num_nodes=920, pool_size=[20]):
        super().__init__()
        self.num_nodes = num_nodes
        self.embed_dim = embed_dim
        self.pool_size = pool_size

        self.gnn_embed = GNN(in_dim, embed_dim)
        self.s1 = nn.Parameter(torch.randn(num_nodes, pool_size[0]))
        self.upsampler = nn.Sequential(
            nn.Linear(pool_size[-1] * embed_dim, num_nodes * embed_dim),
            nn.ReLU()
        )
        self.decoder = Decoder(embed_dim)

    def forward(self, x, adj, return_pooled=False):
        B, N, _ = x.shape
        S1 = torch.softmax(self.s1, dim=-1)

        batch_embeddings, pooled_adjs, pooled_feats = [], [], []

        for b in range(B):
            x_b, adj_b = x[b], adj[b]
            edge_index, edge_weight = dense_to_sparse(adj_b)

            z = self.gnn_embed(x_b, edge_index, edge_weight)
            z_pool, adj_pool, _, _ = dense_diff_pool(z, adj_b, S1)
            adj_pool = (adj_pool > 1e-3).float()

            pooled_adjs.append(adj_pool)
            pooled_feats.append(z_pool)

            upsampled = self.upsampler(z_pool.view(-1))
            batch_embeddings.append(
                upsampled.view(self.num_nodes, self.embed_dim)
            )

        batch_embeddings = torch.stack(batch_embeddings)
        x_recon, adj_recon = self.decoder(batch_embeddings)

        if return_pooled:
            return x_recon, adj_recon, torch.stack(pooled_adjs), torch.stack(pooled_feats)
        return x_recon, adj_recon

