import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, in_dim, out_dim=1):
        super().__init__()
        self.edge_decoder = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU()
        )
        self.feat_decoder = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU()
        )

    def forward(self, z):
        x_recon = self.feat_decoder(z).squeeze(-1)
        adj_logits = z @ z.transpose(1, 2)
        adj_recon = self.edge_decoder(adj_logits.unsqueeze(-1)).squeeze(-1)
        return x_recon, adj_recon

