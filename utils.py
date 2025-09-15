import torch.nn.functional as F

def total_loss(x_true, x_recon, adj_true, adj_recon, alpha=0.5):
    loss_feat = F.mse_loss(x_recon, x_true)
    loss_struct = F.binary_cross_entropy_with_logits(adj_recon, adj_true)
    loss = alpha * loss_feat + (1 - alpha) * loss_struct
    return loss, {
        "loss": loss.item(),
        "feat_loss": loss_feat.item(),
        "struct_loss": loss_struct.item()
    }

