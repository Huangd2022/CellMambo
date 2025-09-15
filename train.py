import torch
from tqdm import tqdm
from utils import total_loss

def train_one_epoch(model, dataloader, optimizer, device, epoch, alpha=0.5):
    model.train()
    logs = {"loss": 0, "feat_loss": 0, "struct_loss": 0}
    progress = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)

    for batch in progress:
        feats, adj = batch['feats'].float().to(device), batch['adj'].float().to(device)

        optimizer.zero_grad()
        x_recon, adj_recon = model(feats, adj)
        x_true = feats.squeeze(-1)

        loss, loss_dict = total_loss(x_true, x_recon, adj, adj_recon, alpha)
        loss.backward()
        optimizer.step()

        for k in logs: logs[k] += loss_dict[k]
        progress.set_postfix(loss_dict)

    n_batches = len(dataloader)
    return {k: v / n_batches for k, v in logs.items()}

