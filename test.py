import torch
from utils import total_loss




def evaluate_one_epoch(model, dataloader, device, alpha=0.5):
model.eval()
logs = {"loss": 0, "feat_loss": 0, "struct_loss": 0}


with torch.no_grad():
for batch in dataloader:
feats, adj = batch['feats'].float().to(device), batch['adj'].float().to(device)
outputs = model(feats, adj)
x_recon, adj_recon = outputs[0], outputs[1]
x_true = feats.squeeze(-1)


loss, loss_dict = total_loss(x_true, x_recon, adj, adj_recon, alpha)
for k in logs: logs[k] += loss_dict[k]


n_batches = len(dataloader)
return {k: v / n_batches for k, v in logs.items()}
