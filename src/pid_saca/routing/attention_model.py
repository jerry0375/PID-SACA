"""Attention-based routing model, supervised-data loaders, and training/evaluation helpers."""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import json
try:
    import wandb
except ImportError:
    wandb = None
import random
import pathlib
from pid_saca.paths import REPO_ROOT, ROUTING_MODEL_DIR
BASE = pathlib.Path(__file__).resolve().parent


EMBEDDING_DIM = 64
N_HEADS = 8
N_LAYERS = 3
LEARNING_RATE = 1e-4
BATCH_SIZE = 512
START = [104.06, 30.67]
NUM_EPOCHS = 5000
RESTART_EVERY = 2000

class SACA_EncoderLayer(nn.Module):


    def __init__(self, embed_dim, n_heads, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )



        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch_size, seq_len, embed_dim)

        # 1. Multi-Head Attention + Residual
        attn_out, _ = self.mha(x, x, x)
        x = x + self.dropout1(attn_out)


        # x_permuted = x.permute(0, 2, 1) # (B, C, L)
        # x_bn = self.bn1(x_permuted)
        # x = x_bn.permute(0, 2, 1) # (B, L, C)

        x = self.ln1(x)

        # 3. Feed Forward + Residual
        ff_out = self.ff(x)
        x = x + self.dropout2(ff_out)

        # x_permuted = x.permute(0, 2, 1)
        # x_bn = self.bn2(x_permuted)
        # x = x_bn.permute(0, 2, 1)

        x = self.ln2(x)

        return x

class AttentionRouteModel(nn.Module):


    def __init__(self, embed_dim, n_heads, n_layers):
        super().__init__()
        self.embed_dim = embed_dim


        self.node_embed = nn.Linear(3, embed_dim)

        # 2. Encoder
        self.encoder_layers = nn.ModuleList(
            [SACA_EncoderLayer(embed_dim, n_heads) for _ in range(n_layers)]
        )

        # 3. Decoder (Pointer Network)

        self.W1_embed = nn.Linear(embed_dim, embed_dim, bias=False) # h_k
        self.W2_decode = nn.Linear(embed_dim, embed_dim, bias=False) # h_i^d
        self.v = nn.Linear(embed_dim, 1, bias=False)                 # v^T

    def supervised_loss(
        self,
        label_route_local,
        station_weights,
        station_coords,             # (B, N_DEST, 2)
        start_coord,                # (B, 2)
        *,
        return_details=False
    ):


        loc = station_coords
        start = start_coord
        B, N, _ = loc.shape
        weight = station_weights


        node_features = torch.cat([loc, weight.unsqueeze(-1)], dim=2)


        start_weight = torch.zeros(B, 1, 1, device=loc.device, dtype=loc.dtype)
        start_features = torch.cat([start.unsqueeze(1), start_weight], dim=2)


        all_nodes_features = torch.cat([start_features, node_features], dim=1)   # (B, N+1, 3)


        enc = self.node_embed(all_nodes_features)

        for layer in self.encoder_layers:
            enc = layer(enc)

        W1_h = self.W1_embed(enc)                                # (B, N+1, D)
        mask = torch.zeros(B, N+1, dtype=torch.bool, device=loc.device)
        mask[:, 0] = True
        dec_h = enc[:, 0, :]
        batch_idx = torch.arange(B, device=loc.device)

        gt_full = label_route_local.long() + 1
        step_logps = []
        for t in range(N):
            scores = self.v(torch.tanh(W1_h + self.W2_decode(dec_h).unsqueeze(1))).squeeze(-1)
            scores = scores.masked_fill(mask.clone(), -float('inf'))
            log_p = F.log_softmax(scores, dim=1)

            idx_t = gt_full[:, t]
            step_logps.append(log_p[batch_idx, idx_t])

            mask[batch_idx, idx_t] = True
            dec_h = enc[batch_idx, idx_t, :]

        logp_mat = torch.stack(step_logps, dim=1)                # (B, N)
        loss_sup = (-logp_mat.mean()).mean()                 # NLL

        if return_details:
            return loss_sup, {"per_step_logp": logp_mat, "sum_logp": logp_mat.sum(dim=1)}
        return loss_sup

    def forward(self, loc, start, weight):


        B, N_DEST, _ = loc.shape


        node_features = torch.cat([loc, weight.unsqueeze(-1)], dim=2)


        start_weight = torch.zeros(B, 1, 1, device=loc.device, dtype=loc.dtype)
        start_features = torch.cat([start.unsqueeze(1), start_weight], dim=2)


        all_nodes_features = torch.cat([start_features, node_features], dim=1)


        all_nodes_embed = self.node_embed(all_nodes_features)

        # 3. Encoder (B, N+1, D)
        encoded_nodes = all_nodes_embed
        for layer in self.encoder_layers:
            encoded_nodes = layer(encoded_nodes)

        # 4. Decoder (Sequential Pointing)
        # (B, N+1, D)
        W1_h = self.W1_embed(encoded_nodes)


        mask = torch.zeros(B, N_DEST + 1, device=loc.device, dtype=torch.bool)
        route_indices = []
        log_probs = []


        decoder_hidden_state = encoded_nodes[:, 0, :]
        mask[:, 0] = True


        for _ in range(N_DEST):
            W2_h_d = self.W2_decode(decoder_hidden_state) # (B, D)


            # (B, N+1, D) + (B, 1, D) -> (B, N+1, D)
            scores_raw = torch.tanh(W1_h + W2_h_d.unsqueeze(1))


            scores = self.v(scores_raw).squeeze(-1) # (B, N+1)


            scores = scores.masked_fill(mask.clone(), -float('inf'))


            probs = F.softmax(scores, dim=1)
            log_p = F.log_softmax(scores, dim=1)


            if self.training:

                next_node_idx = torch.multinomial(probs, 1).squeeze(1) # (B,)
            else:

                next_node_idx = torch.argmax(probs, dim=1) # (B,)


            batch_indices = torch.arange(B, device=loc.device)
            log_probs.append(log_p[batch_indices, next_node_idx])


            route_indices.append(next_node_idx)
            mask[batch_indices, next_node_idx] = True
            decoder_hidden_state = encoded_nodes[batch_indices, next_node_idx, :]

        # (B, N_DEST)
        route_indices = torch.stack(route_indices, dim=1)
        log_probs = torch.stack(log_probs, dim=1)


        return route_indices, log_probs

def haversine_torch(lon1, lat1, lon2, lat2, device):


    R = 6371.0

    lon1, lat1, lon2, lat2 = map(torch.deg2rad, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = torch.sin(dlat / 2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2)**2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))

    distance = R * c
    return distance

def calculate_avg_passenger_distance(route_indices, loc, start, weight, device):


    B, N_DEST, _ = loc.shape

    # (B, N_DEST + 1, 2)
    all_nodes_coords = torch.cat([start.unsqueeze(1), loc], dim=1)


    start_node = torch.zeros(B, 1, dtype=torch.long, device=device)
    route_indices_full = torch.cat([start_node, route_indices], dim=1)

    total_passenger_distance = torch.zeros(B, device=device)



    # (B, N+1, 1) -> (B, N+1, 2)
    idx_expanded = route_indices_full.unsqueeze(-1).expand(-1, -1, 2)
    sorted_coords = torch.gather(all_nodes_coords, 1, idx_expanded)


    # A = sorted_coords[:, :-1] (B, N, 2)
    # B = sorted_coords[:, 1:] (B, N, 2)
    segment_dists = haversine_torch(
        sorted_coords[:, :-1, 0].reshape(-1), # lon1
        sorted_coords[:, :-1, 1].reshape(-1), # lat1
        sorted_coords[:, 1:, 0].reshape(-1),  # lon2
        sorted_coords[:, 1:, 1].reshape(-1),  # lat2
        device
    ).reshape(B, N_DEST)


    # [dist1, dist1+dist2, dist1+dist2+dist3, ...]
    cumulative_bus_dist = torch.cumsum(segment_dists, dim=1)


    dest_indices = route_indices[:, :] - 1 # (B, N_DEST)
    # (B, N_DEST, 1)
    idx_expanded = dest_indices.unsqueeze(-1).expand(-1, -1, 1)

    sorted_weights = torch.gather(weight.unsqueeze(-1), 1, idx_expanded).squeeze(-1)


    # (B, N) * (B, N) -> (B, N)
    total_passenger_distance = (sorted_weights * cumulative_bus_dist).sum(dim=1)


    total_passengers = weight.sum(dim=1)


    avg_passenger_dist = total_passenger_distance / (total_passengers + 1e-6)

    return avg_passenger_dist


def load_supervised_data(file_paths):


    data = []
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            samples = json.load(f)
            samples = [{
                'loc': np.array(sample['station_coords'], dtype=np.float32),
                'weight': np.array(sample['passengers'], dtype=np.float32),
                'label': np.array(sample['local_optimal_order'], dtype=np.float32),
                'start': np.array(START, dtype=np.float32),
                'label_cost': sample['avg_cost_per_passenger']
            } for sample in samples]
            data.append(samples)
    return data

def compute_normalization_stats(train_data_groups):


    all_features = []

    for group in train_data_groups:
        for sample in group:

            loc_features = sample['loc']  # (N, 2)

            weight_features = sample['weight'][:, np.newaxis]  # (N, 1)


            node_feats = np.concatenate([loc_features, weight_features], axis=1)  # (N, 3)
            all_features.append(node_feats)


            start_feat = np.array(
                [[sample['start'][0], sample['start'][1], 0.0]],
                dtype=np.float32
            )  # (1, 3)
            all_features.append(start_feat)


    full_dataset_features = np.concatenate(all_features, axis=0)  # (Total_Nodes, 3)


    mean = np.mean(full_dataset_features, axis=0)
    std = np.std(full_dataset_features, axis=0)


    std = np.where(std == 0, 1e-6, std)

    stats = {
        'mean': torch.tensor(mean, dtype=torch.float32),
        'std': torch.tensor(std, dtype=torch.float32)
    }
    return stats

def getbatch_supervised(data_groups, batch_size, device, stats):


    file_idx = random.randrange(len(data_groups))
    dataset = data_groups[file_idx]
    replace = len(dataset) < batch_size
    samples = np.random.choice(dataset, batch_size, replace=replace)


    loc_list = []
    weight_list = []
    label_list = []
    start_list = []

    for s in samples:
        loc = s['loc']       # (N, 2)
        weight = s['weight'] # (N,)
        label = s['label']   # (N,)
        N = loc.shape[0]


        perm = np.random.permutation(N)


        loc_shuffled = loc[perm]
        weight_shuffled = weight[perm]


        inv_perm = np.empty(N, dtype=np.int64)
        inv_perm[perm] = np.arange(N)


        label_shuffled = inv_perm[label.astype(np.int64)]


        loc_list.append(loc_shuffled)
        weight_list.append(weight_shuffled)
        label_list.append(label_shuffled)
        start_list.append(s['start'])


    loc_batch   = torch.as_tensor(np.stack(loc_list, axis=0), device=device)
    weight_batch = torch.as_tensor(np.stack(weight_list, axis=0), device=device)
    start_batch = torch.as_tensor(np.stack(start_list, axis=0), device=device)
    label_batch = torch.as_tensor(np.stack(label_list, axis=0), device=device, dtype=torch.long)


    mean = stats['mean'].to(device)  # (3,)
    std = stats['std'].to(device)    # (3,)

    # (B, N, 2) - (1, 1, 2) / (1, 1, 2)
    loc_batch = (loc_batch - mean[None, None, :2]) / std[None, None, :2]
    # (B, N) - (1,) / (1,)
    weight_batch = (weight_batch - mean[2]) / std[2]
    # (B, 2) - (1, 2) / (1, 2)
    start_batch = (start_batch - mean[None, :2]) / std[None, :2]

    return loc_batch, start_batch, weight_batch, label_batch

@torch.no_grad()
def evaluate_supervised(model, data_groups, batch_size, device, stats):

    model.eval()
    total_seq_loss = 0.0
    total_seqs = 0
    token_nll_sum = 0.0
    token_cnt = 0

    mean = stats['mean'].to(device)  # (3,)
    std = stats['std'].to(device)    # (3,)

    for dataset in data_groups:
        for i in range(0, len(dataset), batch_size):
            samples = dataset[i:i+batch_size]
            loc_np = np.stack([s['loc'] for s in samples], axis=0)

            weight_np = np.stack([s['weight'] for s in samples], axis=0)
            start_np = np.stack([s['start'] for s in samples], axis=0)
            label_np = np.stack([s['label'] for s in samples], axis=0).astype(np.int64)

            loc   = torch.as_tensor(loc_np, device=device)

            weight = torch.as_tensor(weight_np, device=device)
            start = torch.as_tensor(start_np, device=device)
            label = torch.as_tensor(label_np, device=device)


            loc = (loc - mean[None, None, :2]) / std[None, None, :2]
            weight = (weight - mean[2]) / std[2]
            start = (start - mean[None, :2]) / std[None, :2]

            loss, details = model.supervised_loss(
                label_route_local=label,
                station_coords=loc,
                station_weights=weight,
                start_coord=start,
                return_details=True
            )
            total_seq_loss += loss.item() * len(samples)
            total_seqs += len(samples)


            token_nll_sum += (-details["per_step_logp"]).sum().item()
            token_cnt += loc.shape[0] * loc.shape[1]

    avg_seq_loss = total_seq_loss / max(1, total_seqs)
    avg_token_nll = token_nll_sum / max(1, token_cnt)
    return avg_seq_loss, avg_token_nll

def split_grouped_data(data_groups, val_ratio=0.1, seed=42):

    rng = np.random.default_rng(seed)
    train_groups, val_groups = [], []
    for dataset in data_groups:
        idx = np.arange(len(dataset))
        rng.shuffle(idx)
        val_size = max(1, int(len(dataset) * val_ratio))
        val_idx = idx[:val_size]
        train_idx = idx[val_size:]
        train_groups.append([dataset[i] for i in train_idx])
        val_groups.append([dataset[i] for i in val_idx])
    return train_groups, val_groups

@torch.no_grad()
def evaluate_model_cost(model, data_groups, batch_size, device, stats):


    model.eval()


    mean = stats['mean'].to(device)
    std = stats['std'].to(device)

    all_pred_costs = []
    all_label_costs = []

    print("Evaluating route costs...")

    for dataset in data_groups:

        for i in range(0, len(dataset), batch_size):
            samples = dataset[i:i+batch_size]


            loc_list = [s['loc'] for s in samples]
            weight_list = [s['weight'] for s in samples]
            start_list = [s['start'] for s in samples]


            label_cost_np = np.array([s['label_cost'] for s in samples])


            loc_orig = torch.tensor(np.stack(loc_list), device=device, dtype=torch.float32)
            weight_orig = torch.tensor(np.stack(weight_list), device=device, dtype=torch.float32)
            start_orig = torch.tensor(np.stack(start_list), device=device, dtype=torch.float32)
            label_cost_tensor = torch.tensor(label_cost_np, device=device, dtype=torch.float32)


            loc_norm = (loc_orig - mean[None, None, :2]) / std[None, None, :2]
            weight_norm = (weight_orig - mean[2]) / std[2]
            start_norm = (start_orig - mean[None, :2]) / std[None, :2]


            route_indices, _ = model(loc_norm, start_norm, weight_norm)


            pred_cost_tensor = calculate_avg_passenger_distance(
                route_indices, loc_orig, start_orig, weight_orig, device
            )


            all_pred_costs.append(pred_cost_tensor.cpu())
            all_label_costs.append(label_cost_tensor.cpu())

    print("Evaluation completed; computing metrics...")


    final_pred_costs = torch.cat(all_pred_costs)
    final_label_costs = torch.cat(all_label_costs)


    absolute_error = final_pred_costs - final_label_costs


    approximation_ratio = final_pred_costs / (final_label_costs + 1e-6)


    print("\n--- Model Cost Evaluation Report ---")
    print(f"Total samples: {len(final_pred_costs)}")

    print("\n[ Mean Numeric Error (Predicted - Label) ]")
    print(f"  Mean error: {absolute_error.mean().item():.4f} km")
    print(f"  Predicted costs differ from labels by {absolute_error.mean().item():.4f} km on average.")

    print("\n[ Mean Approximation Ratio (Predicted / Label) ]")
    print(f"  Mean ratio: {approximation_ratio.mean().item():.4f}")
    print(f"  Predicted cost is {approximation_ratio.mean().item() * 100:.2f}% of label cost on average.")
    print(f"  Approximate optimality gap: {(approximation_ratio.mean().item() - 1) * 100:.2f}%")
    print("----------------------------\n")

    return absolute_error.mean().item(), approximation_ratio.mean().item()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_val_loss = float('inf')
    SAVE_PATH = ROUTING_MODEL_DIR / "best_model.pth"
    STATS_PATH = ROUTING_MODEL_DIR / "normalization_stats.pt"


    model = AttentionRouteModel(EMBEDDING_DIM, N_HEADS, N_LAYERS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_mult=1,
        T_0=RESTART_EVERY,
        eta_min=1e-9
    )


    file_paths = [REPO_ROOT / "data" / "routing_supervised" / f"supervised_dataset_{n}_stations.json"
                  for n in range(5, 11)]
    data = load_supervised_data(file_paths)
    train_data, val_data = split_grouped_data(data, val_ratio=0.1, seed=42)


    STEPS_PER_EPOCH = sum(len(g) // BATCH_SIZE for g in train_data)


    if wandb is not None:
        wandb.init(
            project="AM",
            config=dict(
                embedding_dim=EMBEDDING_DIM, n_heads=N_HEADS, n_layers=N_LAYERS,
                lr=LEARNING_RATE, batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS,
                steps_per_epoch=STEPS_PER_EPOCH
            )
        )
        wandb.watch(model, log="gradients", log_freq=100)


    stats = compute_normalization_stats(train_data)


    for epoch in range(NUM_EPOCHS):
        model.train()
        running_seq_loss = 0.0
        token_nll_sum = 0.0
        token_cnt = 0

        for _ in range(STEPS_PER_EPOCH):
            loc, start, weight, label = getbatch_supervised(train_data, BATCH_SIZE, device, stats)

            loss, details = model.supervised_loss(
                label_route_local=label,
                station_coords=loc,
                station_weights=weight,
                start_coord=start,
                return_details=True
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_seq_loss += loss.item()
            token_nll_sum += (-details["per_step_logp"]).sum().item()
            token_cnt += loc.shape[0] * loc.shape[1]

        scheduler.step()

        train_seq_loss = running_seq_loss / max(1, STEPS_PER_EPOCH)
        train_token_nll = token_nll_sum / max(1, token_cnt)


        val_seq_loss, val_token_nll = evaluate_supervised(model, val_data, BATCH_SIZE, device, stats)


        if val_token_nll < best_val_loss:
            best_val_loss = val_token_nll
            torch.save(model.state_dict(), SAVE_PATH)
            torch.save(stats, STATS_PATH)
            best_epoch = epoch


        if wandb is not None:
            wandb.log({
                "train/loss_seq": train_seq_loss,
                "train/nll_token": train_token_nll,
                "val/loss_seq": val_seq_loss,
                "val/nll_token": val_token_nll,
                "lr": optimizer.param_groups[0]["lr"],
                "epoch": epoch + 1
            })
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
                f"TrainSeqLoss: {train_seq_loss:.4f} | TrainTokNLL: {train_token_nll:.4f} | "
                f"ValSeqLoss: {val_seq_loss:.4f} | ValTokNLL: {val_token_nll:.4f}")
    print(f"Training complete. Best validation token NLL: {best_val_loss:.4f} at epoch {best_epoch+1}")

    if wandb is not None:
        wandb.finish()
