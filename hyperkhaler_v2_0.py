# ============================================================
#  Hyperkähler Stack v2.0 — Quaternionic Temporal RNN Model
#  Author: Michael Listrom
#  Description:
#     This version introduces quaternion-aware temporal recurrence.
#     The HyperkählerRNN captures continuous latent memory in
#     quaternion space, enabling smooth orientation evolution across
#     sequential embeddings.
#
#  Stack Architecture:
#     Input → Autoencoder → RNN → Transformer → Fusion → SLERP
#
#  Key Features:
#     • Quaternion-normalized latent evolution
#     • Recurrent temporal coherence via GRU
#     • SLERP-based latent interpolation for trajectory synthesis
#     • Headless runner with persistent output saving
# ============================================================

import torch
import torch.nn as nn
import os

# --------------------------
# Quaternion Utilities
# --------------------------
def normalize_quaternion(q, eps=1e-8):
    norm = torch.norm(q, dim=-1, keepdim=True) + eps
    return q / norm

def quaternion_mul(q, r):
    w1, x1, y1, z1 = q.unbind(-1)
    w2, x2, y2, z2 = r.unbind(-1)
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return torch.stack([w, x, y, z], dim=-1)

def slerp(q0, q1, t):
    """
    Spherical linear interpolation.
    q0, q1: (..., 4) unit quaternions
    t: scalar or tensor broadcastable to (..., 1)
    """
    # ensure t is a torch tensor on the same device
    if not torch.is_tensor(t):
        t = torch.tensor(t, device=q0.device, dtype=q0.dtype)
    else:
        t = t.to(device=q0.device, dtype=q0.dtype)

    dot = torch.sum(q0 * q1, dim=-1, keepdim=True)
    dot = torch.clamp(dot, -1.0, 1.0)

    # if t is scalar make it broadcastable
    if t.dim() == 0:
        t = t.unsqueeze(0).unsqueeze(-1)  # shape (1,1) -> will broadcast
    elif t.dim() == 1:
        t = t.unsqueeze(-1)  # (N) -> (N,1)

    # compute angles
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    small = sin_theta.abs() < 1e-6

    s0 = torch.where(small, 1.0 - t, torch.sin((1 - t) * theta) / sin_theta)
    s1 = torch.where(small, t, torch.sin(t * theta) / sin_theta)
    out = s0 * q0 + s1 * q1
    return normalize_quaternion(out)

# --------------------------
# Hyperkähler Modules
# --------------------------
class HyperkählerAutoencoder(nn.Module):
    def __init__(self, input_dim=16, latent_dim=8):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim * 4)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim * 4, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        # x: (batch, seq, input_dim)
        q_latent = self.encoder(x).view(*x.shape[:-1], -1, 4)  # (batch, seq, latent, 4)
        q_latent = normalize_quaternion(q_latent)
        flat_latent = q_latent.view(*x.shape[:-1], -1)  # (batch, seq, latent*4)
        recon = self.decoder(flat_latent)
        return recon, q_latent

class HyperkählerRNN(nn.Module):
    """
    Recurrent module that operates on flattened quaternion latents.
    Uses a GRU and maps outputs back to quaternion shape + renormalize.
    """
    def __init__(self, latent_dim=8, hidden_dim=None, num_layers=1):
        super().__init__()
        self.latent_dim = latent_dim
        in_dim = latent_dim * 4
        self.hidden_dim = hidden_dim if hidden_dim is not None else in_dim
        self.gru = nn.GRU(input_size=in_dim, hidden_size=self.hidden_dim,
                          num_layers=num_layers, batch_first=True)
        # project back to quaternion space size
        self.project_back = nn.Linear(self.hidden_dim, in_dim)

    def forward(self, q_latent):
        # q_latent: (batch, seq, latent, 4)
        b, seq, latent, four = q_latent.shape
        assert four == 4 and latent == self.latent_dim
        flat = q_latent.view(b, seq, -1)  # (b, seq, latent*4)
        gru_out, _ = self.gru(flat)       # (b, seq, hidden_dim)
        proj = self.project_back(gru_out) # (b, seq, latent*4)
        q_out = proj.view(b, seq, latent, 4)
        return normalize_quaternion(q_out)

class HyperkählerTransformer(nn.Module):
    def __init__(self, latent_dim=8, n_heads=2):
        super().__init__()
        self.latent_dim = latent_dim
        self.attn = nn.MultiheadAttention(embed_dim=latent_dim * 4, num_heads=n_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(latent_dim * 4, latent_dim * 4),
            nn.ReLU(),
            nn.Linear(latent_dim * 4, latent_dim * 4)
        )

    def forward(self, q_latent):
        # q_latent: (batch, seq, latent, 4)
        b, l, latent, _ = q_latent.shape
        flat = q_latent.view(b, l, -1)  # (b, seq, latent*4)
        attn_out, _ = self.attn(flat, flat, flat)
        ff_out = self.ff(attn_out)
        out = ff_out.view(b, l, latent, 4)
        return normalize_quaternion(out)

class HyperkählerFusion(nn.Module):
    def __init__(self, latent_dim=8):
        super().__init__()
        self.latent_dim = latent_dim
        self.freqs = nn.Parameter(torch.linspace(0.1, 1.0, latent_dim).unsqueeze(0))
        self.phase = nn.Parameter(torch.zeros(latent_dim).unsqueeze(0))

    def forward(self, q_latent):
        # q_latent: (batch, seq, latent, 4)
        t = torch.linspace(0, 1, q_latent.shape[1], device=q_latent.device).unsqueeze(0).unsqueeze(-1)  # (1, seq, 1)
        sin_embed = torch.sin(t * self.freqs.unsqueeze(1) + self.phase.unsqueeze(1)).unsqueeze(-1)  # (1, seq, latent, 1)
        # broadcast and add
        return normalize_quaternion(q_latent + sin_embed)

# --------------------------
# Headless Runner (v2.0)
# --------------------------
def run_headless(batch_size=4, seq_len=10, input_dim=16, latent_dim=8, output_dir='outputs_v2'):
    os.makedirs(output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ae = HyperkählerAutoencoder(input_dim=input_dim, latent_dim=latent_dim).to(device)
    rnn = HyperkählerRNN(latent_dim=latent_dim).to(device)
    transformer = HyperkählerTransformer(latent_dim=latent_dim).to(device)
    fusion = HyperkählerFusion(latent_dim=latent_dim).to(device)

    # construct a simple orientation sequence as input (example)
    angles = torch.linspace(0, 3.1415, seq_len, device=device).unsqueeze(0).repeat(batch_size, 1)
    q1 = torch.stack([torch.cos(angles / 2),
                      torch.sin(angles / 2),
                      torch.zeros_like(angles),
                      torch.zeros_like(angles)], dim=-1)  # (batch, seq, 4)
    x = q1.view(batch_size, seq_len, -1).to(device)  # (batch, seq, input_dim) - works if input_dim==4 here

    recon, q_latent = ae(x)  # q_latent: (b, seq, latent, 4)
    torch.save(recon.cpu(), f'{output_dir}/reconstructed.pt')
    torch.save(q_latent.cpu(), f'{output_dir}/latent_quaternions_pre_rnn.pt')

    # RNN (temporal memory)
    q_rnn = rnn(q_latent)
    torch.save(q_rnn.cpu(), f'{output_dir}/latent_quaternions_post_rnn.pt')

    # Transformer + Fusion
    q_trans = transformer(q_rnn)
    torch.save(q_trans.cpu(), f'{output_dir}/transformed_latent.pt')

    q_fused = fusion(q_trans)
    torch.save(q_fused.cpu(), f'{output_dir}/fused_latent.pt')

    # SLERP trajectories between first and last timestep per-batch
    t_vals = torch.linspace(0, 1, 20, device=device)
    slerp_trajs = []
    for b in range(batch_size):
        # q_fused[b,0] and q_fused[b,-1] are shapes (latent,4)
        slerp_seq = torch.stack([slerp(q_fused[b, 0], q_fused[b, -1], t) for t in t_vals], dim=0)  # (20, latent, 4)
        slerp_trajs.append(slerp_seq)
    slerp_trajs = torch.stack(slerp_trajs, dim=0)  # (batch, 20, latent, 4)
    torch.save(slerp_trajs.cpu(), f'{output_dir}/slerp_trajectories.pt')

    print(f'Outputs saved in {output_dir}')

if __name__ == "__main__":
    run_headless()