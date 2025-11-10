
import torch
import torch.nn as nn
import os

def normalize_quaternion(q):
    norm = torch.norm(q, dim=-1, keepdim=True) + 1e-8
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
    dot = torch.sum(q0*q1, dim=-1, keepdim=True)
    dot = torch.clamp(dot, -1.0, 1.0)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    small = sin_theta < 1e-6
    s0 = torch.where(small, 1.0 - t, torch.sin((1 - t) * theta) / sin_theta)
    s1 = torch.where(small, t, torch.sin(t * theta) / sin_theta)
    return normalize_quaternion(s0 * q0 + s1 * q1)

class HyperkählerAutoencoder(nn.Module):
    def __init__(self, input_dim=16, latent_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU(), nn.Linear(32, latent_dim*4))
        self.decoder = nn.Sequential(nn.Linear(latent_dim*4, 32), nn.ReLU(), nn.Linear(32, input_dim))
    def forward(self, x):
        q_latent = self.encoder(x).view(*x.shape[:-1], -1, 4)
        q_latent = normalize_quaternion(q_latent)
        flat_latent = q_latent.view(*x.shape[:-1], -1)
        recon = self.decoder(flat_latent)
        return recon, q_latent

class HyperkählerTransformer(nn.Module):
    def __init__(self, latent_dim=8, n_heads=2):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=latent_dim*4, num_heads=n_heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(latent_dim*4, latent_dim*4), nn.ReLU(), nn.Linear(latent_dim*4, latent_dim*4))
    def forward(self, q_latent):
        b, l, latent, _ = q_latent.shape
        flat = q_latent.view(b, l, -1)
        attn_out, _ = self.attn(flat, flat, flat)
        ff_out = self.ff(attn_out)
        out = ff_out.view(b, l, latent, 4)
        return normalize_quaternion(out)

class HyperkählerFusion(nn.Module):
    def __init__(self, latent_dim=8):
        super().__init__()
        self.freqs = nn.Parameter(torch.linspace(0.1, 1.0, latent_dim).unsqueeze(0))
        self.phase = nn.Parameter(torch.zeros(latent_dim).unsqueeze(0))
    def forward(self, q_latent):
        t = torch.linspace(0, 1, q_latent.shape[1]).unsqueeze(0).unsqueeze(-1)
        sin_embed = torch.sin(t * self.freqs.unsqueeze(1) + self.phase.unsqueeze(1)).unsqueeze(-1)
        return normalize_quaternion(q_latent + sin_embed)

def run_headless(batch_size=4, seq_len=10, input_dim=16, latent_dim=8, output_dir='outputs'):
    os.makedirs(output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ae = HyperkählerAutoencoder(input_dim=input_dim, latent_dim=latent_dim).to(device)
    transformer = HyperkählerTransformer(latent_dim=latent_dim).to(device)
    fusion = HyperkählerFusion(latent_dim=latent_dim).to(device)

    angles = torch.linspace(0, 3.1415, seq_len).unsqueeze(0).repeat(batch_size,1)
    q0 = torch.tensor([1.0,0,0,0]).expand(batch_size,1,4)
    q1 = torch.stack([torch.cos(angles/2), torch.sin(angles/2), torch.zeros_like(angles), torch.zeros_like(angles)], dim=-1)
    x = q1.view(batch_size, seq_len, -1).to(device)

    recon, q_latent = ae(x)
    torch.save(recon.cpu(), f'{output_dir}/reconstructed.pt')
    torch.save(q_latent.cpu(), f'{output_dir}/latent_quaternions.pt')

    q_trans = transformer(q_latent)
    torch.save(q_trans.cpu(), f'{output_dir}/transformed_latent.pt')

    q_fused = fusion(q_trans)
    torch.save(q_fused.cpu(), f'{output_dir}/fused_latent.pt')

    t_vals = torch.linspace(0, 1, 20).to(device)
    slerp_trajs = []
    for b in range(batch_size):
        slerp_seq = torch.stack([slerp(q_fused[b,0], q_fused[b,-1], t) for t in t_vals], dim=0)
        slerp_trajs.append(slerp_seq)
    slerp_trajs = torch.stack(slerp_trajs, dim=0)
    torch.save(slerp_trajs.cpu(), f'{output_dir}/slerp_trajectories.pt')

if __name__ == "__main__":
    run_headless()
