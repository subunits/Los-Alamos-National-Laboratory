# ===============================================================
# Quaternionic-Kahler NLP Demo (Colab-Ready)
# Author: Michael Listrom / subunits adaptation
# ===============================================================
# This script implements a quaternionic Transformer with
# Kahler-inspired attention and runs a synthetic NLP demo.
# ===============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------
# Quaternion Algebra
# ---------------------------------------------------------------
def quat_mul(a, b):
    """Quaternion multiplication for tensors [batch, seq, 4*d]."""
    a_r, a_i, a_j, a_k = torch.chunk(a, 4, dim=-1)
    b_r, b_i, b_j, b_k = torch.chunk(b, 4, dim=-1)
    r = a_r*b_r - a_i*b_i - a_j*b_j - a_k*b_k
    i = a_r*b_i + a_i*b_r + a_j*b_k - a_k*b_j
    j = a_r*b_j - a_i*b_k + a_j*b_r + a_k*b_i
    k = a_r*b_k + a_i*b_j - a_j*b_i + a_k*b_r
    return torch.cat([r, i, j, k], dim=-1)

def quat_norm(q, eps=1e-6):
    """Normalize quaternion tensor."""
    r, i, j, k = torch.chunk(q, 4, dim=-1)
    norm = torch.sqrt(r**2 + i**2 + j**2 + k**2 + eps)
    return torch.cat([r/norm, i/norm, j/norm, k/norm], dim=-1)

# ---------------------------------------------------------------
# Quaternionic Linear Layer
# ---------------------------------------------------------------
class QuaternionLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = nn.Linear(in_features, out_features, bias=bias)
        self.i = nn.Linear(in_features, out_features, bias=bias)
        self.j = nn.Linear(in_features, out_features, bias=bias)
        self.k = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        xr, xi, xj, xk = torch.chunk(x, 4, dim=-1)
        r = self.r(xr) - self.i(xi) - self.j(xj) - self.k(xk)
        i = self.r(xi) + self.i(xr) + self.j(xk) - self.k(xj)
        j = self.r(xj) - self.i(xk) + self.j(xr) + self.k(xi)
        k = self.r(xk) + self.i(xj) - self.j(xi) + self.k(xr)
        return torch.cat([r, i, j, k], dim=-1)

# ---------------------------------------------------------------
# Kahler-Inspired Attention (Geometric Modulation)
# ---------------------------------------------------------------
class KahlerAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.scale = (embed_dim // num_heads) ** -0.5

        self.q_proj = QuaternionLinear(embed_dim, embed_dim)
        self.k_proj = QuaternionLinear(embed_dim, embed_dim)
        self.v_proj = QuaternionLinear(embed_dim, embed_dim)
        self.out_proj = QuaternionLinear(embed_dim, embed_dim)
        self.J = nn.Parameter(torch.randn(embed_dim) * 0.01)  # Kahler modulator

    def forward(self, x):
        B, T, D = x.size()
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = quat_norm(q)
        k = quat_norm(k)

        q_ = q.view(B, T, self.num_heads, D // self.num_heads)
        k_ = k.view(B, T, self.num_heads, D // self.num_heads)
        v_ = v.view(B, T, self.num_heads, D // self.num_heads)

        attn_scores = torch.einsum("bthd,bshd->bhts", q_, k_) * self.scale
        # Kahler modulation (symplectic twist)
        attn_scores = attn_scores + torch.sin(self.J[:attn_scores.size(-1)]).mean() * 0.1
        attn_weights = F.softmax(attn_scores, dim=-1)
        out = torch.einsum("bhts,bshd->bthd", attn_weights, v_)
        out = out.contiguous().view(B, T, D)
        return self.out_proj(out)

# ---------------------------------------------------------------
# Quaternionic Transformer Block
# ---------------------------------------------------------------
class QuaternionicTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim):
        super().__init__()
        self.attn = KahlerAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            QuaternionLinear(embed_dim, mlp_dim),
            nn.ReLU(),
            QuaternionLinear(mlp_dim, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

# ---------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------
class QuaternionicKahlerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, num_heads=4, mlp_dim=128, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.qexpand = nn.Linear(embed_dim, 4*embed_dim)
        self.blocks = nn.ModuleList([
            QuaternionicTransformerBlock(4*embed_dim, num_heads, 4*mlp_dim)
            for _ in range(num_layers)
        ])
        self.out = nn.Linear(4*embed_dim, vocab_size)

    def forward(self, x):
        e = self.embed(x)
        q = self.qexpand(e)
        for blk in self.blocks:
            q = blk(q)
        logits = self.out(q)
        return logits

# ---------------------------------------------------------------
# Synthetic NLP Demo
# ---------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    vocab_size = 500
    seq_len = 12
    batch_size = 3

    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    model = QuaternionicKahlerModel(vocab_size)
    y = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", y.shape)
    print("Output sample:", y[0, 0, :5])