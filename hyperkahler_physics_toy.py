#!/usr/bin/env python3
"""
Hyperkahler Physics Toy Problem
--------------------------------
A minimal PyTorch demo of a Hyperkahler-inspired quaternionic autoencoder applied to a simple PDE-like dataset.
Colab-friendly and lightweight.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

# Quaternion utilities
def normalize_quaternion(q):
    norm = torch.norm(q, dim=-1, keepdim=True) + 1e-8
    return q / norm

def quaternion_mul(q, r):
    w1, x1, y1, z1 = q.unbind(-1)
    w2, x2, y2, z2 = r.unbind(-1)
    return torch.stack((w1*w2 - x1*x2 - y1*y2 - z1*z2,
                        w1*x2 + x1*w2 + y1*z2 - z1*y2,
                        w1*y2 - x1*z2 + y1*w2 + z1*x2,
                        w1*z2 + x1*y2 - y1*x2 + z1*w2), dim=-1)

class QuaternionAutoencoder(nn.Module):
    def __init__(self, latent_dim=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim*4)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim*4, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.Sigmoid()
        )
        self.latent_dim = latent_dim

    def forward(self, x):
        z = self.encoder(x)
        q = normalize_quaternion(z.view(-1, 4))
        return self.decoder(q.view(-1, self.latent_dim*4))

def main():
    x = torch.linspace(0, 1, 64)
    data = torch.sin(2 * torch.pi * x).unsqueeze(0).repeat(100, 1)
    model = QuaternionAutoencoder()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(1000):
        opt.zero_grad()
        recon = model(data)
        loss = ((recon - data)**2).mean()
        loss.backward()
        opt.step()
    plt.plot(data[0].detach(), label="Original")
    plt.plot(recon[0].detach(), label="Reconstruction")
    plt.legend()
    plt.title("Hyperkahler Autoencoder Reconstruction")
    plt.savefig("physics_toy_result.png")
    print("Saved physics_toy_result.png")

if __name__ == "__main__":
    main()
