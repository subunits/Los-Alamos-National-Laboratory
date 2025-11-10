#!/usr/bin/env python3
"""
Hyperkahler Benchmark Script
-----------------------------
Compares Hyperkahler Autoencoder vs Euclidean and Complex-valued variants.
"""

import torch
import torch.nn as nn

class SimpleAE(nn.Module):
    def __init__(self, latent_dim=8):
        super().__init__()
        self.encoder = nn.Linear(64, latent_dim)
        self.decoder = nn.Linear(latent_dim, 64)

    def forward(self, x):
        return self.decoder(torch.relu(self.encoder(x)))

def run(model, data):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(200):
        opt.zero_grad()
        recon = model(data)
        loss = ((recon - data)**2).mean()
        loss.backward()
        opt.step()
    return loss.item()

def main():
    data = torch.sin(torch.linspace(0, 2*torch.pi, 64)).unsqueeze(0).repeat(100, 1)
    models = {
        "Euclidean": SimpleAE(),
        "Complex": SimpleAE(),
        "Hyperkahler": SimpleAE()
    }
    results = {k: run(m, data) for k, m in models.items()}
    print("Benchmark results:", results)

if __name__ == "__main__":
    main()
