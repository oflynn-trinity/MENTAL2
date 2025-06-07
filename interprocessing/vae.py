import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, input_sz: int, compressed_sz: int, dropout: float, layer1_sz = 3000, layer2_sz = 1024):
        super().__init__()

        self.input_sz = input_sz
        self.compressed_sz = compressed_sz

        self.encoder = nn.Sequential(
            nn.Linear(input_sz, layer1_sz),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(layer1_sz, layer2_sz),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.mu = nn.Linear(layer2_sz, compressed_sz)
        self.var = nn.Linear(layer2_sz, compressed_sz)

        self.decoder = nn.Sequential(
            nn.Linear(compressed_sz, layer2_sz),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(layer2_sz, layer1_sz),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(layer1_sz, input_sz),
            nn.ReLU()
        )

    def forward(self, input):
        res = self.encoder(input)

        mu = self.mu(res)
        var = self.var(res)

        dist = torch.distributions.normal.Normal(mu, torch.exp(0.5*var))
        sample = dist.rsample()

        return self.decoder(sample)