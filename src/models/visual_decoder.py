import torch
import torch.nn as nn

class VisualDecoder(nn.Module):
    def __init__(self, latent_dim=1024, out_channels=3):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, out_channels, 4, stride=2, padding=1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        return self.decoder(x)
