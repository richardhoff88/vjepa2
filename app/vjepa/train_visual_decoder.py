import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from vjepa2.src.models.visual_decoder import VisualDecoder
from vjepa2.src.models import vjepa2_vitg_fpc64_384  # example pretrained encoder


# ----------------------------
# Config
# ----------------------------
BATCH_SIZE = 16
EPOCHS = 5
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------
# Dataset (CIFAR10 for demo)
# Replace with video frames dataset later
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)


# ----------------------------
# Load pretrained encoder (frozen)
# ----------------------------
encoder = vjepa2_vitg_fpc64_384(pretrained=True)  # loads pretrained V-JEPA2 model
encoder.eval()
for p in encoder.parameters():
    p.requires_grad = False
encoder.to(DEVICE)


# ----------------------------
# Visual Decoder
# ----------------------------
# NOTE: match latent_dim to encoder output dimension
decoder = VisualDecoder(latent_dim=1024, out_channels=3).to(DEVICE)

criterion = nn.MSELoss()
optimizer = optim.Adam(decoder.parameters(), lr=LR)


# ----------------------------
# Training Loop
# ----------------------------
for epoch in range(EPOCHS):
    decoder.train()
    total_loss = 0.0

    for imgs, _ in train_loader:
        imgs = imgs.to(DEVICE)

        # 1. Get encoder features
        with torch.no_grad():
            feats = encoder(imgs)  # shape: [B, latent_dim, H, W] (depends on encoder)

        # 2. Reconstruct with decoder
        recons = decoder(feats)

        # 3. Compute loss
        loss = criterion(recons, imgs)

        # 4. Update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f}")

    # Save sample reconstructions
    if (epoch + 1) % 1 == 0:
        torch.save(decoder.state_dict(), f"visual_decoder_epoch{epoch+1}.pth")

