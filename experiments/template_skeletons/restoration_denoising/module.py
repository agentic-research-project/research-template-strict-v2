"""Denoising training module — PSNR/SSIM evaluation."""
import json
import math
import torch
import torch.nn as nn


class TrainingModule:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.criterion = nn.L1Loss()

    def train_epoch(self, fabric, train_loader, optimizer):
        self.model.train()
        total_loss = 0.0
        for noisy, clean in train_loader:
            optimizer.zero_grad()
            output = self.model(noisy)
            loss = self.criterion(output, clean)
            fabric.backward(loss)
            optimizer.step()
            total_loss += loss.item()
        return {"train_loss": total_loss / len(train_loader)}

    def val_epoch(self, fabric, val_loader):
        self.model.eval()
        psnr_sum, ssim_sum, n = 0.0, 0.0, 0
        with torch.no_grad():
            for noisy, clean in val_loader:
                output = self.model(noisy).clamp(0, 1)
                mse = ((output - clean) ** 2).mean().item()
                psnr = 10 * math.log10(1.0 / max(mse, 1e-10))
                psnr_sum += psnr
                n += 1
        return {"psnr": psnr_sum / max(n, 1), "ssim": 0.0}  # SSIM placeholder
