"""Super-resolution training module."""
import math, torch, torch.nn as nn


class TrainingModule:
    def __init__(self, model, config):
        self.model = model
        self.criterion = nn.L1Loss()

    def train_epoch(self, fabric, loader, optimizer):
        self.model.train()
        total = 0.0
        for lr, hr in loader:
            optimizer.zero_grad()
            sr = self.model(lr)
            loss = self.criterion(sr, hr)
            fabric.backward(loss)
            optimizer.step()
            total += loss.item()
        return {"train_loss": total / len(loader)}

    def val_epoch(self, fabric, loader):
        self.model.eval()
        psnr_sum, n = 0.0, 0
        with torch.no_grad():
            for lr, hr in loader:
                sr = self.model(lr).clamp(0, 1)
                mse = ((sr - hr) ** 2).mean().item()
                psnr_sum += 10 * math.log10(1.0 / max(mse, 1e-10))
                n += 1
        return {"psnr": psnr_sum / max(n, 1), "ssim": 0.0}
