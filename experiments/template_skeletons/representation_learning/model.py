"""Representation learning model skeleton — encoder + projection head."""
import torch.nn as nn


def build_model(config: dict) -> nn.Module:
    return RepresentationNet(config)


class RepresentationNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        ch = config.get("channels", 64)
        emb_dim = config.get("embedding_dim", 128)
        proj_dim = config.get("projection_dim", 64)
        in_ch = config.get("in_channels", 3)

        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, ch, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch * 2, emb_dim),
        )
        self.projection_head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), nn.ReLU(inplace=True),
            nn.Linear(emb_dim, proj_dim),
        )

    def forward(self, x):
        embedding = self.encoder(x)
        projection = self.projection_head(embedding)
        return {"embedding": embedding, "projection": projection}
