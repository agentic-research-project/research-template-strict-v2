"""Few-shot learning model skeleton — prototypical network."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_model(config: dict) -> nn.Module:
    return ProtoNet(config)


class ProtoNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        ch = config.get("channels", 64)
        emb_dim = config.get("embedding_dim", 64)
        in_ch = config.get("in_channels", 3)

        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch, emb_dim),
        )

    def forward(self, x):
        return self.encoder(x)

    def compute_prototypes(self, support, support_labels, n_way):
        """support embeddings → class prototypes."""
        embeddings = self.encoder(support)
        prototypes = []
        for c in range(n_way):
            mask = support_labels == c
            prototypes.append(embeddings[mask].mean(dim=0))
        return torch.stack(prototypes)

    def classify(self, query, prototypes):
        """query embeddings → distances to prototypes → logits."""
        q_emb = self.encoder(query)
        dists = torch.cdist(q_emb, prototypes)
        return -dists  # negative distance as logits
