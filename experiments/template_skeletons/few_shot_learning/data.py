"""Few-shot learning data skeleton — episodic sampler."""
import torch
from torch.utils.data import DataLoader, Dataset


class EpisodicDataset(Dataset):
    """N-way K-shot episodic sampler."""
    def __init__(self, images, labels, n_way, k_shot, q_query, n_episodes):
        self.images = images
        self.labels = labels
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.n_episodes = n_episodes
        self.classes = labels.unique().tolist()

    def __len__(self):
        return self.n_episodes

    def __getitem__(self, idx):
        selected = torch.randperm(len(self.classes))[:self.n_way]
        support_x, support_y, query_x, query_y = [], [], [], []
        for new_label, cls_idx in enumerate(selected):
            cls = self.classes[cls_idx]
            mask = self.labels == cls
            indices = mask.nonzero(as_tuple=True)[0]
            perm = indices[torch.randperm(len(indices))[:self.k_shot + self.q_query]]
            support_x.append(self.images[perm[:self.k_shot]])
            support_y.extend([new_label] * self.k_shot)
            query_x.append(self.images[perm[self.k_shot:]])
            query_y.extend([new_label] * self.q_query)
        return (torch.cat(support_x), torch.tensor(support_y),
                torch.cat(query_x), torch.tensor(query_y))


def build_dataloaders(config: dict):
    n = config.get("n_samples", 1000)
    n_classes = config.get("num_classes", 20)
    n_way = config.get("n_way", 5)
    k_shot = config.get("k_shot", 1)
    q_query = config.get("q_query", 15)
    img_size = config.get("img_size", 28)
    in_ch = config.get("in_channels", 1)
    n_episodes = config.get("n_episodes", 100)

    images = torch.rand(n, in_ch, img_size, img_size)
    labels = torch.randint(0, n_classes, (n,))

    train_ds = EpisodicDataset(images, labels, n_way, k_shot, q_query, n_episodes)
    val_ds = EpisodicDataset(images, labels, n_way, k_shot, q_query, n_episodes // 5)
    return DataLoader(train_ds, batch_size=1), DataLoader(val_ds, batch_size=1)
