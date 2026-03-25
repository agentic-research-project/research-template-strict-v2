"""
data.py — Dataset / DataLoader (템플릿)

build_dataloaders(config) → (train_loader, val_loader) 를 반환해야 한다.

Claude가 가설의 dataset / task_type에 맞게 이 파일 전체를 재작성한다:
  - 복원/생성: (input_tensor, target_tensor) 반환
  - 분류:      (image_tensor, label_int) 반환
  - 분할/탐지: (image_tensor, mask_or_bbox) 반환
"""
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, random_split


class TaskDataset(Dataset):
    """
    범용 Dataset 템플릿.

    config keys (필수):
        data_dir : 데이터 루트 경로
    config keys (선택):
        val_ratio: 검증 분할 비율 (기본 0.1)
    """
    def __init__(self, data_dir: str, split: str = "train", config: dict = None):
        config = config or {}
        self.data_dir = Path(data_dir)
        self.split    = split
        self.config   = config

        # 파일 목록 수집 — 태스크에 맞는 확장자/구조로 교체
        self.files = sorted(f for f in self.data_dir.rglob("*") if f.is_file())

        if not self.files:
            # 데이터 없을 때 더미 모드 (smoke test 통과용)
            self._dummy_mode = True
            self._dummy_size = 16 if split == "train" else 4
        else:
            self._dummy_mode = False

    def __len__(self) -> int:
        return self._dummy_size if self._dummy_mode else len(self.files)

    def __getitem__(self, idx: int):
        """
        (inputs, targets) 형태로 반환한다.
        Claude가 태스크에 맞게 이 메서드를 구현한다.
        """
        if self._dummy_mode:
            # 더미 데이터 — 실제 입출력 shape으로 교체할 것
            inputs  = torch.zeros(1, 64, 64)
            targets = torch.zeros(1, 64, 64)
            return inputs, targets

        raise NotImplementedError(
            "__getitem__()을 태스크에 맞게 구현해야 합니다.\n"
            "예: 이미지 로드, 전처리, augmentation, label 매핑 등"
        )


def build_dataloaders(config: dict):
    data_dir    = config.get("data_dir", "data/")
    batch_size  = config["batch_size"]
    num_workers = config.get("num_workers", 2)
    val_ratio   = config.get("val_ratio", 0.1)

    full_dataset = TaskDataset(data_dir, split="train", config=config)
    n_val   = max(1, int(len(full_dataset) * val_ratio))
    n_train = len(full_dataset) - n_val

    if n_train < batch_size * 10:
        print(f"[data] WARNING: 학습 샘플({n_train}) < batch_size*10 → drop_last=False")
        drop_last = False
    else:
        drop_last = True

    train_ds, val_ds = random_split(full_dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        worker_init_fn=lambda wid: torch.manual_seed(config["seed"] + wid),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
