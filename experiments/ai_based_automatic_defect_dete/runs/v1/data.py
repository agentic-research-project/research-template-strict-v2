"""
data.py — ATI 반도체 이상탐지 Dataset / DataLoader

구조:
  source: /data/0_Data/0_1_INDUST/ATI/source/{category}/{lot_id}/Full.bmp
  reference: /data/0_Data/0_1_INDUST/ATI/reference_normal/*.jpg

build_dataloaders(config) → (train_loader, val_loader) 반환
  - train_loader: reference_normal 패치 (FAISS 인덱스 구축용)
  - val_loader: source 이미지 패치 (검증용 — 배치크기 작게)
"""
from pathlib import Path
from typing import Tuple, Optional, List
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as transforms


# ── ATI 참조 패치 Dataset (train) ─────────────────────────────────────────
class ATIReferenceDataset(Dataset):
    """정상 참조 패치를 로드하는 학습용 Dataset."""

    def __init__(self, reference_dir: str, transform=None,
                 target_size: Tuple[int, int] = (256, 256)):
        self.reference_dir = Path(reference_dir)
        self.transform = transform
        self.target_size = target_size

        # jpg/png/bmp 확장자 수집
        exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
        self.files: List[Path] = []
        for ext in exts:
            self.files.extend(sorted(self.reference_dir.glob(ext)))

        self._dummy_mode = len(self.files) == 0
        if self._dummy_mode:
            print(f"[data] WARNING: reference_dir={reference_dir} 파일 없음 → 더미 모드")

    def __len__(self) -> int:
        return 32 if self._dummy_mode else len(self.files)

    def __getitem__(self, idx: int):
        if self._dummy_mode:
            patch = torch.zeros(1, self.target_size[0], self.target_size[1])
            return patch, 0

        img_path = self.files[idx % len(self.files)]
        img = Image.open(img_path).convert("L")
        if self.target_size:
            img = img.resize(self.target_size, Image.BILINEAR)
        img_t = transforms.ToTensor()(img)  # [1, H, W], float [0,1]
        if self.transform:
            img_t = self.transform(img_t)
        return img_t, 0  # label=0 (정상)


# ── ATI Source 이미지 Dataset (validation) ────────────────────────────────
class ATISourceDataset(Dataset):
    """
    source 디렉토리의 Full.bmp를 패치로 분할하여 반환하는 검증용 Dataset.
    메모리 절약을 위해 각 이미지를 patch_size 크기로 분할.
    """

    def __init__(self, source_dir: str, transform=None,
                 patch_size: int = 256, overlap: int = 32,
                 max_source_images: int = 3,
                 patches_per_image: int = 50):
        self.source_dir = Path(source_dir)
        self.transform = transform
        self.patch_size = patch_size
        self.overlap = overlap
        self.patches_per_image = patches_per_image

        # Full.bmp 파일 수집 (최대 max_source_images)
        self.source_files: List[Path] = []
        for bmp_file in self.source_dir.rglob("Full.bmp"):
            self.source_files.append(bmp_file)
            if len(self.source_files) >= max_source_images:
                break

        self._dummy_mode = len(self.source_files) == 0
        if self._dummy_mode:
            print(f"[data] WARNING: source_dir={source_dir} Full.bmp 없음 → 더미 모드")
            self._dummy_patches = 16

        # 각 이미지에서 (img_idx, patch_y, patch_x) 튜플 생성
        if not self._dummy_mode:
            self._build_patch_index()

    def _build_patch_index(self):
        """패치 인덱스 사전 구축."""
        self.patch_index = []
        step = self.patch_size - self.overlap
        for img_idx, img_path in enumerate(self.source_files):
            try:
                img = Image.open(img_path)
                W, H = img.size
                ys = list(range(0, H - self.patch_size + 1, step))
                xs = list(range(0, W - self.patch_size + 1, step))
                positions = [(y, x) for y in ys for x in xs]
                # 균등 샘플링
                if len(positions) > self.patches_per_image:
                    indices = np.linspace(0, len(positions) - 1,
                                          self.patches_per_image, dtype=int)
                    positions = [positions[i] for i in indices]
                for (y, x) in positions:
                    self.patch_index.append((img_idx, y, x))
            except Exception as e:
                print(f"[data] 이미지 읽기 오류 {img_path}: {e}")

    def __len__(self) -> int:
        if self._dummy_mode:
            return self._dummy_patches
        return len(self.patch_index)

    def __getitem__(self, idx: int):
        if self._dummy_mode:
            patch = torch.zeros(1, self.patch_size, self.patch_size)
            return patch, 0

        img_idx, y, x = self.patch_index[idx]
        img_path = self.source_files[img_idx]
        try:
            img = Image.open(img_path).convert("L")
            patch_img = img.crop((x, y, x + self.patch_size, y + self.patch_size))
            patch_t = transforms.ToTensor()(patch_img)  # [1, H, W]
            if self.transform:
                patch_t = self.transform(patch_t)
            return patch_t, 0
        except Exception as e:
            print(f"[data] 패치 추출 오류 {img_path} ({y},{x}): {e}")
            return torch.zeros(1, self.patch_size, self.patch_size), 0


# ── 합성 데이터 Dataset (fallback) ────────────────────────────────────────
class SyntheticPatchDataset(Dataset):
    """연결망 없을 때 사용하는 소형 합성 패치 Dataset."""

    def __init__(self, num_samples: int = 64, patch_size: int = 256,
                 anomaly_ratio: float = 0.1, seed: int = 42):
        rng = np.random.RandomState(seed)
        self.patch_size = patch_size
        n_anom = int(num_samples * anomaly_ratio)
        self.labels = [1] * n_anom + [0] * (num_samples - n_anom)
        rng.shuffle(self.labels)
        self.num_samples = num_samples
        self.rng = rng

    def __len__(self) -> int:
        return self.num_samples

    def _make_patch(self, label: int):
        p = self.patch_size
        rng = self.rng
        # 주기성 텍스처 패턴
        pattern = np.zeros((p, p), dtype=np.float32)
        x = np.arange(p)
        y = np.arange(p)
        X, Y = np.meshgrid(x, y)
        pattern += 0.3 * np.sin(X / 32) * np.cos(Y / 32)
        pattern += rng.normal(0, 0.05, (p, p)).astype(np.float32)
        if label == 1:
            # 결함 삽입
            cx, cy = rng.randint(50, p - 50, size=2)
            r = rng.randint(5, 20)
            mask = (X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2
            pattern[mask] = 0.9
        pattern = np.clip(pattern, 0, 1)
        return torch.from_numpy(pattern).unsqueeze(0)

    def __getitem__(self, idx: int):
        label = self.labels[idx]
        patch = self._make_patch(label)
        return patch, label


# ── build_dataloaders ─────────────────────────────────────────────────────
def build_dataloaders(config: dict):
    """
    학습/검증 DataLoader 생성.
    config keys:
        source_dir:      source 이미지 루트
        reference_dir:   정상 참조 패치 루트
        data_dir:        fallback data root
        batch_size:      학습 배치 크기
        val_batch_size:  검증 배치 크기 (기본 4)
        patch_size:      패치 크기 (기본 256)
        patch_overlap:   패치 겹침 픽셀 (기본 32)
        num_workers:     DataLoader workers
        seed:            재현성 seed
    """
    patch_size = config.get("patch_size", 256)
    overlap    = config.get("patch_overlap", 32)
    batch_size = config.get("batch_size", 16)
    val_bs     = config.get("val_batch_size", 4)
    num_workers = config.get("num_workers", 2)
    seed       = config.get("seed", 42)

    # 정규화 transform
    normalize_transform = transforms.Normalize(mean=[0.5], std=[0.5])

    # ── reference_dir 처리 ──────────────────────────────────────────
    reference_dir = config.get("reference_dir", "")
    if reference_dir and Path(reference_dir).exists():
        train_ds = ATIReferenceDataset(
            reference_dir=reference_dir,
            transform=normalize_transform,
            target_size=(patch_size, patch_size),
        )
        print(f"[data] ATI reference dataset: {len(train_ds)} patches from {reference_dir}")
    else:
        print(f"[data] reference_dir 없음 → 합성 데이터 사용")
        train_ds = SyntheticPatchDataset(
            num_samples=config.get("num_synthetic_samples", 64),
            patch_size=patch_size,
            seed=seed,
        )

    # ── source_dir 처리 ─────────────────────────────────────────────
    source_dir = config.get("source_dir", "")
    if source_dir and Path(source_dir).exists():
        val_ds = ATISourceDataset(
            source_dir=source_dir,
            transform=normalize_transform,
            patch_size=patch_size,
            overlap=overlap,
            max_source_images=config.get("max_val_source_images", 3),
            patches_per_image=config.get("val_patches_per_image", 50),
        )
        print(f"[data] ATI source dataset: {len(val_ds)} patches from {source_dir}")
    else:
        print(f"[data] source_dir 없음 → 합성 데이터 사용")
        val_ds = SyntheticPatchDataset(
            num_samples=config.get("num_synthetic_val_samples", 32),
            patch_size=patch_size,
            anomaly_ratio=0.2,
            seed=seed + 1,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=lambda wid: torch.manual_seed(seed + wid),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=val_bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader
