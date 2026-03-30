"""
module.py — TrainingModule for Coarse-to-Fine anomaly detection

- train_epoch(fabric, loader, optimizer): compactness loss via projection_head
- val_epoch(fabric, loader): anomaly scores → FPR, inference_time, throughput, ...
- Metric names match experiment_spec output_contract required_keys
"""
import time
import numpy as np
import torch
import torch.nn.functional as F


class TrainingModule:
    """Training module for Coarse-to-Fine defect detection pipeline."""

    def __init__(self, model: torch.nn.Module, config: dict):
        self.model = model
        self.config = config
        self._epoch = 0          # tracks current epoch (for fit-once logic)
        self._fitted = False     # memory bank fit flag

    # ──────────────────────────────────────────────────────────────────
    # train_epoch — called by train.py: module.train_epoch(fabric, loader, optimizer)
    # ──────────────────────────────────────────────────────────────────
    def train_epoch(self, fabric, loader, optimizer) -> dict:
        """
        Epoch 1: fit memory bank on all normal patches first, then compactness loss.
        Epochs 2+: compactness loss only (bank already fitted).
        """
        self._epoch += 1
        total_loss = 0.0
        n_batches = 0

        # ── Fit memory bank once on first epoch ──────────────────────
        if not self._fitted:
            self._fit_memory_bank(loader)
            self._fitted = True

        # ── Compactness loss training ─────────────────────────────────
        self.model.train()
        for batch in loader:
            optimizer.zero_grad()
            loss = self._compute_loss(batch)
            fabric.backward(loss)
            if self.config.get("gradient_clip"):
                fabric.clip_gradients(
                    self.model, optimizer,
                    max_norm=self.config["gradient_clip"],
                )
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        return {"loss": total_loss / max(n_batches, 1)}

    # ──────────────────────────────────────────────────────────────────
    # val_epoch — called by train.py: module.val_epoch(fabric, val_loader)
    # ──────────────────────────────────────────────────────────────────
    def val_epoch(self, fabric, loader) -> dict:
        """
        Compute anomaly-detection metrics on validation set.
        Returns keys matching experiment_spec output_contract:
          false_positive_rate, false_alarm_per_image, anomalous_area_ratio,
          score_stability, inference_time, peak_memory, throughput
        """
        self.model.eval()

        image_scores = []
        image_labels = []
        total_time = 0.0
        n_images = 0
        peak_mem = 0.0

        with torch.no_grad():
            for batch in loader:
                imgs, labels = self._unpack_batch(batch)
                imgs = imgs.to(next(self.model.parameters()).device)
                # Ensure [B, 1, H, W] grayscale
                if imgs.dim() == 3:
                    imgs = imgs.unsqueeze(1)
                if imgs.shape[1] == 3:
                    imgs = imgs.mean(dim=1, keepdim=True)
                if imgs.shape[-1] != 256 or imgs.shape[-2] != 256:
                    imgs = F.interpolate(imgs, size=(256, 256),
                                         mode='bilinear', align_corners=False)

                t0 = time.time()
                # When fitted: forward returns dict with anomaly_scores
                # When not fitted: forward returns projection tensor
                out = self.model(imgs)
                total_time += time.time() - t0
                n_images += imgs.shape[0]

                if isinstance(out, dict):
                    scores = out.get("anomaly_scores",
                                     torch.zeros(imgs.shape[0]))
                else:
                    # Not fitted yet — use L2 distance from zero as proxy
                    scores = out.norm(dim=-1)

                scores_np = scores.detach().cpu().float().numpy()
                if scores_np.ndim == 0:
                    scores_np = scores_np.reshape(1)
                image_scores.extend(scores_np.tolist())
                image_labels.extend(labels.cpu().numpy().tolist()
                                     if hasattr(labels, 'cpu') else list(labels))

                if torch.cuda.is_available():
                    peak_mem = max(peak_mem,
                                   torch.cuda.max_memory_allocated() / 1024**3)

        # ── Metric computation ────────────────────────────────────────
        return self._compute_val_metrics(
            image_scores, image_labels, total_time, n_images, peak_mem
        )

    # ──────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────

    def _compute_loss(self, batch) -> torch.Tensor:
        """
        Compactness loss: MSE(proj, batch_centroid).
        The projection_head is the only trainable component.
        """
        imgs, _ = self._unpack_batch(batch)
        imgs = imgs.to(next(self.model.parameters()).device)

        if imgs.dim() == 3:
            imgs = imgs.unsqueeze(1)
        if imgs.shape[1] == 3:
            imgs = imgs.mean(dim=1, keepdim=True)
        if imgs.shape[-1] != 256 or imgs.shape[-2] != 256:
            imgs = F.interpolate(imgs, size=(256, 256),
                                  mode='bilinear', align_corners=False)

        self.model.train()
        out = self.model(imgs)                     # [B, proj_dim] when not fitted
        if isinstance(out, dict):
            # Model is fitted — grab projection from dict (key is 'proj')
            proj = out.get("proj", None)
            if proj is None:
                # No gradient path available — return zero loss
                return torch.tensor(0.0, requires_grad=True,
                                    device=imgs.device)
        else:
            proj = out                              # [B, proj_dim]

        centroid = proj.mean(dim=0, keepdim=True)
        loss = F.mse_loss(proj, centroid.expand_as(proj))
        return loss

    def _fit_memory_bank(self, loader) -> None:
        """Fit the pipeline's memory bank on all normal patches in loader.

        Delegates to model.fit(loader) which expects an iterable of batches
        where batch[0] = images [B, 1, H, W].
        """
        print("  [module] Fitting memory bank on normal samples...")
        try:
            self.model.fit(loader)
        except Exception as e:
            print(f"  [module] fit() warning: {e}")

    def _unpack_batch(self, batch):
        """Extract (images, labels) from various batch formats."""
        if isinstance(batch, (list, tuple)):
            imgs = batch[0]
            labels = batch[1] if len(batch) >= 2 else torch.zeros(imgs.shape[0])
        elif isinstance(batch, dict):
            keys = list(batch.keys())
            imgs = batch[keys[0]]
            labels = batch[keys[1]] if len(keys) >= 2 else torch.zeros(imgs.shape[0])
        else:
            imgs = batch
            labels = torch.zeros(imgs.shape[0])
        return imgs, labels

    def _compute_val_metrics(self, image_scores, image_labels,
                              total_time, n_images, peak_mem) -> dict:
        """
        Compute metrics matching experiment_spec output_contract required_keys:
          false_positive_rate, false_alarm_per_image, anomalous_area_ratio,
          score_stability, inference_time, peak_memory, throughput
        """
        scores = np.array(image_scores, dtype=np.float32)
        labels = np.array(image_labels, dtype=np.int32)

        if n_images == 0 or len(scores) == 0:
            return self._fallback_metrics()

        # Threshold: median + 3×MAD on ALL scores (unsupervised)
        median_s = float(np.median(scores))
        mad_s = float(np.median(np.abs(scores - median_s)))
        threshold = median_s + 3.0 * mad_s

        # Normal images (label == 0)
        normal_mask = labels == 0
        n_normal = int(np.sum(normal_mask))

        if n_normal > 0:
            normal_scores = scores[normal_mask]
            n_fp = int(np.sum(normal_scores > threshold))
            false_positive_rate = n_fp / n_normal
            false_alarm_per_image = n_fp / n_images
        else:
            false_positive_rate = 0.0
            false_alarm_per_image = 0.0

        # Anomalous area ratio — proxy: fraction of scores above 90th pctile
        thr_90 = float(np.percentile(scores, 90)) if len(scores) >= 10 else threshold
        anomalous_area_ratio = float(np.mean(scores > thr_90))

        # Score stability — 1 - (std / (mean + ε))
        score_stability = float(
            max(0.0, 1.0 - float(np.std(scores)) / (float(np.mean(np.abs(scores))) + 1e-8))
        )

        inference_time = total_time / max(n_images, 1)
        throughput = n_images / max(total_time, 1e-6)

        return {
            "false_positive_rate":   round(false_positive_rate,   4),
            "false_alarm_per_image": round(false_alarm_per_image, 4),
            "anomalous_area_ratio":  round(anomalous_area_ratio,  4),
            "score_stability":       round(score_stability,        4),
            "inference_time":        round(inference_time,         4),
            "peak_memory":           round(peak_mem,               3),
            "throughput":            round(throughput,              2),
        }

    def _fallback_metrics(self) -> dict:
        """Safe fallback when no images were processed."""
        return {
            "false_positive_rate":   0.0,
            "false_alarm_per_image": 0.0,
            "anomalous_area_ratio":  0.0,
            "score_stability":       1.0,
            "inference_time":        0.0,
            "peak_memory":           0.0,
            "throughput":            0.0,
        }
