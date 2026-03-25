"""
module.py — TrainingModule (step 로직)

- 손실 함수 / 평가 지표 계산 담당
- Fabric 인스턴스는 인자로 받음 (직접 생성 금지)
- 실험 가설에 맞게 Claude가 교체하는 핵심 파일
- 하드코딩 금지: 손실 함수·지표·배치 구조는 config 또는 hypothesis에서 결정
"""
import torch
import torch.nn.functional as F


# ── 손실 함수 레지스트리 ──────────────────────────────────────────

def _build_loss_fn(config: dict):
    """config["loss_function"]에 따라 손실 함수를 반환한다."""
    name = config.get("loss_function", "l1").lower()
    if name in ("l1", "mae"):
        return F.l1_loss
    if name in ("l2", "mse"):
        return F.mse_loss
    if name in ("ce", "cross_entropy"):
        return F.cross_entropy
    if name in ("bce", "binary_cross_entropy"):
        return F.binary_cross_entropy_with_logits
    return F.l1_loss  # 기본값


class TrainingModule:
    def __init__(self, model: torch.nn.Module, config: dict):
        self.model   = model
        self.config  = config
        self.loss_fn = _build_loss_fn(config)

    # ──────────────────────────────────────────
    # 학습 1 epoch
    # ──────────────────────────────────────────
    def train_epoch(self, fabric, loader, optimizer) -> dict:
        total_loss = 0.0
        n_batches  = 0

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
            n_batches  += 1

        return {"loss": total_loss / max(n_batches, 1)}

    # ──────────────────────────────────────────
    # 검증 1 epoch
    # ──────────────────────────────────────────
    def val_epoch(self, fabric, loader) -> dict:
        """primary_metric / secondary_metrics를 계산하여 반환한다."""
        metric_sum: dict[str, float] = {}
        n_batches = 0

        for batch in loader:
            inputs, targets = self._unpack(batch)
            preds = self.model(inputs)
            for k, v in self._compute_metrics(preds, targets).items():
                metric_sum[k] = metric_sum.get(k, 0.0) + float(v)
            n_batches += 1

        n = max(n_batches, 1)
        return {k: round(v / n, 4) for k, v in metric_sum.items()}

    # ──────────────────────────────────────────
    # 내부 헬퍼
    # ──────────────────────────────────────────
    def _compute_loss(self, batch) -> torch.Tensor:
        inputs, targets = self._unpack(batch)
        preds = self.model(inputs)
        loss = self.loss_fn(preds, targets)
        aux_weight = self.config.get("aux_loss_weight", 0.0)
        if aux_weight > 0 and hasattr(self.model, "aux_forward"):
            aux_out, aux_target = self.model.aux_forward(inputs)
            loss = loss + aux_weight * F.binary_cross_entropy_with_logits(
                aux_out, aux_target
            )
        return loss

    def _unpack(self, batch) -> tuple:
        """배치에서 (inputs, targets) 쌍을 추출한다.
        data.py의 Dataset.__getitem__ 반환 구조에 맞게 수정한다.
        """
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            return batch[0], batch[1]
        if isinstance(batch, dict):
            keys = list(batch.keys())
            return batch[keys[0]], batch[keys[1]]
        raise ValueError(f"지원하지 않는 배치 형식: {type(batch)}")

    def _compute_metrics(self, preds: torch.Tensor, targets: torch.Tensor) -> dict:
        """
        가설에서 정의한 primary_metric / secondary_metrics를 계산한다.
        Claude가 실험별로 이 메서드를 구현한다.

        반환 형식: {metric_name: value, ...}
        experiment_spec의 evaluation_config.primary_metric과 일치해야 함.

        구현 예시:
          psnr       → 10 * log10(1 / (mse + 1e-8))
          accuracy   → (preds.argmax(1) == targets).float().mean()
          iou / dice → 픽셀별 교집합/합집합 계산
        """
        raise NotImplementedError(
            "_compute_metrics()를 가설의 evaluation_metrics에 맞게 구현하세요."
        )
