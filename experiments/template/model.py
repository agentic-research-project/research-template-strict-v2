"""
model.py — 모델 아키텍처 (템플릿 / GPT patch 허용)

- build_model(config) 함수가 반드시 존재해야 한다
- hypothesis.json의 model_architecture를 구현한다
- 하드코딩 금지: 모든 크기/레이어 수는 config에서 읽는다

이 파일은 Claude가 가설에 맞는 실제 아키텍처로 교체한다.
아래 코드는 config 기반 범용 인코더-디코더 구조의 scaffolding이다.
"""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """기본 Conv-BN-ReLU 블록."""
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=kernel // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Model(nn.Module):
    """
    템플릿 모델 — 실험 가설에 따라 Claude가 교체한다.

    config keys:
        in_channels  : 입력 채널 수 (기본 1)
        base_channels: 첫 번째 레이어 채널 수 (기본 32)
        depth        : encoder 단계 수 (기본 3)
        use_aux_head : 보조 출력 헤드 활성화 (기본 False)
    """
    def __init__(self, config: dict):
        super().__init__()
        in_ch   = config.get("in_channels", 1)
        base_ch = config.get("base_channels", 32)
        depth   = config.get("depth", 3)

        # Encoder
        self.encoder = nn.ModuleList()
        ch = in_ch
        for i in range(depth):
            out_ch = base_ch * (2 ** i)
            self.encoder.append(ConvBlock(ch, out_ch))
            ch = out_ch

        # Bottleneck
        self.bottleneck = ConvBlock(ch, ch * 2)
        ch = ch * 2

        # Decoder (symmetric)
        self.decoder = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            out_ch = base_ch * (2 ** i)
            self.decoder.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    ConvBlock(ch, out_ch),
                )
            )
            ch = out_ch

        # Output head (가설의 output_shape에 맞게 교체)
        self.out_conv = nn.Conv2d(ch, in_ch, 1)

        # 보조 출력 헤드 (가설에서 use_aux_head=True로 활성화)
        self.use_aux = config.get("use_aux_head", False)
        if self.use_aux:
            aux_out_ch = config.get("aux_out_channels", 1)
            self.aux_head = nn.Conv2d(ch, aux_out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc_features = []
        h = x
        for enc in self.encoder:
            h = enc(h)
            enc_features.append(h)
            h = nn.functional.avg_pool2d(h, 2)

        h = self.bottleneck(h)

        for i, dec in enumerate(self.decoder):
            h = dec(h)
            skip = enc_features[-(i + 1)]
            if h.shape == skip.shape:
                h = h + skip

        return self.out_conv(h)

    def aux_forward(self, x: torch.Tensor):
        """보조 헤드 출력 (use_aux_head=True일 때만 호출)."""
        raise NotImplementedError("aux_forward는 가설별로 구현해야 합니다")


def build_model(config: dict) -> Model:
    return Model(config)
