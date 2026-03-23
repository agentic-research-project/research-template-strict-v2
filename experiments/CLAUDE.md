# Experiments — 생성된 모델 코드

## 역할
`lab/model_generator.py`가 생성한 PyTorch 모델 코드를 저장한다.
`lab/research_loop.py`가 이 코드를 동적으로 임포트하여 실험을 실행한다.

## 파일 명명 규칙

```
{topic_slug}_v{version}.py
예: deep_learning_denoising_v1.py
    deep_learning_denoising_v2.py   ← research_loop이 목표 미달 시 재생성
```

## 모델 파일 구조 (생성 규칙)

각 생성 파일은 다음 인터페이스를 반드시 포함해야 한다:

```python
# 1. 모델 클래스
class Model(nn.Module):
    def __init__(self, config: dict): ...
    def forward(self, x): ...

# 2. 기본 설정
DEFAULT_CONFIG = {
    "lr": 1e-4,
    "batch_size": 16,
    "epochs": 50,
    ...
}

# 3. 학습 함수
def train(config: dict) -> dict:
    """metrics 딕셔너리 반환: {"psnr": float, "ssim": float, ...}"""
    ...
```

## 규칙
- 파일은 model_generator.py가 자동 생성 — 직접 수정하지 말 것
- 재실험이 필요하면 research_loop이 model_generator를 재호출하여 새 버전 생성
- 각 버전은 삭제하지 않고 보존 (실험 재현성)
