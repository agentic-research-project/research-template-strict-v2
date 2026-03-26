# 연구 계획 - LLM을 활용해서 작성해 주세요

## 하고 싶은 것
<!-- 연구 주제를 자유롭게 서술하세요. -->
FashionMNIST 28x28 grayscale 이미지 분류를 위한 경량 CNN 모델을 설계하고 싶다.
작은 파라미터 수와 빠른 추론 속도를 유지하면서도 높은 분류 정확도를 달성하는 것이 목표다.

## 문제 상황
<!-- 현재 어떤 문제가 있는지, 왜 어려운지 설명하세요. -->
- 작은 모델은 표현력이 부족해서 정확도가 쉽게 떨어진다.
- FashionMNIST는 MNIST보다 클래스 간 형태가 비슷해 단순한 구조로는 성능 한계가 있다.
- GitHub 기반 자동 실험 루프에서 빠르게 반복 가능한 문제여야 하므로 학습 시간이 짧아야 한다.

## 원하는 결과
<!-- 무엇을 달성하면 성공인지 서술하세요. -->
Validation Accuracy 92.5% 이상을 달성하는 경량 분류 모델을 얻고 싶다.
모델은 빠르게 학습되고 추론 가능해야 하며, 2~3회의 실험 수정(Path A)을 거쳐 성능이 점진적으로 개선되는 것이 바람직하다.

## 제약 조건
<!-- 모델 크기, 추론 속도, 데이터 제한 등 (없으면 비워두세요) -->
- parameter budget 0.2M 이하
- single GPU 환경에서 실행 가능해야 함
- 학습 시간은 짧아야 함
- 추론 속도는 가벼운 CNN 수준으로 유지
- 과도하게 복잡한 backbone이나 대형 pretrained model 사용 금지

## 측정 지표
<!-- 성능을 평가할 지표를 적어주세요. 비워두면 도메인에서 자동 추론됩니다.
     예시:
       denoising/restoration → PSNR, SSIM, LPIPS
       segmentation          → mIoU, Dice Score
       detection             → mAP, AP50
       classification        → Accuracy, F1, AUC-ROC
       generation/diffusion  → FID, IS, LPIPS
       depth estimation      → AbsRel, RMSE
       medical imaging       → AUC-ROC, Dice Score
       NLP/captioning        → BLEU, ROUGE, BERTScore
-->
Accuracy

## 데이터 경로
<!-- 실제 훈련에 사용할 데이터 경로 알려주세요. -->
/data/0_Data/5_OpenSource/1_fashion_mnist

## 참조 이미지 (선택)
<!-- 참고할 이미지 파일 경로를 적어주세요. 없으면 비워두세요.
- path/to/input_example.png   # 입력 예시
- path/to/target_example.png  # 목표 예시
-->


