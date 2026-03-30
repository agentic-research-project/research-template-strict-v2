# 연구 계획 - LLM을 활용해서 작성해 주세요

## 하고 싶은 것
<!-- 연구 주제를 자유롭게 서술하세요. -->
6559x6559 grayscale 반도체 source 이미지에서 결함을 자동으로 검출하는 인공지능 기반 기술을 개발하고 싶다.
기존 룰기반 검사 방식을 대체하거나 보완하는 것이 목적이며, 결함이 없는 reference 이미지(grayscale) 패치(256x256, 512x512 또는 그보다 작은 크기)만을 주로 활용하여
훈련을 최소화한 상태에서 결함 위치를 찾아내는 방법을 목표로 한다.

특히 다음 조건을 만족하는 방향을 원한다.
- 정상(reference) 데이터 위주로 학습 또는 사전 구축
- 새로운 스타일이나 노이즈 수준에도 강건한 검출
- source 이미지 1장당 1-2분 이내 검사
- 대규모 결함 라벨링 없이 적용 가능
- 결함 유무뿐 아니라 결함 위치까지 제시 가능

## 문제 상황
<!-- 현재 어떤 문제가 있는지, 왜 어려운지 설명하세요. -->
- source 이미지는 6559x6559의 초고해상도 grayscale 이미지이므로 한 번에 처리하기 어렵고, patch 단위 분할 및 병합 전략이 필요하다.
- 실제 결함 데이터는 수가 적고 종류가 다양하며, 결함 라벨링 비용이 높다.
- reference로 사용할 수 있는 이미지는 대부분 결함이 없는 정상 패치이며, supervised detection 학습에 필요한 충분한 결함 샘플을 확보하기 어렵다.
- 이미지 스타일, 밝기, 공정 노이즈, 장비 차이 등에 따라 정상 패턴도 다르게 보일 수 있어 단순 rule-based 방식은 오검이 많아질 수 있다.
- 매우 작은 결함과 미세한 텍스처 차이를 잡아내야 하므로 pixel-level 비교만으로는 한계가 있다.

## 원하는 결과
<!-- 무엇을 달성하면 성공인지 서술하세요. -->
- 정상 reference 패치만으로도 결함 후보를 안정적으로 검출할 수 있을 것
- source 이미지에서 anomaly map 또는 defect score map을 생성할 수 있을 것
- 최종적으로 결함 위치를 bounding box 또는 mask 형태로 제시할 수 있을 것
- 스타일 변화와 노이즈 변화에도 성능 저하가 크지 않을 것
- source 이미지 1장당 전체 검사 시간이 1-2분 이내일 것
- 새로운 lot / 장비 / 스타일이 들어와도 재학습 없이 또는 소량의 정상 reference 추가만으로 대응 가능할 것


## 제약 조건
<!-- 모델 크기, 추론 속도, 데이터 제한 등 (없으면 비워두세요) -->
- 입력 이미지는 6559x6559 grayscale source 이미지
- reference 이미지는 grayscale, 결함이 없는 정상 패치 위주 (256x256, 512x512 또는 더 작은 크기 포함)
- 입력, 출력 이미지는 h,w 크기가 다를수 있음
- defect annotation은 매우 제한적이거나 없을 수 있음
- 학습은 최소화해야 하며, 가능하면 normal-only / few-shot / training-light 방식 선호
- source 이미지 1장당 검사 시간은 1-2분 이내여야 함
- 이미지 스타일, contrast, 노이즈 수준 변화에 대해 강건해야 함
- 실환경 적용을 위해 false positive를 과도하게 증가시키는 방식은 피해야 함

## 권장 접근 방향
다음과 같은 접근을 우선 검토하고 싶다.
- normal-only anomaly detection
- reference-based feature matching
- memory bank 기반 patch anomaly localization
- self-supervised 또는 pretrained feature extractor의 frozen / light-tuning 활용
- multi-scale patch inference 및 anomaly score aggregation
- domain shift / style variation에 강건한 feature-space 비교 방식


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
실제 결함 라벨이 없으므로, supervised detection 지표(Recall, Precision, F1, AUROC, mAP 등)는 현재 단계의 핵심 평가 지표로 사용하지 않는다.
대신 정상 데이터 기반의 오검 억제, 강건성, 운영 성능을 중심으로 평가한다.

- Score Stability under Style/Noise Perturbation: 동일한 정상 이미지에 대해 brightness, contrast, blur, noise 등의 경미한 변형을 가한 뒤 anomaly score의 변화량을 측정한다. 원본과 변형본 사이의 image-level anomaly score 차이의 평균(Mean Absolute Score Shift)을 계산하고, 동일 threshold 기준에서 변형된 정상 샘플이 계속 정상으로 유지되는 비율(Normal Acceptance Rate under Perturbation)을 함께 평가한다. 해당 값들이 작거나 높을수록 스타일 및 노이즈 변화에 강건한 것으로 판단한다.

- False Positive Rate on Normal Images
- False Alarm per Image
- Anomalous Area Ratio on Normal Images
- Score Stability under Style/Noise Perturbation
- Threshold Robustness
- Inference Time per Source Image
- Peak Memory Usage
- Throughput

## 데이터 경로
<!-- 실제 훈련에 사용할 데이터 경로 알려주세요. -->
sorce : /data/0_Data/0_1_INDUST/ATI/source
reference : /data/0_Data/0_1_INDUST/ATI/reference_normal

## 참조 이미지 (선택)
<!-- 참고할 이미지 파일 경로를 적어주세요. 없으면 비워두세요.
- path/to/input_example.png   # 입력 예시
- path/to/target_example.png  # 목표 예시
-->


