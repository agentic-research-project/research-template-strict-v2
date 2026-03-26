python -m lab.topic_analyzer \
  --topic "industrial image denoising" \
  --details "single-scan noisy image restoration" \
  --problem-definition "single scan images are noisy and hurt downstream defect detection" \
  --desired-outcome "PSNR 30dB+, lightweight model" \
  --constraints "fast inference, lightweight model" \
  --target-metric "PSNR, SSIM"

python -m lab.paper_researcher \
  --topic-file experiments/industrial_image_denoising/reports/topic_analysis.json

python -m lab.hypothesis_generator \
  --topic-file experiments/industrial_image_denoising/reports/topic_analysis.json \
  --papers-file experiments/industrial_image_denoising/reports/papers.json \
  --mode collaborative

python -m lab.hypothesis_validator \
  --hypothesis-file experiments/industrial_image_denoising/reports/hypothesis.json \
  --topic-file experiments/industrial_image_denoising/reports/topic_analysis.json \
  --refine \
  --target-score 9.0

python -m lab.user_approval \
  --topic-file experiments/industrial_image_denoising/reports/topic_analysis.json \
  --hypothesis-file experiments/industrial_image_denoising/reports/hypothesis.json \
  --validation-file experiments/industrial_image_denoising/reports/validation.json \
  --papers-file experiments/industrial_image_denoising/reports/papers.json

python -m lab.code_analyzer \
  --topic-file experiments/industrial_image_denoising/reports/topic_analysis.json \
  --hypothesis-file experiments/industrial_image_denoising/reports/hypothesis.json

python -m lab.model_generator \
  --topic-file experiments/industrial_image_denoising/reports/topic_analysis.json \
  --hypothesis-file experiments/industrial_image_denoising/reports/hypothesis.json \
  --code-file experiments/industrial_image_denoising/reports/code_analysis.json \
  --version 1

python -m lab.research_loop \
  --pkg-dir experiments/industrial_image_denoising/runs/v1 \
  --topic-file experiments/industrial_image_denoising/reports/topic_analysis.json \
  --hypothesis-file experiments/industrial_image_denoising/reports/hypothesis.json \
  --code-file experiments/industrial_image_denoising/reports/code_analysis.json \
  --max-rounds 3 \
  --runner-type github