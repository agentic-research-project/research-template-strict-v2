"""
Task-Family Engine — 범용 연구 코드 생성을 위한 task-family abstraction

각 task family는 다음을 제공한다:
  - pattern library (architectural priors)
  - baseline synthesizer (기본 baseline set)
  - starter skeleton selector
  - family-specific experiment_spec contracts
  - generation prior (risk + required interfaces)
  - literature-to-code bridge (hints)
  - contract tests (family-specific validation)

지원 family:
  classification, restoration_denoising, super_resolution, segmentation,
  detection, representation_learning, few_shot_learning,
  tabular_prediction, timeseries_prediction
"""

# ──────────────────────────────────────────────────────────
# Task Family Enum
# ──────────────────────────────────────────────────────────

TASK_FAMILIES = [
    "classification",
    "restoration_denoising",
    "super_resolution",
    "segmentation",
    "detection",
    "representation_learning",
    "few_shot_learning",
    "tabular_prediction",
    "timeseries_prediction",
    "generation",
    "meta_learning",
    "contrastive_learning",
    "anomaly_detection",
    "similarity_measure",
    "image_manipulation",
    "zero_shot_learning",
    "one_shot_learning",
    "physics_informed",
]

# ──────────────────────────────────────────────────────────
# Latest Trends Registry (2024-2025)
#
# 코드 생성/수정 시 최신 트렌드를 반영하기 위한 참조 정보.
# model_generator prompt에 주입되어 Claude가 최신 기법을 인지하도록 한다.
# ──────────────────────────────────────────────────────────

LATEST_TRENDS = {
    "architectures": [
        "State Space Models — Mamba/Mamba2 (linear-time), MambaVision (SSM+ViT hybrid), Vim (Vision Mamba)",
        "DINOv2/DINOv3 — self-supervised ViT, strongest general-purpose vision features",
        "SigLIP 2 — improved vision-language model, better than CLIP for zero-shot",
        "ConvNeXt V2 — modernized ConvNet with FCMAE self-supervised pretraining",
        "SAM 2 — Segment Anything Model 2, improved spatial understanding",
        "Mixture of Experts (MoE) — conditional computation for scaling",
        "KAN (Kolmogorov-Arnold Networks) — learnable activation functions",
    ],
    "training_techniques": [
        "LoRA / QLoRA — parameter-efficient fine-tuning",
        "Lion / Sophia optimizer — modern alternatives to AdamW",
        "Cosine annealing with warm restarts (SGDR)",
        "Gradient accumulation for effective large batch",
        "Mixed precision training (bf16/fp16) with loss scaling",
        "EMA (Exponential Moving Average) model for stable evaluation",
    ],
    "augmentation": [
        "TrivialAugment — zero-hyperparameter augmentation",
        "RandAugment — simplified AutoAugment",
        "CutMix / MixUp — interpolation-based regularization",
        "AugMax — adversarial augmentation for robustness",
    ],
    "losses": [
        "Label smoothing cross-entropy",
        "Focal loss — class imbalance handling",
        "Poly loss — polynomial cross-entropy",
        "Charbonnier loss — smooth L1 for restoration",
        "LPIPS — perceptual similarity loss",
    ],
    "normalization": [
        "RMSNorm — simplified LayerNorm (transformer 표준)",
        "GroupNorm — batch-size independent",
        "LayerNorm — transformer / SSM 표준",
    ],
    "evaluation": [
        "DINO/DINOv2 features for representation quality",
        "FID / KID for generation quality",
        "Fréchet distance variants for distribution matching",
    ],
    "self_supervised": [
        "DINOv2/DINOv3 — self-supervised ViT with distillation, universal features",
        "SigLIP 2 — sigmoid loss vision-language, better calibrated than CLIP",
        "MAE (Masked Autoencoder) — mask-then-predict pretraining",
        "I-JEPA — joint embedding predictive architecture",
        "VICReg / Barlow Twins — variance-invariance-covariance",
        "MambaVision — SSM+Attention hybrid, linear complexity with global context",
    ],
}

# ──────────────────────────────────────────────────────────
# A. Pattern Library
# ──────────────────────────────────────────────────────────

PATTERN_LIBRARY: dict[str, list[dict]] = {
    "classification": [
        {"pattern_id": "cls_cnn_lite", "core_blocks": ["conv_block", "fc_head"],
         "training_defaults": {"optimizer": "adam", "loss_function": "cross_entropy"},
         "evaluation_defaults": {"primary_metric": "accuracy"},
         "strengths": ["simple", "fast training"], "weaknesses": ["limited capacity"],
         "must_not_do": ["oversized backbone under param budget"]},
        {"pattern_id": "cls_residual_small", "core_blocks": ["residual_block", "global_avg_pool", "fc_head"],
         "training_defaults": {"optimizer": "adam", "loss_function": "cross_entropy"},
         "evaluation_defaults": {"primary_metric": "accuracy"},
         "strengths": ["stable gradient flow", "good baseline"], "weaknesses": ["may need more depth"],
         "must_not_do": ["skip connections without batch norm"]},
    ],
    "restoration_denoising": [
        {"pattern_id": "restoration_unet_lite", "core_blocks": ["encoder", "decoder", "skip_connection"],
         "training_defaults": {"optimizer": "adam", "loss_function": "l1"},
         "evaluation_defaults": {"primary_metric": "psnr"},
         "strengths": ["multi-scale features", "proven for restoration"], "weaknesses": ["param heavy if deep"],
         "must_not_do": ["heavy perceptual loss on small data as default"]},
        {"pattern_id": "restoration_dncnn_like", "core_blocks": ["conv_bn_relu_stack", "residual_learning"],
         "training_defaults": {"optimizer": "adam", "loss_function": "mse"},
         "evaluation_defaults": {"primary_metric": "psnr"},
         "strengths": ["simple residual denoising", "well-studied"], "weaknesses": ["limited receptive field"],
         "must_not_do": ["skip residual connection — must predict noise, not clean image"]},
        {"pattern_id": "restoration_residual_cnn", "core_blocks": ["residual_block", "conv_tail"],
         "training_defaults": {"optimizer": "adam", "loss_function": "l1"},
         "evaluation_defaults": {"primary_metric": "psnr"},
         "strengths": ["flexible depth", "stable"], "weaknesses": ["no multi-scale"],
         "must_not_do": ["output without clamp to valid pixel range"]},
    ],
    "super_resolution": [
        {"pattern_id": "superres_srcnn_like", "core_blocks": ["feature_extraction", "mapping", "reconstruction"],
         "training_defaults": {"optimizer": "adam", "loss_function": "mse"},
         "evaluation_defaults": {"primary_metric": "psnr"},
         "strengths": ["minimal complexity", "fast"], "weaknesses": ["limited quality"],
         "must_not_do": ["upscale factor without contract in spec"]},
        {"pattern_id": "superres_edsr_lite", "core_blocks": ["residual_block", "upsample_block"],
         "training_defaults": {"optimizer": "adam", "loss_function": "l1"},
         "evaluation_defaults": {"primary_metric": "psnr"},
         "strengths": ["stable residual learning", "good reconstruction"], "weaknesses": ["weak perceptual sharpness"],
         "must_not_do": ["oversized width under strict budget"]},
        {"pattern_id": "superres_subpixel_cnn", "core_blocks": ["conv_block", "pixel_shuffle"],
         "training_defaults": {"optimizer": "adam", "loss_function": "l1"},
         "evaluation_defaults": {"primary_metric": "psnr"},
         "strengths": ["efficient upsampling", "no deconv artifacts"], "weaknesses": ["checkerboard if untrained"],
         "must_not_do": ["pixel shuffle without correct channel factor"]},
    ],
    "segmentation": [
        {"pattern_id": "seg_unet_lite", "core_blocks": ["encoder", "decoder", "skip_connection", "seg_head"],
         "training_defaults": {"optimizer": "adam", "loss_function": "cross_entropy"},
         "evaluation_defaults": {"primary_metric": "iou"},
         "strengths": ["proven for segmentation"], "weaknesses": ["param heavy"],
         "must_not_do": ["missing skip connections in decoder"]},
        {"pattern_id": "seg_fcn_lite", "core_blocks": ["backbone", "upsample", "pixel_classifier"],
         "training_defaults": {"optimizer": "adam", "loss_function": "cross_entropy+dice"},
         "evaluation_defaults": {"primary_metric": "iou"},
         "strengths": ["simple fully convolutional"], "weaknesses": ["no multi-scale"],
         "must_not_do": ["output without spatial alignment to input"]},
    ],
    "detection": [
        {"pattern_id": "detection_fcos_lite", "core_blocks": ["backbone", "fpn", "cls_head", "reg_head"],
         "training_defaults": {"optimizer": "sgd", "loss_function": "focal+giou"},
         "evaluation_defaults": {"primary_metric": "mAP"},
         "strengths": ["anchor-free", "simple"], "weaknesses": ["needs FPN for multi-scale"],
         "must_not_do": ["NMS/eval contract missing", "bbox format mismatch"]},
        {"pattern_id": "detection_yolo_lite", "core_blocks": ["backbone", "neck", "detect_head"],
         "training_defaults": {"optimizer": "sgd", "loss_function": "bce+ciou"},
         "evaluation_defaults": {"primary_metric": "mAP"},
         "strengths": ["fast inference", "single-stage"], "weaknesses": ["anchor tuning needed"],
         "must_not_do": ["deploy without NMS postprocess"]},
    ],
    "representation_learning": [
        {"pattern_id": "repr_contrastive_encoder", "core_blocks": ["encoder", "projection_head", "contrastive_loss"],
         "training_defaults": {"optimizer": "adam", "loss_function": "ntxent"},
         "evaluation_defaults": {"primary_metric": "linear_probe_accuracy"},
         "strengths": ["self-supervised", "transferable"], "weaknesses": ["needs augmentation strategy"],
         "must_not_do": ["contrastive without projection head", "no downstream probe"]},
        {"pattern_id": "repr_supervised_encoder_baseline", "core_blocks": ["encoder", "fc_head"],
         "training_defaults": {"optimizer": "adam", "loss_function": "cross_entropy"},
         "evaluation_defaults": {"primary_metric": "accuracy"},
         "strengths": ["simple supervised baseline"], "weaknesses": ["not self-supervised"],
         "must_not_do": ["claim representation learning without embedding export"]},
    ],
    "few_shot_learning": [
        {"pattern_id": "fewshot_protonet_encoder", "core_blocks": ["encoder", "prototype_layer", "distance_metric"],
         "training_defaults": {"optimizer": "adam", "loss_function": "cross_entropy_on_distances"},
         "evaluation_defaults": {"primary_metric": "few_shot_accuracy"},
         "strengths": ["simple metric learning", "canonical"], "weaknesses": ["fixed prototype"],
         "must_not_do": ["non-episodic dataloader", "missing support/query split"]},
        {"pattern_id": "fewshot_cosine_classifier", "core_blocks": ["encoder", "cosine_similarity_head"],
         "training_defaults": {"optimizer": "adam", "loss_function": "cross_entropy"},
         "evaluation_defaults": {"primary_metric": "few_shot_accuracy"},
         "strengths": ["scale-invariant", "simple"], "weaknesses": ["limited expressiveness"],
         "must_not_do": ["episodic sampler absent"]},
    ],
    "tabular_prediction": [
        {"pattern_id": "tabular_mlp", "core_blocks": ["fc_block", "batch_norm", "dropout"],
         "training_defaults": {"optimizer": "adam", "loss_function": "mse"},
         "evaluation_defaults": {"primary_metric": "rmse"},
         "strengths": ["universal approximator"], "weaknesses": ["no inductive bias for tabular"],
         "must_not_do": ["image augmentation on tabular data"]},
        {"pattern_id": "tabular_residual_mlp", "core_blocks": ["residual_fc_block", "layer_norm", "dropout"],
         "training_defaults": {"optimizer": "adam", "loss_function": "mse"},
         "evaluation_defaults": {"primary_metric": "rmse"},
         "strengths": ["deeper networks via residual", "stable gradient"], "weaknesses": ["more params"],
         "must_not_do": ["conv layers on tabular data"]},
        {"pattern_id": "tabular_embedding_mlp", "core_blocks": ["categorical_embedding", "fc_block"],
         "training_defaults": {"optimizer": "adam", "loss_function": "cross_entropy"},
         "evaluation_defaults": {"primary_metric": "accuracy"},
         "strengths": ["handles categorical features natively"], "weaknesses": ["requires feature engineering"],
         "must_not_do": ["one-hot encoding without embedding for high cardinality"]},
    ],
    "timeseries_prediction": [
        {"pattern_id": "ts_lstm_lite", "core_blocks": ["lstm", "fc_head"],
         "training_defaults": {"optimizer": "adam", "loss_function": "mse"},
         "evaluation_defaults": {"primary_metric": "mae"},
         "strengths": ["temporal modeling"], "weaknesses": ["slow training"],
         "must_not_do": ["shuffle time series without preserving order"]},
        {"pattern_id": "ts_tcn_lite", "core_blocks": ["causal_conv1d", "residual_block", "fc_head"],
         "training_defaults": {"optimizer": "adam", "loss_function": "mse"},
         "evaluation_defaults": {"primary_metric": "mae"},
         "strengths": ["parallelizable", "flexible receptive field"], "weaknesses": ["needs careful dilation"],
         "must_not_do": ["non-causal convolution for autoregressive task"]},
        {"pattern_id": "ts_transformer_lite", "core_blocks": ["positional_encoding", "transformer_encoder", "fc_head"],
         "training_defaults": {"optimizer": "adam", "loss_function": "mse"},
         "evaluation_defaults": {"primary_metric": "mae"},
         "strengths": ["attention for long range", "proven architecture"], "weaknesses": ["quadratic memory"],
         "must_not_do": ["standard attention on very long sequences without windowing"]},
    ],
    "generation": [
        {"pattern_id": "gen_vae_lite", "core_blocks": ["encoder", "reparameterize", "decoder"],
         "training_defaults": {"optimizer": "adam", "loss_function": "elbo"},
         "evaluation_defaults": {"primary_metric": "fid"},
         "strengths": ["principled latent space", "stable training"], "weaknesses": ["blurry samples"],
         "must_not_do": ["decoder without skip connections for high-res"]},
        {"pattern_id": "gen_gan_lite", "core_blocks": ["generator", "discriminator"],
         "training_defaults": {"optimizer": "adam", "loss_function": "bce_adversarial"},
         "evaluation_defaults": {"primary_metric": "fid"},
         "strengths": ["sharp samples"], "weaknesses": ["mode collapse", "training instability"],
         "must_not_do": ["generator without spectral norm under small budget"]},
        {"pattern_id": "gen_diffusion_ddpm", "core_blocks": ["unet_denoiser", "noise_scheduler", "forward_diffusion", "reverse_sampling"],
         "training_defaults": {"optimizer": "adam", "loss_function": "mse_noise_prediction"},
         "evaluation_defaults": {"primary_metric": "fid"},
         "strengths": ["high quality", "stable training", "principled"], "weaknesses": ["slow sampling (1000 steps)"],
         "must_not_do": ["diffusion without noise schedule contract"]},
        {"pattern_id": "gen_diffusion_ddim", "core_blocks": ["unet_denoiser", "noise_scheduler", "deterministic_sampling"],
         "training_defaults": {"optimizer": "adam", "loss_function": "mse_noise_prediction"},
         "evaluation_defaults": {"primary_metric": "fid"},
         "strengths": ["fast deterministic sampling (50 steps)", "same training as DDPM"], "weaknesses": ["slightly lower quality than DDPM"],
         "must_not_do": ["non-deterministic sampling claiming DDIM"]},
        {"pattern_id": "gen_latent_diffusion", "core_blocks": ["vae_encoder", "vae_decoder", "unet_denoiser", "latent_noise_scheduler"],
         "training_defaults": {"optimizer": "adam", "loss_function": "mse_latent_noise"},
         "evaluation_defaults": {"primary_metric": "fid"},
         "strengths": ["efficient (diffusion in latent space)", "high-res capable"], "weaknesses": ["VAE quality bottleneck"],
         "must_not_do": ["diffusion in pixel space for high-res (use latent)"]},
        {"pattern_id": "gen_conditional_diffusion", "core_blocks": ["unet_denoiser", "conditioning_module", "noise_scheduler", "classifier_free_guidance"],
         "training_defaults": {"optimizer": "adam", "loss_function": "mse_noise_prediction + cfg"},
         "evaluation_defaults": {"primary_metric": "fid"},
         "strengths": ["controllable generation", "text/class/image conditioning"], "weaknesses": ["needs conditioning data"],
         "must_not_do": ["classifier-free guidance without dropout on condition"]},
    ],
    "meta_learning": [
        {"pattern_id": "meta_maml_lite", "core_blocks": ["base_learner", "meta_optimizer", "inner_loop"],
         "training_defaults": {"optimizer": "adam", "loss_function": "cross_entropy"},
         "evaluation_defaults": {"primary_metric": "meta_test_accuracy"},
         "strengths": ["model-agnostic", "gradient-based"], "weaknesses": ["second-order gradients expensive"],
         "must_not_do": ["inner loop without task-specific adaptation"]},
        {"pattern_id": "meta_reptile_lite", "core_blocks": ["base_learner", "outer_step"],
         "training_defaults": {"optimizer": "sgd", "loss_function": "cross_entropy"},
         "evaluation_defaults": {"primary_metric": "meta_test_accuracy"},
         "strengths": ["first-order only", "simple"], "weaknesses": ["slower convergence than MAML"],
         "must_not_do": ["single task training without meta-batch"]},
    ],
    "contrastive_learning": [
        {"pattern_id": "cl_simclr_lite", "core_blocks": ["encoder", "projection_head", "ntxent_loss"],
         "training_defaults": {"optimizer": "adam", "loss_function": "ntxent"},
         "evaluation_defaults": {"primary_metric": "linear_probe_accuracy"},
         "strengths": ["simple framework", "strong baselines"], "weaknesses": ["needs large batch size"],
         "must_not_do": ["contrastive without augmentation pipeline", "no projection head"]},
        {"pattern_id": "cl_byol_lite", "core_blocks": ["online_encoder", "target_encoder", "predictor"],
         "training_defaults": {"optimizer": "adam", "loss_function": "mse_cosine"},
         "evaluation_defaults": {"primary_metric": "linear_probe_accuracy"},
         "strengths": ["no negative pairs needed"], "weaknesses": ["requires momentum update"],
         "must_not_do": ["target encoder without EMA update"]},
        {"pattern_id": "cl_moco_lite", "core_blocks": ["encoder_q", "encoder_k", "queue"],
         "training_defaults": {"optimizer": "sgd", "loss_function": "infonce"},
         "evaluation_defaults": {"primary_metric": "linear_probe_accuracy"},
         "strengths": ["memory efficient negatives"], "weaknesses": ["queue management complexity"],
         "must_not_do": ["queue without momentum encoder"]},
    ],
    "anomaly_detection": [
        {"pattern_id": "ad_autoencoder", "core_blocks": ["encoder", "decoder", "reconstruction_error"],
         "training_defaults": {"optimizer": "adam", "loss_function": "mse"},
         "evaluation_defaults": {"primary_metric": "auroc"},
         "strengths": ["unsupervised", "simple"], "weaknesses": ["may reconstruct anomalies too"],
         "must_not_do": ["training on anomalous data"]},
        {"pattern_id": "ad_patchcore", "core_blocks": ["pretrained_feature_extractor", "memory_bank", "knn_scorer", "multi_scale_patches"],
         "training_defaults": {"optimizer": "none", "loss_function": "none (memory bank — no gradient)"},
         "evaluation_defaults": {"primary_metric": "auroc"},
         "compatible_backbones": ["wide_resnet50_2", "dinov2_vitb14", "clip_vitb16", "swin_small", "efficientnet_b4"],
         "strengths": ["no training needed", "SOTA localization", "multi-scale patch inference", "any frozen backbone"],
         "weaknesses": ["memory bank scales with data", "inference slower"],
         "must_not_do": ["fine-tuning feature extractor end-to-end", "single-scale only"]},
        {"pattern_id": "ad_padim", "core_blocks": ["pretrained_feature_extractor", "gaussian_modeling", "mahalanobis_distance", "multi_scale_features"],
         "training_defaults": {"optimizer": "none", "loss_function": "none (statistical modeling)"},
         "evaluation_defaults": {"primary_metric": "auroc"},
         "strengths": ["fast inference", "probabilistic score", "multi-scale"],
         "weaknesses": ["Gaussian assumption may not hold"],
         "must_not_do": ["fitting on anomalous data", "ignoring covariance regularization"]},
        {"pattern_id": "ad_reference_matching", "core_blocks": ["pretrained_encoder", "reference_feature_bank", "cosine_similarity_scorer", "style_invariant_features"],
         "training_defaults": {"optimizer": "none", "loss_function": "none (feature matching)"},
         "evaluation_defaults": {"primary_metric": "auroc"},
         "compatible_backbones": ["dinov2_vitb14", "clip_vitb16", "clip_vitl14", "sam_vit_b"],
         "strengths": ["reference-based", "domain-agnostic", "robust to style/domain shift", "CLIP enables text-guided anomaly"],
         "weaknesses": ["reference set quality critical"],
         "must_not_do": ["reference set containing anomalies", "raw pixel comparison"]},
        {"pattern_id": "ad_efficientad", "core_blocks": ["teacher_encoder", "student_encoder", "autoencoder", "knowledge_distillation"],
         "training_defaults": {"optimizer": "adam", "loss_function": "mse_distillation"},
         "evaluation_defaults": {"primary_metric": "auroc"},
         "strengths": ["fast inference", "lightweight", "teacher-student asymmetry"],
         "weaknesses": ["teacher quality dependent"],
         "must_not_do": ["student matching teacher perfectly"]},
        {"pattern_id": "ad_diffusion", "core_blocks": ["unet_denoiser", "noise_scheduler", "reconstruction_score", "normal_only_training"],
         "training_defaults": {"optimizer": "adam", "loss_function": "mse_noise_prediction"},
         "evaluation_defaults": {"primary_metric": "auroc"},
         "compatible_backbones": ["unet_custom", "stable_diffusion_unet"],
         "strengths": ["strong reconstruction", "density estimation via score", "localization via reconstruction error"],
         "weaknesses": ["slow inference (multi-step denoising)", "heavy model"],
         "must_not_do": ["training on anomalous data", "single-step reconstruction (need multi-step)"]},
        {"pattern_id": "ad_stable_diffusion_features", "core_blocks": ["sd_vae_encoder", "sd_unet_features", "feature_matching", "multi_scale_extraction"],
         "training_defaults": {"optimizer": "none", "loss_function": "none (feature extraction only)"},
         "evaluation_defaults": {"primary_metric": "auroc"},
         "compatible_backbones": ["stable_diffusion_v1_5", "stable_diffusion_v2_1", "sdxl"],
         "strengths": ["rich semantic features from SD", "no training needed", "strong generalization"],
         "weaknesses": ["heavy model", "slow feature extraction"],
         "must_not_do": ["fine-tuning SD end-to-end for AD", "ignoring multi-scale features"]},
        {"pattern_id": "ad_flow_based", "core_blocks": ["normalizing_flow", "log_likelihood"],
         "training_defaults": {"optimizer": "adam", "loss_function": "nll"},
         "evaluation_defaults": {"primary_metric": "auroc"},
         "strengths": ["exact likelihood", "principled"], "weaknesses": ["architecture constraints"],
         "must_not_do": ["non-invertible transforms in flow"]},
    ],
    "similarity_measure": [
        {"pattern_id": "sim_siamese", "core_blocks": ["shared_encoder", "distance_layer"],
         "training_defaults": {"optimizer": "adam", "loss_function": "contrastive_margin"},
         "evaluation_defaults": {"primary_metric": "recall_at_k"},
         "strengths": ["pairwise similarity", "interpretable"], "weaknesses": ["pair mining needed"],
         "must_not_do": ["non-shared weights between branches"]},
        {"pattern_id": "sim_triplet_net", "core_blocks": ["shared_encoder", "triplet_loss"],
         "training_defaults": {"optimizer": "adam", "loss_function": "triplet_margin"},
         "evaluation_defaults": {"primary_metric": "recall_at_k"},
         "strengths": ["relative ranking", "robust"], "weaknesses": ["triplet mining expensive"],
         "must_not_do": ["random triplets without hard mining"]},
        {"pattern_id": "sim_metric_learning", "core_blocks": ["encoder", "proxy_anchor"],
         "training_defaults": {"optimizer": "adam", "loss_function": "proxy_anchor"},
         "evaluation_defaults": {"primary_metric": "recall_at_1"},
         "strengths": ["proxy-based efficiency"], "weaknesses": ["proxy initialization sensitive"],
         "must_not_do": ["proxy count mismatch with class count"]},
    ],
    "image_manipulation": [
        {"pattern_id": "imgmanip_pix2pix_lite", "core_blocks": ["encoder", "decoder", "skip_connection", "discriminator"],
         "training_defaults": {"optimizer": "adam", "loss_function": "l1+adversarial"},
         "evaluation_defaults": {"primary_metric": "psnr"},
         "strengths": ["paired image-to-image translation"], "weaknesses": ["requires paired data"],
         "must_not_do": ["unpaired training with paired loss"]},
        {"pattern_id": "imgmanip_style_transfer_lite", "core_blocks": ["encoder", "adain", "decoder"],
         "training_defaults": {"optimizer": "adam", "loss_function": "content+style"},
         "evaluation_defaults": {"primary_metric": "lpips"},
         "strengths": ["arbitrary style transfer"], "weaknesses": ["style-content tradeoff"],
         "must_not_do": ["style loss without gram matrix or equivalent"]},
        {"pattern_id": "imgmanip_inpainting_lite", "core_blocks": ["encoder", "partial_conv", "decoder"],
         "training_defaults": {"optimizer": "adam", "loss_function": "l1+perceptual"},
         "evaluation_defaults": {"primary_metric": "psnr"},
         "strengths": ["mask-aware reconstruction"], "weaknesses": ["artifacts at mask boundaries"],
         "must_not_do": ["inpainting without mask input contract"]},
    ],
    "zero_shot_learning": [
        {"pattern_id": "zsl_attribute_classifier", "core_blocks": ["visual_encoder", "attribute_embedding", "compatibility_fn"],
         "training_defaults": {"optimizer": "adam", "loss_function": "cross_entropy"},
         "evaluation_defaults": {"primary_metric": "harmonic_mean_accuracy"},
         "strengths": ["no seen-class bias"], "weaknesses": ["attribute quality dependent"],
         "must_not_do": ["evaluate only on unseen classes without harmonic mean"]},
        {"pattern_id": "zsl_generative", "core_blocks": ["feature_generator", "classifier", "semantic_embedding"],
         "training_defaults": {"optimizer": "adam", "loss_function": "cross_entropy+reconstruction"},
         "evaluation_defaults": {"primary_metric": "harmonic_mean_accuracy"},
         "strengths": ["synthesize unseen features"], "weaknesses": ["generator quality critical"],
         "must_not_do": ["training classifier on seen classes only (hubness problem)"]},
    ],
    "physics_informed": [
        {"pattern_id": "pinn_mlp", "core_blocks": ["mlp_network", "pde_residual_loss", "boundary_loss"],
         "training_defaults": {"optimizer": "adam", "loss_function": "mse_pde+boundary"},
         "evaluation_defaults": {"primary_metric": "pde_residual_l2"},
         "strengths": ["mesh-free", "continuous solution", "physics-constrained"],
         "weaknesses": ["training instability for stiff PDEs", "spectral bias"],
         "must_not_do": ["train without PDE residual loss", "ignore boundary conditions"]},
        {"pattern_id": "pinn_fourier_features", "core_blocks": ["fourier_embedding", "mlp_network", "pde_residual_loss"],
         "training_defaults": {"optimizer": "adam", "loss_function": "mse_pde+boundary"},
         "evaluation_defaults": {"primary_metric": "pde_residual_l2"},
         "strengths": ["overcomes spectral bias", "captures high-frequency"], "weaknesses": ["more params"],
         "must_not_do": ["fourier features without frequency tuning"]},
        {"pattern_id": "deeponet", "core_blocks": ["branch_net", "trunk_net", "inner_product"],
         "training_defaults": {"optimizer": "adam", "loss_function": "mse"},
         "evaluation_defaults": {"primary_metric": "relative_l2_error"},
         "strengths": ["operator learning", "generalizes across inputs"], "weaknesses": ["needs diverse training data"],
         "must_not_do": ["single PDE instance only (defeats operator learning purpose)"]},
        {"pattern_id": "neural_ode_lite", "core_blocks": ["ode_func_net", "ode_solver", "adjoint_method"],
         "training_defaults": {"optimizer": "adam", "loss_function": "mse"},
         "evaluation_defaults": {"primary_metric": "trajectory_mse"},
         "strengths": ["continuous dynamics", "memory efficient (adjoint)"], "weaknesses": ["slow solver"],
         "must_not_do": ["fixed-step solver for stiff systems"]},
    ],
    "one_shot_learning": [
        {"pattern_id": "oneshot_matching_net", "core_blocks": ["embedding_encoder", "attention_kernel", "softmax_classifier"],
         "training_defaults": {"optimizer": "adam", "loss_function": "cross_entropy"},
         "evaluation_defaults": {"primary_metric": "one_shot_accuracy"},
         "strengths": ["attention over support set"], "weaknesses": ["fixed support size"],
         "must_not_do": ["non-episodic evaluation"]},
        {"pattern_id": "oneshot_siamese_verifier", "core_blocks": ["shared_encoder", "distance_layer", "binary_classifier"],
         "training_defaults": {"optimizer": "adam", "loss_function": "binary_cross_entropy"},
         "evaluation_defaults": {"primary_metric": "one_shot_accuracy"},
         "strengths": ["pairwise verification", "simple"], "weaknesses": ["O(N) comparisons at test"],
         "must_not_do": ["non-shared encoder weights"]},
    ],
}

# 각 pattern에 task_family와 modality 자동 추가
_NON_IMAGE_FAMILIES = {"tabular_prediction", "timeseries_prediction"}
for _family, _patterns in PATTERN_LIBRARY.items():
    _modality = "tabular" if _family in _NON_IMAGE_FAMILIES else "image"
    for _p in _patterns:
        _p.setdefault("task_family", _family)
        _p.setdefault("modality", _modality)


# ──────────────────────────────────────────────────────────
# B. Baseline Synthesizer
# ──────────────────────────────────────────────────────────

BASELINE_LIBRARY: dict[str, list[dict]] = {
    "classification": [
        {"name": "simple_cnn_baseline", "source": "internal_prior", "expected_role": "weak baseline", "complexity": "low"},
        {"name": "residual_cnn_baseline", "source": "internal_prior", "expected_role": "strong baseline", "complexity": "medium"},
    ],
    "restoration_denoising": [
        {"name": "dncnn_baseline", "source": "internal_prior", "expected_role": "strong baseline", "complexity": "medium"},
        {"name": "unet_denoising_baseline", "source": "internal_prior", "expected_role": "strong baseline", "complexity": "medium"},
        {"name": "residual_cnn_denoising_baseline", "source": "internal_prior", "expected_role": "weak baseline", "complexity": "low"},
    ],
    "super_resolution": [
        {"name": "srcnn_baseline", "source": "internal_prior", "expected_role": "weak baseline", "complexity": "low"},
        {"name": "edsr_lite_baseline", "source": "internal_prior", "expected_role": "strong baseline", "complexity": "medium"},
        {"name": "subpixel_cnn_baseline", "source": "internal_prior", "expected_role": "strong baseline", "complexity": "medium"},
    ],
    "detection": [
        {"name": "fcos_lite_baseline", "source": "internal_prior", "expected_role": "strong baseline", "complexity": "medium"},
        {"name": "yolo_lite_baseline", "source": "internal_prior", "expected_role": "strong baseline", "complexity": "medium"},
    ],
    "representation_learning": [
        {"name": "supervised_encoder_baseline", "source": "internal_prior", "expected_role": "supervised baseline", "complexity": "low"},
        {"name": "contrastive_encoder_baseline", "source": "internal_prior", "expected_role": "self-supervised baseline", "complexity": "medium"},
    ],
    "few_shot_learning": [
        {"name": "protonet_baseline", "source": "internal_prior", "expected_role": "strong baseline", "complexity": "medium"},
        {"name": "cosine_classifier_baseline", "source": "internal_prior", "expected_role": "simple baseline", "complexity": "low"},
    ],
    "generation": [
        {"name": "vae_baseline", "source": "internal_prior", "expected_role": "strong baseline", "complexity": "medium"},
        {"name": "gan_baseline", "source": "internal_prior", "expected_role": "strong baseline", "complexity": "high"},
    ],
    "meta_learning": [
        {"name": "maml_baseline", "source": "internal_prior", "expected_role": "strong baseline", "complexity": "high"},
        {"name": "reptile_baseline", "source": "internal_prior", "expected_role": "simple baseline", "complexity": "medium"},
    ],
    "contrastive_learning": [
        {"name": "simclr_baseline", "source": "internal_prior", "expected_role": "strong baseline", "complexity": "medium"},
        {"name": "byol_baseline", "source": "internal_prior", "expected_role": "strong baseline", "complexity": "medium"},
    ],
    "anomaly_detection": [
        {"name": "autoencoder_ad_baseline", "source": "internal_prior", "expected_role": "simple baseline", "complexity": "low"},
        {"name": "patchcore_baseline", "source": "internal_prior", "expected_role": "SOTA baseline (memory bank)", "complexity": "medium"},
        {"name": "padim_baseline", "source": "internal_prior", "expected_role": "statistical baseline (Gaussian)", "complexity": "medium"},
        {"name": "efficientad_baseline", "source": "internal_prior", "expected_role": "lightweight teacher-student baseline", "complexity": "medium"},
    ],
    "similarity_measure": [
        {"name": "siamese_baseline", "source": "internal_prior", "expected_role": "strong baseline", "complexity": "medium"},
        {"name": "triplet_net_baseline", "source": "internal_prior", "expected_role": "strong baseline", "complexity": "medium"},
    ],
    "image_manipulation": [
        {"name": "pix2pix_baseline", "source": "internal_prior", "expected_role": "strong baseline", "complexity": "high"},
        {"name": "unet_inpainting_baseline", "source": "internal_prior", "expected_role": "simple baseline", "complexity": "medium"},
    ],
    "zero_shot_learning": [
        {"name": "attribute_classifier_baseline", "source": "internal_prior", "expected_role": "strong baseline", "complexity": "medium"},
        {"name": "generative_zsl_baseline", "source": "internal_prior", "expected_role": "strong baseline", "complexity": "high"},
    ],
    "one_shot_learning": [
        {"name": "matching_net_baseline", "source": "internal_prior", "expected_role": "strong baseline", "complexity": "medium"},
        {"name": "siamese_verifier_baseline", "source": "internal_prior", "expected_role": "simple baseline", "complexity": "low"},
    ],
    "physics_informed": [
        {"name": "pinn_mlp_baseline", "source": "internal_prior", "expected_role": "standard PINN baseline", "complexity": "medium"},
        {"name": "deeponet_baseline", "source": "internal_prior", "expected_role": "operator learning baseline", "complexity": "high"},
    ],
}

for _family, _baselines in BASELINE_LIBRARY.items():
    for _b in _baselines:
        _b.setdefault("task_family", _family)


def synthesize_baselines(task_family: str) -> list[dict]:
    """task_family에 맞는 baseline set을 반환한다."""
    return BASELINE_LIBRARY.get(task_family, BASELINE_LIBRARY.get("classification", []))


# ──────────────────────────────────────────────────────────
# B-2. Family File Layout Registry
# ──────────────────────────────────────────────────────────

FAMILY_LAYOUTS: dict[str, list[str]] = {
    "classification":           ["model.py", "module.py", "data.py", "configs/default.yaml"],
    "restoration_denoising":    ["model.py", "module.py", "data.py", "configs/default.yaml"],
    "super_resolution":         ["model.py", "module.py", "data.py", "configs/default.yaml"],
    "segmentation":             ["model.py", "module.py", "data.py", "configs/default.yaml"],
    "detection":                ["model.py", "module.py", "data.py", "ops.py", "postprocess.py", "eval_detection.py", "configs/default.yaml"],
    "representation_learning":  ["encoder.py", "module.py", "data.py", "probe.py", "eval_probe.py", "configs/default.yaml"],
    "few_shot_learning":        ["model.py", "module.py", "data.py", "episodic.py", "eval_fewshot.py", "configs/default.yaml"],
    "generation":               ["generator.py", "discriminator.py", "module.py", "data.py", "eval_generation.py", "configs/default.yaml"],
    "meta_learning":            ["model.py", "module.py", "data.py", "meta_learner.py", "eval_meta.py", "configs/default.yaml"],
    "contrastive_learning":     ["encoder.py", "module.py", "data.py", "augmentation.py", "probe.py", "configs/default.yaml"],
    "anomaly_detection":        ["feature_extractor.py", "memory_bank.py", "anomaly_scorer.py", "module.py", "data.py", "eval_anomaly.py", "configs/default.yaml"],
    "similarity_measure":       ["encoder.py", "module.py", "data.py", "mining.py", "eval_retrieval.py", "configs/default.yaml"],
    "image_manipulation":       ["model.py", "module.py", "data.py", "configs/default.yaml"],
    "zero_shot_learning":       ["encoder.py", "module.py", "data.py", "semantic_embed.py", "eval_zsl.py", "configs/default.yaml"],
    "one_shot_learning":        ["model.py", "module.py", "data.py", "episodic.py", "eval_oneshot.py", "configs/default.yaml"],
    "physics_informed":         ["model.py", "module.py", "data.py", "pde_residual.py", "boundary.py", "collocation.py", "configs/default.yaml"],
    "tabular_prediction":       ["model.py", "module.py", "data.py", "configs/default.yaml"],
    "timeseries_prediction":    ["model.py", "module.py", "data.py", "configs/default.yaml"],
}


def get_family_layout(task_family: str) -> list[str]:
    """task_family에 맞는 생성 파일 목록을 반환한다."""
    return FAMILY_LAYOUTS.get(task_family, FAMILY_LAYOUTS["classification"])


# ──────────────────────────────────────────────────────────
# C. Skeleton Selector
# ──────────────────────────────────────────────────────────

def select_task_skeleton(task_family: str) -> str:
    """task_family에 맞는 skeleton 디렉토리명을 반환한다.

    experiments/template_skeletons/{family}/ 또는 fallback으로 experiments/template/
    """
    from pathlib import Path
    skeleton_dir = Path("experiments/template_skeletons") / task_family
    if skeleton_dir.exists():
        return str(skeleton_dir)
    return "experiments/template"


# ──────────────────────────────────────────────────────────
# D. Family-Specific Experiment Spec Contracts
# ──────────────────────────────────────────────────────────

FAMILY_CONTRACTS: dict[str, dict] = {
    "restoration_denoising": {
        "noise_model_contract": {"expected_noise_type": "gaussian|poisson|unknown", "paired_data_required": True},
        "pixel_range_contract": {"input_range": "[0,1]", "output_range": "[0,1]"},
        "reconstruction_loss_contract": {"allowed": ["l1", "mse", "charbonnier"]},
    },
    "super_resolution": {
        "upscale_factor_contract": {"required": True, "default_factor": 4},
        "lr_hr_alignment_contract": {"bicubic_downscale": True},
        "reconstruction_metric_contract": {"required_keys": ["psnr", "ssim"]},
    },
    "detection": {
        "bbox_format_contract": {"format": "xyxy"},
        "class_head_contract": {"required": True},
        "postprocess_contract": {"nms_required": True, "score_threshold": 0.05},
        "detection_eval_contract": {"primary_metric": "mAP", "required_keys": ["mAP", "AP50"]},
    },
    "representation_learning": {
        "embedding_contract": {"embedding_dim": 128, "l2_normalized": True},
        "projection_head_contract": {"required_for_contrastive": True},
        "probe_evaluation_contract": {"required": True, "probe_type": "linear_probe"},
    },
    "few_shot_learning": {
        "episodic_sampling_contract": {"required": True, "n_way": 5, "k_shot": 1, "q_query": 15},
        "support_query_contract": {"explicit_split_required": True},
        "fewshot_eval_contract": {"primary_metric": "few_shot_accuracy"},
    },
    "generation": {
        "latent_space_contract": {"required": True, "latent_dim": 128},
        "sample_quality_contract": {"primary_metric": "fid", "required_keys": ["fid"]},
        "noise_schedule_contract": {"required_for_diffusion": True},
    },
    "meta_learning": {
        "inner_loop_contract": {"required": True, "inner_steps": 5, "inner_lr": 0.01},
        "meta_batch_contract": {"required": True, "tasks_per_batch": 4},
        "meta_eval_contract": {"primary_metric": "meta_test_accuracy"},
    },
    "contrastive_learning": {
        "augmentation_contract": {"required": True, "min_augmentations": 2},
        "projection_head_contract": {"required": True, "hidden_dim": 256},
        "temperature_contract": {"default": 0.07},
        "probe_evaluation_contract": {"required": True, "probe_type": "linear_probe"},
    },
    "anomaly_detection": {
        "normal_only_training_contract": {"required": True},
        "anomaly_score_contract": {"required": True, "score_type": "reconstruction_error|distance|likelihood|feature_distance"},
        "feature_extractor_contract": {
            "type": "pretrained|self_supervised|frozen|light_tuning",
            "backbone_options": {
                "cnn": ["resnet18", "resnet50", "wide_resnet50_2", "efficientnet_b0", "efficientnet_b4",
                        "convnext_tiny", "convnext_small", "convnextv2_tiny", "convnextv2_base"],
                "vit_self_supervised": ["dino_vits16", "dino_vitb16",
                                        "dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14",
                                        "dinov3_vitb14", "dinov3_vitl14"],
                "vit_vision_language": ["clip_vitb32", "clip_vitb16", "clip_vitl14",
                                        "siglip_vitb16", "siglip2_vitb16", "siglip2_so400m"],
                "swin": ["swin_tiny", "swin_small", "swin_base", "swinv2_tiny", "swinv2_base"],
                "sam": ["sam_vit_b", "sam_vit_l", "sam2_vit_b", "sam2_vit_l"],
                "mae": ["mae_vitb16", "mae_vitl16"],
                "ssm_hybrid": ["mambavision_tiny", "mambavision_small", "mambavision_base",
                               "vim_small", "vmamba_tiny"],
                "diffusion_backbone": ["stable_diffusion_v1_5", "stable_diffusion_v2_1", "sdxl"],
            },
            "selection_guide": {
                "industrial_texture": "dinov2_vitb14 or wide_resnet50_2 (strong texture features)",
                "fine_grained_defect": "dinov2_vitl14 or dinov3_vitl14 (highest feature quality)",
                "domain_shift_robust": "siglip2_vitb16 or dinov2 (vision-language or self-supervised)",
                "lightweight_fast": "resnet18 or efficientnet_b0 or dino_vits16 or mambavision_tiny (SSM: linear complexity)",
                "multi_modal_context": "siglip2_vitb16 or clip_vitb16 (text-guided anomaly definition)",
                "long_range_dependency": "mambavision_small or swinv2_base (global context with efficiency)",
                "highest_quality": "dinov3_vitl14 or dinov2_vitg14 (largest self-supervised ViT)",
            },
            "multi_scale_layers": True,
            "freeze_policy": "frozen|light_tuning_last_block|adapter_tuning",
        },
        "memory_bank_contract": {
            "required_for": ["patchcore", "padim", "reference_matching"],
            "patch_level": True,
            "coreset_sampling": "greedy|random",
        },
        "anomaly_localization_contract": {
            "pixel_level_score": True,
            "score_aggregation": "max|mean|weighted",
            "multi_scale_inference": True,
        },
        "domain_robustness_contract": {
            "style_invariant_features": True,
            "score_stability_under_perturbation": True,
        },
        "threshold_contract": {"method": "percentile", "default_percentile": 95},
        "ad_eval_contract": {
            "primary_metric": "auroc",
            "required_keys": ["auroc", "f1", "fpr_at_normal", "false_alarm_per_image"],
            "optional_keys": ["anomalous_area_ratio", "score_stability", "threshold_robustness",
                              "inference_time_ms", "peak_memory_mb", "throughput_fps"],
        },
    },
    "similarity_measure": {
        "embedding_contract": {"required": True, "embedding_dim": 128, "l2_normalized": True},
        "pair_mining_contract": {"required": True, "strategy": "hard|semi-hard|random"},
        "similarity_eval_contract": {"primary_metric": "recall_at_1", "required_keys": ["recall_at_1", "recall_at_5"]},
    },
    "image_manipulation": {
        "input_output_pair_contract": {"required": True, "paired_data": True},
        "mask_contract": {"required_for_inpainting": True},
        "pixel_range_contract": {"input_range": "[0,1]", "output_range": "[0,1]"},
        "manipulation_eval_contract": {"primary_metric": "psnr", "required_keys": ["psnr", "ssim"]},
    },
    "zero_shot_learning": {
        "seen_unseen_split_contract": {"required": True},
        "semantic_embedding_contract": {"required": True, "type": "attribute|word2vec|bert"},
        "zsl_eval_contract": {"primary_metric": "harmonic_mean_accuracy", "required_keys": ["seen_acc", "unseen_acc", "harmonic_mean"]},
    },
    "one_shot_learning": {
        "episodic_sampling_contract": {"required": True, "n_way": 5, "k_shot": 1, "q_query": 15},
        "support_query_contract": {"explicit_split_required": True},
        "oneshot_eval_contract": {"primary_metric": "one_shot_accuracy"},
    },
    "physics_informed": {
        "pde_residual_contract": {"required": True, "residual_type": "strong_form|weak_form"},
        "boundary_condition_contract": {"required": True, "bc_types": ["dirichlet", "neumann"]},
        "collocation_contract": {"required": True, "sampling_strategy": "random|latin_hypercube|sobol"},
        "conservation_law_contract": {"optional": True, "description": "energy/mass/momentum conservation"},
        "physics_eval_contract": {"primary_metric": "pde_residual_l2", "required_keys": ["pde_residual_l2", "boundary_error"]},
    },
}


def get_family_contract(task_family: str) -> dict:
    """task_family에 맞는 추가 spec contract를 반환한다."""
    return FAMILY_CONTRACTS.get(task_family, {})


# ──────────────────────────────────────────────────────────
# E. Generation Prior Planner
# ──────────────────────────────────────────────────────────

GENERATION_PRIORS: dict[str, dict] = {
    "classification": {
        "critical_interfaces": ["forward", "loss_computation", "metric_computation"],
        "likely_failure_modes": ["overfitting on small data", "class imbalance"],
        "must_not_do": ["oversized backbone under param budget"],
    },
    "restoration_denoising": {
        "critical_interfaces": ["forward (noisy→clean)", "reconstruction_loss", "psnr_ssim_eval"],
        "likely_failure_modes": ["pixel range mismatch", "residual vs direct prediction confusion"],
        "must_not_do": ["heavy perceptual loss as default on small data"],
    },
    "super_resolution": {
        "critical_interfaces": ["forward (LR→HR)", "upsample_block", "reconstruction_metric"],
        "likely_failure_modes": ["upscale factor mismatch", "checkerboard artifacts"],
        "must_not_do": ["upscale without factor contract", "deconv without anti-aliasing"],
    },
    "detection": {
        "critical_interfaces": ["bbox_head_forward", "target_assignment", "postprocess_nms", "coco_eval"],
        "likely_failure_modes": ["bbox format mismatch", "metric contract mismatch", "empty detections"],
        "must_not_do": ["detection model without NMS", "bbox format undefined"],
    },
    "representation_learning": {
        "critical_interfaces": ["encoder_forward", "projection_head_forward", "embedding_export", "probe_eval"],
        "likely_failure_modes": ["projection head missing", "embedding normalization mismatch", "no probe"],
        "must_not_do": ["contrastive without projection head", "no downstream evaluation"],
    },
    "few_shot_learning": {
        "critical_interfaces": ["episodic_sampler", "support_query_split", "prototype_construction", "distance_metric"],
        "likely_failure_modes": ["non-episodic dataloader", "wrong episode shape", "metric mismatch"],
        "must_not_do": ["standard dataloader for episodic task", "missing support/query split"],
    },
    "physics_informed": {
        "critical_interfaces": ["pde_residual_computation", "boundary_condition_loss", "collocation_sampling", "solution_prediction"],
        "likely_failure_modes": ["spectral bias (low-frequency only)", "stiff PDE instability", "boundary condition violation", "collocation point imbalance"],
        "must_not_do": ["train without PDE residual loss", "ignore boundary conditions", "fixed collocation without resampling"],
    },
    "generation": {
        "critical_interfaces": ["generator_forward", "latent_sampling", "sample_quality_eval"],
        "likely_failure_modes": ["mode collapse (GAN)", "blurry output (VAE)", "slow sampling (diffusion)"],
        "must_not_do": ["GAN without gradient penalty or spectral norm", "diffusion without noise schedule"],
    },
    "meta_learning": {
        "critical_interfaces": ["inner_loop_update", "outer_loop_update", "task_sampler", "meta_eval"],
        "likely_failure_modes": ["inner loop divergence", "meta-overfitting", "task distribution mismatch"],
        "must_not_do": ["single-task training claiming meta-learning", "no inner loop adaptation"],
    },
    "contrastive_learning": {
        "critical_interfaces": ["encoder_forward", "projection_head", "augmentation_pipeline", "contrastive_loss", "probe_eval"],
        "likely_failure_modes": ["representation collapse", "weak augmentation", "no downstream eval"],
        "must_not_do": ["contrastive without augmentation", "no projection head", "no linear probe"],
    },
    "anomaly_detection": {
        "critical_interfaces": [
            "feature_extractor_forward (pretrained, frozen/light-tuning)",
            "memory_bank_construction (normal features only)",
            "multi_scale_patch_extraction",
            "anomaly_score_computation (kNN/Mahalanobis/cosine)",
            "score_aggregation (patch→image level)",
            "anomaly_localization_map",
            "threshold_selection",
            "robustness_evaluation (style/noise perturbation)",
        ],
        "likely_failure_modes": [
            "training on anomalies (data contamination)",
            "feature extractor domain gap (ImageNet features on industrial data)",
            "memory bank too large (OOM) or too small (underfitting)",
            "score collapse (all scores identical)",
            "threshold sensitivity (small threshold change → large FPR change)",
            "domain shift between normal training and test conditions",
            "single-scale inference missing fine-grained anomalies",
        ],
        "must_not_do": [
            "training set containing anomalies",
            "fine-tuning pretrained backbone end-to-end without freezing",
            "single-scale patch inference only",
            "raw pixel comparison instead of feature-space comparison",
            "no anomaly localization output",
        ],
    },
    "similarity_measure": {
        "critical_interfaces": ["encoder_forward", "distance_computation", "pair_mining", "retrieval_eval"],
        "likely_failure_modes": ["embedding collapse", "poor hard mining", "metric mismatch"],
        "must_not_do": ["non-shared weights in siamese", "random pairs without mining strategy"],
    },
    "image_manipulation": {
        "critical_interfaces": ["forward (input→output)", "mask_handling", "reconstruction_loss", "perceptual_loss"],
        "likely_failure_modes": ["paired/unpaired data mismatch", "mask boundary artifacts", "color shift"],
        "must_not_do": ["inpainting without mask input", "style transfer without content preservation loss"],
    },
    "zero_shot_learning": {
        "critical_interfaces": ["visual_encoder", "semantic_embedding", "compatibility_function", "generalized_eval"],
        "likely_failure_modes": ["seen-class bias", "hubness problem", "attribute quality"],
        "must_not_do": ["evaluate only on unseen classes (must use harmonic mean)", "ignore seen-class performance"],
    },
    "one_shot_learning": {
        "critical_interfaces": ["encoder_forward", "support_embedding", "query_classification", "episodic_eval"],
        "likely_failure_modes": ["overfitting to support", "non-episodic evaluation", "class imbalance in episodes"],
        "must_not_do": ["non-episodic training/evaluation", "fixed class assignment across episodes"],
    },
}


def get_generation_prior(task_family: str) -> dict:
    """generation prior를 반환한다."""
    prior = GENERATION_PRIORS.get(task_family, GENERATION_PRIORS.get("classification", {}))
    return {
        "task_family": task_family,
        "recommended_patterns": [p["pattern_id"] for p in PATTERN_LIBRARY.get(task_family, [])],
        "baseline_families": [b["name"] for b in synthesize_baselines(task_family)],
        **prior,
    }


# ──────────────────────────────────────────────────────────
# F. Literature-to-Code Bridge
# ──────────────────────────────────────────────────────────

LITERATURE_CODE_PRIORS: dict[str, dict] = {
    "restoration_denoising": {
        "architecture_hint": "residual learning (predict noise, not clean)",
        "loss_hint": "L1 or Charbonnier for stable reconstruction",
        "training_hint": "patch-based training for efficiency",
        "evaluation_hint": "PSNR/SSIM on full image, not patches",
        "forbidden_patterns": ["output without pixel clamp"],
    },
    "super_resolution": {
        "architecture_hint": "residual trunk + sub-pixel upsampling",
        "loss_hint": "L1 for PSNR, perceptual for visual quality",
        "training_hint": "random crop pairs (LR/HR) for augmentation",
        "evaluation_hint": "PSNR/SSIM on Y channel (if applicable)",
        "forbidden_patterns": ["upscale without factor alignment"],
    },
    "detection": {
        "architecture_hint": "anchor-free head preferred for simplicity",
        "loss_hint": "focal loss for classification + GIoU for regression",
        "training_hint": "multi-scale training helps generalization",
        "evaluation_hint": "COCO-style mAP with IoU thresholds",
        "forbidden_patterns": ["detection without NMS postprocess"],
    },
    "representation_learning": {
        "architecture_hint": "encoder + projection head (2-layer MLP)",
        "loss_hint": "NT-Xent for contrastive, MSE for reconstruction",
        "training_hint": "strong augmentation is critical for contrastive",
        "evaluation_hint": "linear probe accuracy on frozen features",
        "forbidden_patterns": ["contrastive without projection head"],
    },
    "few_shot_learning": {
        "architecture_hint": "lightweight encoder + metric-based classifier",
        "loss_hint": "cross-entropy on episode distances",
        "training_hint": "episodic training with n-way k-shot sampling",
        "evaluation_hint": "accuracy averaged over 600+ episodes",
        "forbidden_patterns": ["non-episodic training for few-shot"],
    },
    "physics_informed": {
        "architecture_hint": "MLP with Fourier features or sinusoidal activations (SIREN); DeepONet for operator learning",
        "loss_hint": "PDE residual (MSE on collocation) + boundary condition (MSE on boundary) + optional data fit",
        "training_hint": "adaptive collocation resampling; curriculum on PDE complexity; learning rate warmup",
        "evaluation_hint": "relative L2 error on solution; PDE residual norm; boundary condition satisfaction",
        "forbidden_patterns": ["training without physics loss", "ignoring boundary conditions"],
    },
    "generation": {
        "architecture_hint": "VAE: encoder-decoder with reparameterization; GAN: generator-discriminator; Diffusion: UNet denoiser",
        "loss_hint": "VAE: ELBO (recon + KL); GAN: adversarial + R1 penalty; Diffusion: MSE noise prediction",
        "training_hint": "GAN: alternate G/D updates; Diffusion: random timestep sampling",
        "evaluation_hint": "FID on 10K+ generated samples vs real distribution",
        "forbidden_patterns": ["GAN without training stability mechanism"],
    },
    "meta_learning": {
        "architecture_hint": "lightweight base learner with few inner-loop steps",
        "loss_hint": "task-specific loss in inner loop, meta-loss across tasks",
        "training_hint": "MAML: 2nd order gradients (or first-order approximation); Reptile: simple outer step",
        "evaluation_hint": "accuracy on held-out tasks, not held-out samples of same task",
        "forbidden_patterns": ["evaluating on training tasks"],
    },
    "contrastive_learning": {
        "architecture_hint": "encoder + 2-layer MLP projection head; SimCLR/BYOL/MoCo variants",
        "loss_hint": "NT-Xent with temperature scaling; BYOL: MSE/cosine on predictions",
        "training_hint": "strong augmentation is critical; large batch or momentum queue for negatives",
        "evaluation_hint": "freeze encoder, train linear probe, report probe accuracy",
        "forbidden_patterns": ["contrastive without strong augmentation", "evaluate without linear probe"],
    },
    "anomaly_detection": {
        "architecture_hint": (
            "Backbone selection: "
            "CNN (ResNet/WideResNet/EfficientNet) — fast, proven for texture anomaly. "
            "DINOv2 (ViT) — self-supervised, strongest general features, best for domain shift. "
            "CLIP (ViT) — vision-language aligned, enables text-guided anomaly definition. "
            "Swin Transformer — hierarchical ViT, good multi-scale features. "
            "SAM (ViT) — segment-anything features, strong spatial understanding. "
            "MAE (ViT) — masked autoencoder features, good for reconstruction-based AD. "
            "PatchCore/PaDiM: frozen backbone + multi-scale patch features + memory bank. "
            "EfficientAD: teacher-student distillation with lightweight student. "
            "Reference matching: feature bank from reference normal images + cosine/kNN. "
            "All approaches: freeze backbone or light-tune last block only."
        ),
        "loss_hint": (
            "Memory bank methods: no loss (feature caching only). "
            "Teacher-student: MSE distillation loss. "
            "Flow-based: negative log-likelihood. "
            "Score: kNN distance / Mahalanobis distance / cosine similarity."
        ),
        "training_hint": (
            "Ensure training set is 100% normal (no anomalies). "
            "Use pretrained ImageNet features as starting point. "
            "Freeze backbone or light-tune last block only. "
            "Multi-scale patch extraction (layers 2-3 of backbone). "
            "Coreset sampling for memory bank efficiency."
        ),
        "evaluation_hint": (
            "Image-level: AUROC, F1, FPR on normal images, false alarm per image. "
            "Pixel-level: per-pixel AUROC, anomalous area ratio on normals. "
            "Robustness: score stability under style/noise perturbation. "
            "Efficiency: inference time, peak memory, throughput."
        ),
        "forbidden_patterns": [
            "training on anomalous samples",
            "fine-tuning pretrained backbone end-to-end",
            "single-scale feature extraction only",
            "no anomaly localization map",
            "raw pixel distance instead of feature distance",
        ],
    },
    "similarity_measure": {
        "architecture_hint": "shared encoder with distance/similarity head; triplet or contrastive loss",
        "loss_hint": "contrastive margin loss or triplet loss with semi-hard mining",
        "training_hint": "hard negative mining improves convergence; batch-all or batch-hard strategy",
        "evaluation_hint": "Recall@K on retrieval benchmark",
        "forbidden_patterns": ["non-shared encoder weights in siamese", "random pair mining only"],
    },
    "image_manipulation": {
        "architecture_hint": "encoder-decoder with skip connections; discriminator for adversarial quality",
        "loss_hint": "L1 + perceptual (VGG features) + adversarial (optional)",
        "training_hint": "paired data preferred; random crop augmentation for spatial invariance",
        "evaluation_hint": "PSNR/SSIM for distortion, LPIPS for perceptual quality",
        "forbidden_patterns": ["inpainting without mask-aware convolution", "unpaired loss on paired task"],
    },
    "zero_shot_learning": {
        "architecture_hint": "visual encoder + semantic embedding space + compatibility function",
        "loss_hint": "cross-entropy on compatibility scores; calibration loss for GZSL",
        "training_hint": "train on seen classes with attribute annotations; evaluate with harmonic mean",
        "evaluation_hint": "harmonic mean of seen and unseen accuracy (GZSL protocol)",
        "forbidden_patterns": ["evaluating unseen-only accuracy without seen accuracy"],
    },
    "one_shot_learning": {
        "architecture_hint": "lightweight conv encoder + attention/matching mechanism",
        "loss_hint": "cross-entropy on episode predictions",
        "training_hint": "episodic training with 1-shot support; augment support examples if possible",
        "evaluation_hint": "accuracy averaged over 600+ 1-shot episodes",
        "forbidden_patterns": ["non-episodic evaluation", "k>1 claiming one-shot"],
    },
}


def get_literature_code_prior(task_family: str) -> dict:
    """literature-derived code prior를 반환한다."""
    return LITERATURE_CODE_PRIORS.get(task_family, {})


# ──────────────────────────────────────────────────────────
# G. Contract Tests
# ──────────────────────────────────────────────────────────

def run_family_contract_tests(task_family: str, code_files: dict, spec: dict) -> dict:
    """family-specific contract test를 실제 실행하고 결과를 반환한다.

    Returns: {"task_family_contract_ok": bool, "passed": [...], "failed": [...]}
    """
    passed: list[str] = []
    failed: list[str] = []
    contract = get_family_contract(task_family)
    model_code = code_files.get("model.py", code_files.get("model_py", ""))
    module_code = code_files.get("module.py", code_files.get("module_py", ""))
    data_code = code_files.get("data.py", code_files.get("data_py", ""))
    all_code = f"{model_code}\n{module_code}\n{data_code}".lower()
    yaml_code = code_files.get("default.yaml", code_files.get("default_yaml", "")).lower()

    # 공통 테스트
    # metric key sanity
    ev = spec.get("evaluation_config", {})
    primary = ev.get("primary_metric", "")
    required_keys = spec.get("output_contract", {}).get("required_keys", [])
    if primary and (primary in all_code or "metrics" in all_code):
        passed.append("metric_key_sanity_test")
    else:
        failed.append(f"metric_key_sanity_test: '{primary}' not found in code")

    # param budget
    arch = spec.get("model_architecture", {})
    budget = arch.get("param_budget_M")
    if budget and str(budget) in yaml_code:
        passed.append("param_budget_test")
    elif not budget:
        passed.append("param_budget_test: no budget specified")
    else:
        failed.append(f"param_budget_test: {budget}M not in default.yaml")

    # Family-specific tests
    if task_family == "restoration_denoising":
        if "noise" in data_code or "noisy" in data_code:
            passed.append("noisy_clean_pair_shape_test")
        else:
            failed.append("noisy_clean_pair_shape_test: no noisy/clean pair handling in data.py")
        if "clamp" in model_code or "clamp" in module_code:
            passed.append("pixel_range_contract_test")
        else:
            failed.append("pixel_range_contract_test: no clamp for pixel range")
        if "psnr" in all_code:
            passed.append("psnr_ssim_key_test")
        else:
            failed.append("psnr_ssim_key_test: psnr not found in code")

    elif task_family == "super_resolution":
        if "upscale" in all_code or "pixel_shuffle" in all_code or "pixelshuffle" in all_code:
            passed.append("upscale_factor_output_shape_test")
        else:
            failed.append("upscale_factor_output_shape_test: no upscale/pixel_shuffle found")
        if "psnr" in all_code:
            passed.append("psnr_ssim_key_test")
        else:
            failed.append("psnr_ssim_key_test: psnr not found")

    elif task_family == "detection":
        if "bbox" in all_code or "box" in all_code:
            passed.append("bbox_output_format_test")
        else:
            failed.append("bbox_output_format_test: no bbox handling found")
        if "cls" in model_code or "class" in model_code:
            passed.append("class_logits_presence_test")
        else:
            failed.append("class_logits_presence_test: no classification head found")
        if "map" in all_code or "ap50" in all_code or "ap" in all_code:
            passed.append("map_ap50_key_test")
        else:
            failed.append("map_ap50_key_test: mAP/AP50 not found")

    elif task_family == "representation_learning":
        if "embedding" in all_code or "emb" in all_code:
            passed.append("embedding_dimensionality_test")
        else:
            failed.append("embedding_dimensionality_test: no embedding found")
        if "projection" in all_code or "proj" in all_code:
            passed.append("projection_head_presence_test")
        else:
            failed.append("projection_head_presence_test: no projection head")
        if "probe" in all_code or "linear_probe" in all_code:
            passed.append("probe_output_contract_test")
        else:
            failed.append("probe_output_contract_test: no probe evaluation")

    elif task_family == "few_shot_learning":
        if "episod" in all_code or "episode" in all_code:
            passed.append("episodic_batch_shape_test")
        else:
            failed.append("episodic_batch_shape_test: no episodic handling")
        if "support" in all_code and "query" in all_code:
            passed.append("support_query_split_test")
        else:
            failed.append("support_query_split_test: no support/query split")
        if "n_way" in all_code or "k_shot" in all_code:
            passed.append("nway_kshot_contract_test")
        else:
            failed.append("nway_kshot_contract_test: n_way/k_shot not found")

    elif task_family == "generation":
        if "latent" in all_code or "z" in model_code:
            passed.append("latent_space_test")
        else:
            failed.append("latent_space_test: no latent space found")
        if "fid" in all_code or "sample" in all_code:
            passed.append("sample_quality_eval_test")
        else:
            failed.append("sample_quality_eval_test: no sample quality evaluation")

    elif task_family == "meta_learning":
        if "inner" in all_code or "adapt" in all_code:
            passed.append("inner_loop_test")
        else:
            failed.append("inner_loop_test: no inner loop adaptation found")
        if "task" in data_code or "meta" in data_code:
            passed.append("meta_batch_test")
        else:
            failed.append("meta_batch_test: no task/meta-batch handling")

    elif task_family == "contrastive_learning":
        if "augment" in all_code or "transform" in all_code:
            passed.append("augmentation_pipeline_test")
        else:
            failed.append("augmentation_pipeline_test: no augmentation found")
        if "proj" in all_code or "projection" in all_code:
            passed.append("projection_head_test")
        else:
            failed.append("projection_head_test: no projection head")
        if "probe" in all_code or "linear" in module_code:
            passed.append("linear_probe_test")
        else:
            failed.append("linear_probe_test: no probe evaluation")

    elif task_family == "anomaly_detection":
        if "anomal" in all_code or "score" in all_code or "reconstruct" in all_code:
            passed.append("anomaly_score_test")
        else:
            failed.append("anomaly_score_test: no anomaly scoring mechanism")
        if "auroc" in all_code or "auc" in all_code:
            passed.append("auroc_key_test")
        else:
            failed.append("auroc_key_test: auroc not found in eval")

    elif task_family == "similarity_measure":
        if "embed" in all_code or "encod" in all_code:
            passed.append("embedding_output_test")
        else:
            failed.append("embedding_output_test: no embedding output")
        if "distance" in all_code or "similar" in all_code or "triplet" in all_code:
            passed.append("distance_computation_test")
        else:
            failed.append("distance_computation_test: no distance/similarity computation")
        if "recall" in all_code or "retriev" in all_code:
            passed.append("retrieval_eval_test")
        else:
            failed.append("retrieval_eval_test: no retrieval evaluation")

    elif task_family == "image_manipulation":
        if "mask" in all_code or "pair" in data_code:
            passed.append("input_output_pair_test")
        else:
            failed.append("input_output_pair_test: no paired data/mask handling")
        if "psnr" in all_code or "ssim" in all_code or "lpips" in all_code:
            passed.append("manipulation_metric_test")
        else:
            failed.append("manipulation_metric_test: no quality metric found")

    elif task_family == "zero_shot_learning":
        if "seen" in all_code or "unseen" in all_code or "attribute" in all_code:
            passed.append("seen_unseen_split_test")
        else:
            failed.append("seen_unseen_split_test: no seen/unseen split handling")
        if "harmonic" in all_code or "hm" in all_code:
            passed.append("harmonic_mean_eval_test")
        else:
            failed.append("harmonic_mean_eval_test: no harmonic mean evaluation")

    elif task_family == "one_shot_learning":
        if "episod" in all_code or "support" in all_code:
            passed.append("episodic_one_shot_test")
        else:
            failed.append("episodic_one_shot_test: no episodic/support handling")
        if "1" in yaml_code and ("shot" in all_code or "k_shot" in yaml_code):
            passed.append("one_shot_contract_test")
        else:
            failed.append("one_shot_contract_test: k_shot=1 not verified")

    elif task_family == "physics_informed":
        if "residual" in all_code or "pde" in all_code:
            passed.append("pde_residual_loss_test")
        else:
            failed.append("pde_residual_loss_test: no PDE residual computation")
        if "boundary" in all_code or "bc" in all_code:
            passed.append("boundary_condition_test")
        else:
            failed.append("boundary_condition_test: no boundary condition handling")
        if "collocation" in all_code or "sample" in data_code:
            passed.append("collocation_sampling_test")
        else:
            failed.append("collocation_sampling_test: no collocation point sampling")

    return {
        "task_family_contract_ok": len(failed) == 0,
        "task_family": task_family,
        "passed": passed,
        "failed": failed,
    }


def get_family_contract_tests(task_family: str) -> list[str]:
    """task_family별 필수 contract test 목록을 반환한다."""
    common = [
        "param_budget_test",
        "dummy_batch_shape_test",
        "metric_key_sanity_test",
        "optimizer_loss_wiring_test",
        "output_contract_completeness_test",
    ]
    family_tests = {
        "restoration_denoising": [
            "noisy_clean_pair_shape_test",
            "pixel_range_contract_test",
            "psnr_ssim_key_test",
        ],
        "super_resolution": [
            "lr_hr_scale_consistency_test",
            "upscale_factor_output_shape_test",
            "psnr_ssim_key_test",
        ],
        "detection": [
            "bbox_output_format_test",
            "class_logits_presence_test",
            "empty_batch_detection_test",
            "map_ap50_key_test",
        ],
        "representation_learning": [
            "embedding_dimensionality_test",
            "projection_head_presence_test",
            "probe_output_contract_test",
        ],
        "few_shot_learning": [
            "episodic_batch_shape_test",
            "support_query_split_test",
            "nway_kshot_contract_test",
            "episodic_eval_metric_test",
        ],
    }
    return common + family_tests.get(task_family, [])


# ──────────────────────────────────────────────────────────
# H. Task Family Inference (topic → family 자동 감지)
# ──────────────────────────────────────────────────────────

_FAMILY_KEYWORDS: dict[str, list[str]] = {
    "restoration_denoising": ["denois", "restoration", "noise removal", "clean"],
    "super_resolution": ["super-resolution", "super resolution", "upscale", "upsampl", "sr "],
    "segmentation": ["segmentation", "semantic", "instance seg", "panoptic"],
    "detection": ["object detect", "bbox", "yolo", "detr", "anchor", "detection model"],
    "representation_learning": ["representation", "contrastive", "self-supervised", "pretrain", "embedding"],
    "few_shot_learning": ["few-shot", "few shot", "meta-learn", "n-way", "k-shot", "episodic"],
    "tabular_prediction": ["tabular", "structured data", "csv", "dataframe"],
    "timeseries_prediction": ["time series", "timeseries", "temporal", "forecasting", "sequence predict"],
    "classification": ["classif", "recognition", "categoriz", "image classification", "predict class", "predict label", "accuracy"],
    "generation": ["generat", "gan", "vae", "diffusion", "synthesis", "image generat"],
    "meta_learning": ["meta-learn", "meta learn", "maml", "reptile", "learning to learn"],
    "contrastive_learning": ["contrastive", "simclr", "byol", "moco", "self-supervis", "ssl"],
    "anomaly_detection": ["anomaly", "outlier", "out-of-distribution", "ood", "novelty detect", "anomaly detect"],
    "similarity_measure": ["similarity", "siamese", "triplet", "metric learn", "retrieval", "embedding match"],
    "image_manipulation": ["image manipulat", "image edit", "inpaint", "style transfer", "pix2pix", "image-to-image", "coloriz"],
    "zero_shot_learning": ["zero-shot", "zero shot", "zsl", "gzsl", "unseen class"],
    "one_shot_learning": ["one-shot", "one shot", "1-shot", "single example"],
    "physics_informed": ["physics-inform", "pinn", "pde", "neural ode", "deeponet", "operator learn", "physics constrain", "navier-stokes", "conservation law"],
}


# 특화 family는 generic보다 우선 (tie-breaking)
_FAMILY_SPECIFICITY: dict[str, int] = {
    "classification": 0, "segmentation": 1, "tabular_prediction": 1, "timeseries_prediction": 1,
    "restoration_denoising": 2, "super_resolution": 2, "detection": 2, "generation": 2,
    "representation_learning": 3, "image_manipulation": 3,
    "contrastive_learning": 4, "anomaly_detection": 4, "similarity_measure": 4,
    "few_shot_learning": 5, "meta_learning": 5,
    "zero_shot_learning": 6, "one_shot_learning": 6,
}


def infer_task_family(topic: str, target_metric: str = "", problem: str = "") -> str:
    """topic, target_metric, problem_definition에서 task_family를 추론한다.

    동일 score 시 더 특화된 family를 우선한다 (zero_shot > classification 등).
    """
    text = f"{topic} {target_metric} {problem}".lower()
    scores: dict[str, int] = {}
    for family, keywords in _FAMILY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text)
        if score > 0:
            scores[family] = score
    if scores:
        return max(scores, key=lambda f: (scores[f], _FAMILY_SPECIFICITY.get(f, 0)))
    return "classification"


# ──────────────────────────────────────────────────────────
# 통합 API
# ──────────────────────────────────────────────────────────

def get_task_family_bundle(task_family: str) -> dict:
    """task_family에 대한 전체 bundle을 반환한다.

    model_generator.py에서 generation 전에 호출하여
    pattern candidates, baselines, contracts, priors를 한번에 얻는다.
    """
    return {
        "task_family": task_family,
        "pattern_candidates": PATTERN_LIBRARY.get(task_family, []),
        "synthesized_baselines": synthesize_baselines(task_family),
        "family_contract": get_family_contract(task_family),
        "generation_prior": get_generation_prior(task_family),
        "literature_code_prior": get_literature_code_prior(task_family),
        "contract_tests": get_family_contract_tests(task_family),
        "skeleton_path": select_task_skeleton(task_family),
        "file_layout": get_family_layout(task_family),
    }
