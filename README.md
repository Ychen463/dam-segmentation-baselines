# Dynamic Difficulty-Aware Curriculum Learning with Boundary Refinement for Fine-Grained Crack and Spalling Segmentation in Concrete Dams

Official implementation of our proposed method for fine-grained crack and spalling segmentation in concrete dam surfaces, featuring dynamic difficulty-aware curriculum learning and boundary refinement.

## Method Overview

Our approach builds on SegFormer-B2 with three tightly integrated modules:

1. **Online Difficulty Estimation** — tracks per-sample difficulty via EMA-smoothed loss, prediction uncertainty, boundary complexity, and foreground sparsity, combined with epoch-level z-score normalization.
2. **Class-Aware Dynamic Sampling** — tier-gated curriculum (Easy → Medium → Hard) with softmax-weighted sampling based on difficulty scores, plus spalling and late-stage hard-crack bonuses.
3. **Composite Loss with Boundary Refinement** — cross-entropy + foreground Dice + crack-specific Tversky (FN-heavy) + boundary BCE, with stage-scheduled loss weights.

<p align="center"><img src="assets/pipeline.png" width="90%" alt="Method pipeline"></p>

## Project Structure

```
Codes/
├── full_method/               # Proposed method
│   ├── config.py              # Hyperparameters & ablation presets
│   ├── model.py               # SegFormer-B2 + boundary head
│   ├── difficulty.py          # Online difficulty estimator
│   ├── sampler.py             # Tier-aware dynamic sampler
│   ├── losses.py              # Composite loss (CE + Dice + Tversky + BoundaryBCE)
│   ├── scheduler.py           # Curriculum stage & loss weight scheduler
│   ├── dataset.py             # Dataset with tier/spalling metadata
│   └── train.py               # Training loop
│
├── baseline_unet/             # U-Net (ResNet-34) — sanity check baseline
├── baseline_deeplab/          # DeepLabV3+ (ResNet-50 / ResNet-34)
├── baseline_segformer/        # SegFormer-B2 (plain & static curriculum)
├── baseline_mask2former/      # Mask2Former (Swin-Small) — upper-bound reference
│
├── shared_eval/               # Unified evaluation pipeline
│   ├── model_registry.py      # Model loading & inference wrappers
│   ├── eval_all.py            # Batch evaluation with per-tier breakdown
│   ├── metrics_full.py        # IoU, Dice, BF1, clDice, ConnR
│   ├── cldice.py              # Centreline Dice via skeletonization
│   ├── stats_significance.py  # Paired Wilcoxon + bootstrap CI
│   ├── efficiency.py          # Params / FLOPs / latency / FPS
│   └── subset_splits.py       # Low-data split generation
│
├── scripts/                   # Executable pipeline scripts
│   ├── 00_setup.sh            # Environment setup
│   ├── 00a_verify_dataset.sh  # Dataset integrity check
│   ├── 00b_make_splits.sh     # Stratified train/val/test splits
│   ├── 01–06_train_*.sh       # Baseline training (independent)
│   ├── 07_train_full_method.sh
│   ├── 07a/b/c_ablation_*.sh  # Ablation studies
│   ├── 08_eval_all.sh         # Unified evaluation
│   ├── 09_exp_e_lowdata.sh    # Low-data experiment
│   ├── 10_exp_f_efficiency.sh # Efficiency benchmark
│   └── 11_stats_significance.sh
│
├── Dataset/                   # DamSegment dataset (see below)
├── results/                   # Evaluation outputs
└── requirements.txt
```

## Dataset

**DamSegment** — concrete dam surface images with pixel-level annotations across three difficulty tiers.

| Tier | Description |
|------|-------------|
| Easy | Clear, well-lit surfaces with obvious damage |
| Medium | Moderate complexity (noise, partial occlusion) |
| Hard | Challenging lighting, fine cracks, ambiguous spalling |

**3 semantic classes:** background (0), crack (1), spalling (2)

Masks are stored as RGB PNGs: crack in the red channel, spalling in the blue channel (threshold > 127).

Splits are stratified by difficulty tier and spalling presence (80/10/10 train/val/test).

## Installation

```bash
# Clone
git clone https://github.com/Ychen463/dam-segmentation-baselines.git
cd dam-segmentation-baselines

# Create environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
# (On RunPod/CUDA machines, torch is usually pre-installed)
pip install -r requirements.txt

# Verify dataset & generate splits
bash scripts/00_setup.sh
bash scripts/00a_verify_dataset.sh
bash scripts/00b_make_splits.sh
```

## Training

All training scripts are independent and can run in parallel (GPU memory permitting).

```bash
# Each script in its own tmux session:
tmux new -s full_method
bash scripts/07_train_full_method.sh    # Proposed method (100 epochs)
# Ctrl+B, D to detach

tmux new -s segformer_a
bash scripts/01_train_segformer_A.sh    # SegFormer-B2 plain (100 epochs)

tmux new -s deeplab
bash scripts/03_train_deeplab_A.sh      # DeepLabV3+ R50 (50 epochs)

# ... similarly for 02, 04, 05, 06
```

### Ablation Studies

Ablation scripts require `07_train_full_method.sh` to define the baseline config but run independently of each other:

```bash
bash scripts/07a_ablation_modules.sh     # Module-level: A2–A5
bash scripts/07b_ablation_difficulty.sh  # Difficulty score: B0–B5
bash scripts/07c_ablation_classaware.sh  # Class-aware: A3a, A3b
```

You can also run individual ablation presets directly:

```bash
python -m full_method.train --ablation A2
python -m full_method.train --ablation B0
python -m full_method.train --no-boundary-loss --no-tversky-loss --name custom_run
```

## Evaluation

```bash
# Unified evaluation (requires all training to be complete)
bash scripts/08_eval_all.sh

# Statistical significance testing
bash scripts/11_stats_significance.sh

# Low-data experiment (requires 01 complete)
bash scripts/09_exp_e_lowdata.sh

# Efficiency benchmark (params, FLOPs, latency)
bash scripts/10_exp_f_efficiency.sh
```

## Execution Order

```
Phase 0 — Setup
  00_setup.sh → 00a_verify_dataset.sh → 00b_make_splits.sh

Phase 1 — Training (all independent, run in parallel)
  01_train_segformer_A.sh    (100 ep)
  02_train_segformer_B.sh    (100 ep)
  03_train_deeplab_A.sh      (50 ep)
  04_train_mask2former.sh    (40 ep)
  05_train_deeplab_B.sh      (30 ep)
  06_train_unet.sh           (30 ep)
  07_train_full_method.sh    (100 ep)

Phase 2 — Ablation (after 07; 07a/b/c independent of each other)
  07a_ablation_modules.sh
  07b_ablation_difficulty.sh
  07c_ablation_classaware.sh

Phase 3 — Evaluation (after Phase 1 + 2)
  08_eval_all.sh → 11_stats_significance.sh

Phase 4 — Additional experiments
  09_exp_e_lowdata.sh        (after 01)
  10_exp_f_efficiency.sh     (anytime)
```

## Metrics

| Metric | Description |
|--------|-------------|
| **mIoU** | Mean Intersection-over-Union (foreground classes) |
| **Dice** | Per-class Dice coefficient |
| **BF1** | Boundary F1 score (2-pixel tolerance) |
| **clDice** | Centreline Dice via skeletonization — measures topological correctness |
| **ConnR** | Connectivity Ratio — fraction of GT components with >= 50% overlap |

All metrics are computed per-class (crack / spalling) and averaged for foreground.

## Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Backbone | SegFormer-B2 | Pretrained on ADE20K-512 |
| Image size | 512 | |
| Batch size | 4 | |
| Learning rate | 6e-5 | AdamW with 5-epoch warmup |
| Epochs | 100 | |
| Difficulty weights | alpha=1.0, beta=0.5, gamma=0.3, delta=0.3 | loss / uncertainty / boundary / sparsity |
| Sampling temperature | tau=0.5 | Softmax sampling sharpness |
| Tversky alpha/beta | 0.3 / 0.7 | FP / FN weights (crack-specific) |
| Curriculum stages | 0–30% / 30–70% / 70–100% | Easy / +Medium / +Hard |

## Requirements

- Python >= 3.9
- PyTorch >= 2.1 with CUDA (or MPS for Apple Silicon)
- See `requirements.txt` for full dependency list

## Citation

```bibtex
@article{chen2025dynamic,
  title={Dynamic Difficulty-Aware Curriculum Learning with Boundary Refinement for Fine-Grained Crack and Spalling Segmentation in Concrete Dams},
  author={Chen, Y.},
  year={2025}
}
```

## License

This project is released for academic research purposes.
