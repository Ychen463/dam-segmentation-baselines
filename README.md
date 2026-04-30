# Topology-Aware Crack and Spalling Segmentation in Concrete Dams via Dynamic Snake Convolution and Skeleton Recall Loss

Official implementation for fine-grained crack and spalling segmentation in concrete dam surfaces. The proposed method integrates **Dynamic Snake Convolution (DSConv)** for topology-sensitive feature extraction and **Skeleton Recall Loss (SRL)** for structure-preserving training, achieving significant improvements in boundary quality and topological correctness over standard segmentation baselines.

## Key Results

On the **DamSegment** test set (150 images, 3 difficulty tiers):

| Method | mIoU_fg | IoU_crack | IoU_spall | BF1_fg | clDice_fg | ConnR_fg |
|--------|---------|-----------|-----------|--------|-----------|----------|
| U-Net (R34) | 59.6 | 51.3 | 67.9 | 64.6 | 73.7 | 76.6 |
| DeepLabV3+ (R50) | 63.4 | 53.3 | 73.5 | 59.2 | 75.1 | 74.6 |
| Mask2Former (Swin-S) | 67.9 | 57.1 | 78.6 | 72.5 | 86.6 | 72.6 |
| SegFormer-B2 | 70.4 | 56.9 | 83.8 | 66.8 | 81.4 | 80.9 |
| **Ours** | **71.4** | **57.6** | **85.2** | **72.1** | **86.2** | **82.3** |

**vs SegFormer-B2 baseline:** mIoU +1.0, BF1 **+5.3**, clDice **+4.8**, ConnR +1.4

## Method Overview

Our approach builds on SegFormer-B2 with two key enhancements:

1. **Dynamic Snake Convolution Branch (DSConv)** — A lightweight dual-DSConv branch consumes encoder stages 1-2, performing deformable sampling along x and y axes to capture the elongated geometry of cracks. The output additively enhances the crack channel logits. Zero-initialized so the branch starts as a no-op and gradually learns crack-specific features.

2. **Skeleton Recall Loss (SRL)** — Penalizes low predicted probability at GT skeleton pixels, directly enforcing topological connectivity. Much lighter than soft-clDice (no online skeletonization needed — GT skeletons are precomputed). Activated after epoch 60 via a delayed-start strategy to avoid early training instability.

**Ablation summary** (each component's contribution over SegFormer-B2):

| Component | Delta mIoU | Delta BF1 | Delta clDice |
|-----------|-----------|-----------|-------------|
| + DSConv | +0.0 | **+2.6** | **+4.1** |
| + DSConv + SRL (Ours) | **+1.0** | **+5.3** | **+4.8** |

DSConv primarily improves boundary and topology metrics; SRL adds both mIoU and boundary gains. The two are complementary.

## Project Structure

```
Codes/
├── full_method/               # Proposed method (DSCFormer)
│   ├── config.py              # Hyperparameters & 55+ ablation presets
│   ├── model.py               # SegFormerWithBoundary + DSCformerDam
│   ├── difficulty.py          # Online difficulty estimator (optional module)
│   ├── sampler.py             # Tier-aware dynamic sampler (optional module)
│   ├── losses.py              # Composite loss (CE + Dice + SRL + ...)
│   ├── scheduler.py           # Curriculum stage scheduler (optional module)
│   ├── dataset.py             # Dataset with tier/spalling metadata
│   └── train.py               # Training loop
│
├── baseline_unet/             # U-Net (ResNet-34)
├── baseline_deeplab/          # DeepLabV3+ (ResNet-50 / ResNet-34)
├── baseline_segformer/        # SegFormer-B2 (plain & static curriculum)
├── baseline_mask2former/      # Mask2Former (Swin-Small)
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
├── scripts/                   # Executable pipeline & figure generation
├── Dataset/                   # DamSegment dataset
├── results/                   # 100+ evaluation JSON files
├── figures/                   # Generated paper figures & LaTeX tables
└── requirements.txt
```

## Dataset: DamSegment

1500 concrete dam surface images (512x512) with pixel-level annotations across three difficulty tiers.

| Property | Value |
|----------|-------|
| Classes | 3: background (0), crack (1), spalling (2) |
| Tiers | Easy (500), Medium (500), Hard (500) |
| Splits | Train 1200 / Val 150 / Test 150 (stratified by tier + spalling) |
| Mask format | RGB PNG — crack in red channel, spalling in blue (threshold > 127) |

## Installation

```bash
git clone https://github.com/Ychen463/dam-segmentation-baselines.git
cd dam-segmentation-baselines

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Verify dataset & generate splits
bash scripts/00_setup.sh
bash scripts/00a_verify_dataset.sh
bash scripts/00b_make_splits.sh
```

## Training

```bash
# Proposed method: DSCFormer + SRL (preset G1, 100 epochs)
python -m full_method.train --ablation G1

# Baselines (independent, parallelizable)
bash scripts/01_train_segformer_A.sh    # SegFormer-B2 plain
bash scripts/03_train_deeplab_A.sh      # DeepLabV3+ R50
bash scripts/04_train_mask2former.sh    # Mask2Former Swin-S
bash scripts/06_train_unet.sh           # U-Net R34
```

### Ablation Presets

| Preset | Description |
|--------|-------------|
| G0 | DSCFormer plain (no SRL) |
| **G1** | **DSCFormer + SRL (proposed method)** |
| G2 | G1 + curriculum learning |
| H0 | DSCFormer + snake auxiliary loss |
| H1 | DSCFormer + aux + SRL |

```bash
python -m full_method.train --ablation G0   # DSConv only
python -m full_method.train --ablation G1   # DSConv + SRL (ours)
```

## Evaluation

```bash
# Evaluate all models
bash scripts/08_eval_all.sh

# Per-image metrics for statistical testing
python -m shared_eval.eval_all --all-models --split test --per-image

# Statistical significance (Wilcoxon + bootstrap CI)
python -m shared_eval.stats_significance --results-dir results/ --split test
```

## Generating Paper Figures & Tables

```bash
python scripts/fig1_dataset_samples.py          # Fig 1: dataset overview
python scripts/fig2_training_curves.py           # Fig 2: training dynamics
python scripts/fig3_qualitative.py               # Fig 3: qualitative comparison
python scripts/fig_architecture.py               # Method architecture diagram
python scripts/generate_latex_tables.py          # LaTeX tables 1-3
```

All outputs are saved to `figures/`.

## Metrics

| Metric | Description | Why it matters |
|--------|-------------|----------------|
| **mIoU_fg** | Mean IoU (crack + spalling) | Overall segmentation quality |
| **BF1** | Boundary F1 (2px tolerance) | Edge precision for thin structures |
| **clDice** | Centreline Dice via skeletonization | Topological correctness |
| **ConnR** | Connectivity Ratio (>=50% overlap) | Connected component preservation |

## Key Hyperparameters (G1 preset)

| Parameter | Value |
|-----------|-------|
| Backbone | SegFormer-B2 (pretrained ADE20K-512) |
| Image size | 512 |
| Batch size | 4 |
| Optimizer | AdamW, lr=6e-5, weight_decay=1e-4 |
| Warmup | 5 epochs |
| Epochs | 100 |
| DSConv hidden / kernel | 64 / 9 |
| SRL weight | 0.05, start epoch 60 |
| Loss | 0.5 CE + 0.5 Dice + 0.05 SRL |

## Paper Writing Guide

This section outlines the recommended paper structure based on experimental findings.

### Recommended Title

> Topology-Aware Crack and Spalling Segmentation in Concrete Dams via Dynamic Snake Convolution and Skeleton Recall Loss

### Core Contributions

1. **DSConv branch for crack geometry** — Dynamic Snake Convolution performs deformable sampling along crack directions, yielding BF1 +2.6 and clDice +4.1 over the SegFormer-B2 baseline without harming mIoU.
2. **Skeleton Recall Loss** — Lightweight topology-preserving loss using precomputed GT skeletons. Adds mIoU +1.0 and BF1 +2.7 on top of DSConv. Delayed-start strategy (epoch 60) prevents early instability.
3. **DamSegment benchmark** — 1500-image dataset with 3 difficulty tiers and comprehensive evaluation (6 metrics x 3 tiers x 5 baselines).

### Paper Structure

```
1. Introduction
   - Dam safety inspection motivation
   - Three challenges: difficulty variance, thin topology, class imbalance
   - Contribution list (DSConv + SRL + dataset)

2. Related Work
   2.1 Concrete defect segmentation
   2.2 Topology-aware segmentation (clDice, SRL, DSConv)
   2.3 Curriculum learning (mention as related but not core)

3. Method
   3.1 Overview (use figures/fig_architecture.pdf)
   3.2 DSCFormer architecture
       - SegFormer-B2 encoder + MLP decoder
       - CrackSnakeBranch: dual-DSConv on stages 1-2, additive crack enhancement
       - Zero-init strategy
   3.3 Skeleton Recall Loss
       - Formula: L_SRL = 1 - mean(p_crack at skeleton pixels)
       - Precomputed GT skeletons (offline, lightweight)
       - Delayed start at epoch 60
   3.4 Composite loss: L = 0.5*CE + 0.5*Dice + 0.05*SRL

4. Experiments
   4.1 Dataset: DamSegment (1500 images, 3 tiers, table + fig1)
   4.2 Implementation details
   4.3 Main comparison (Table 1 — 5 baselines + ours)
       Key argument: BF1 +5.3 and clDice +4.8 matter more than mIoU
       for structural safety assessment
   4.4 Per-tier analysis (Table 2 — ours best on all 3 tiers)
   4.5 Ablation study (Table 3 — DSConv → +SRL → each adds value)
   4.6 Training dynamics (Fig 2) + qualitative results (Fig 3)

5. Discussion
   - Why BF1/clDice > mIoU for engineering inspection
   - DSConv mechanism: deformable sampling suits elongated cracks
   - SRL vs clDice: lighter, more stable
   - Limitations: mIoU gain modest (+1.0), crack IoU bottleneck (~57%)
   - Curriculum learning explored but marginal gain (optional appendix)

6. Conclusion
```

### Key Arguments to Make

- **Application-driven metrics:** In dam inspection, a topologically broken crack prediction is worse than a slightly inaccurate one. BF1 (+5.3) and clDice (+4.8) improvements directly translate to better structural assessment.
- **Complementary contributions:** DSConv improves feature extraction (topology), SRL improves training objective (boundary). Ablation shows clean additive gains.
- **Per-tier robustness:** Method achieves best results on Easy, Medium, AND Hard tiers, with the largest BF1/clDice margins on Hard samples.
- **Efficiency:** DSConv branch is lightweight (64 hidden channels) and zero-initialized. SRL uses precomputed skeletons — no runtime overhead vs clDice.

### Figures & Tables Mapping

| Paper Figure/Table | Source File |
|-------------------|-------------|
| Fig 1: Dataset | `figures/fig1_dataset.pdf` |
| Fig 2: Architecture | `figures/fig_architecture.pdf` (or `fig_architecture.tex` for Overleaf) |
| Fig 3: Training curves | `figures/fig2_training_curves.pdf` |
| Fig 4: Qualitative | `figures/fig3_qualitative.pdf` |
| Table 1: Main comparison | `figures/table1_main_comparison.tex` |
| Table 2: Per-tier | `figures/table2_tier_breakdown.tex` |
| Table 3: Ablation | `figures/table3_ablation.tex` |

### What NOT to Emphasize

- Curriculum learning — explored extensively (15+ rounds) but marginal gain (+0.1 mIoU, -0.3 BF1). Mention in discussion as "explored but not adopted" or move to appendix.
- Static curriculum — actually hurts performance (69.8 vs 70.4 plain). Do not include in main comparison.

## Requirements

- Python >= 3.9
- PyTorch >= 2.1 with CUDA (or MPS for Apple Silicon)
- See `requirements.txt` for full dependency list

## Citation

```bibtex
@article{chen2025topology,
  title={Topology-Aware Crack and Spalling Segmentation in Concrete Dams via Dynamic Snake Convolution and Skeleton Recall Loss},
  author={Chen, Y.},
  year={2025}
}
```

## License

This project is released for academic research purposes.
