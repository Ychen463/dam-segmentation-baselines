# Topology-Aware Crack and Spalling Segmentation in Concrete Dams via Dynamic Snake Convolution and Skeleton Recall Loss

## Core Problem

Concrete dam surface inspection requires accurate segmentation of cracks and spalling defects. Cracks are thin, elongated structures whose topological connectivity is critical for structural safety assessment. Standard segmentation models optimize pixel-level metrics (IoU) but fail to preserve the continuity and boundary precision of crack predictions — a topologically broken crack prediction is worse than a slightly inaccurate one for engineering decisions.

Three key challenges:
1. **Thin topology**: Cracks are 1-5 pixel wide, making them vulnerable to fragmentation in predictions
2. **Class imbalance**: Crack pixels are rare (< 5% of image area); spalling is region-based but variable in shape
3. **Difficulty variance**: Images range from easy (clear, isolated defects) to hard (overlapping defects, poor lighting, complex textures)

## Proposed Method: DSCFormer

We build on SegFormer-B2 (a state-of-the-art hierarchical transformer encoder with MLP decoder) and add two complementary enhancements:

### 1. Dynamic Snake Convolution Branch (DSConv)

A lightweight dual-DSConv branch that consumes encoder stages 1 and 2 (high-resolution features at H/4 and H/8). It performs deformable sampling along x and y axes to capture the elongated, curvilinear geometry of cracks.

- Architecture: Two parallel DSConv paths (x-axis and y-axis), each with kernel_size=9 and hidden_dim=64
- Output: Additively enhances the crack channel logits from the main segmentation head
- Zero-initialization: The branch output starts as zero (no-op) and gradually learns crack-specific features, ensuring stable training

DSConv is inspired by Dynamic Snake Convolution (Qi et al., 2023) which was designed for tubular structure segmentation in medical imaging. We adapt it specifically for crack geometry in structural inspection.

### 2. Skeleton Recall Loss (SRL)

A topology-preserving loss that directly penalizes low predicted probability at ground-truth skeleton pixels:

$$\mathcal{L}_{SRL} = 1 - \frac{1}{|S|} \sum_{p \in S} \hat{y}_{crack}(p)$$

where S is the set of GT skeleton pixels (precomputed offline via morphological skeletonization).

Key design choices:
- **Precomputed GT skeletons**: No online skeletonization needed at training time (unlike soft-clDice), making it much lighter
- **Delayed start at epoch 60**: Prevents early training instability when the model hasn't yet learned basic crack features
- **Weight = 0.05**: Small relative weight in the composite loss to avoid dominating gradient

### 3. Composite Loss

$$\mathcal{L} = 0.5 \cdot \mathcal{L}_{CE} + 0.5 \cdot \mathcal{L}_{Dice} + 0.05 \cdot \mathcal{L}_{SRL}$$

## Why This Works (Intuition)

- **DSConv**: Standard convolutions have fixed receptive fields that don't align with crack geometry. DSConv deforms its sampling grid along the crack direction, improving feature extraction for thin structures. This directly improves boundary precision (BF1) and topological connectivity (clDice).
- **SRL**: Cross-entropy and Dice losses treat all pixels equally — they don't specifically penalize broken connectivity. SRL focuses gradient on the most critical pixels (the skeleton centerline), ensuring the model maintains crack continuity even when overall pixel accuracy is already high.
- **Complementarity**: DSConv improves the feature extraction (architectural), SRL improves the training signal (loss). The ablation shows clean additive gains: DSConv alone gives +2.6 BF1 and +4.1 clDice; adding SRL gives another +2.7 BF1 and +0.7 clDice.

## Contributions

1. A lightweight DSConv branch for crack-specific topology-aware feature extraction, achieving +5.3 BF1 and +4.8 clDice over SegFormer-B2 baseline
2. Skeleton Recall Loss as a simple, efficient alternative to soft-clDice for enforcing topological connectivity
3. DamSegment: A 1500-image benchmark with 3 difficulty tiers and comprehensive evaluation (6 metrics × 3 tiers × 5 baselines)

## Related Work Context

- Dynamic Snake Convolution (Qi et al., ICCV 2023): Proposed for tubular structures in medical images
- clDice (Shit et al., CVPR 2021): Topology loss via differentiable skeletonization — effective but expensive
- SegFormer (Xie et al., NeurIPS 2021): Hierarchical transformer with efficient MLP decoder
- Concrete crack detection: CrackNet, DeepCrack, etc. — mostly binary classification, not multi-class segmentation with topology metrics
