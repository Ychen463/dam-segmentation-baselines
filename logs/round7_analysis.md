# Round 7 Results Analysis: Softmax Sampling vs Loss Reweight

## Context

Round 7 tested whether replacing **dynamic loss reweighting** (Round 6) with **softmax sampling** could improve performance. The core hypothesis: instead of reweighting the loss by difficulty, use difficulty scores to bias the sampling probability via softmax.

---

## Test Set Results Summary

### All Experiments (sorted by mIoU_fg)

| Experiment | Strategy | mIoU_fg | IoU_crack | IoU_spalling | BF1_fg | clDice_fg | ConnR_fg |
|---|---|---|---|---|---|---|---|
| **F1** (R6) | loss reweight + C2 + clDice | **0.7091** | 0.5723 | 0.8460 | 0.7169 | 0.8613 | 0.7911 |
| P1 (baseline) | plain + clDice | 0.7081 | **0.5731** | 0.8432 | 0.7172 | **0.8698** | 0.7832 |
| C2 (baseline) | competence curriculum | 0.7051 | 0.5700 | 0.8402 | **0.7176** | 0.8582 | 0.8219 |
| **D0v2** (R7) | softmax sampling | 0.7038 | 0.5643 | 0.8433 | 0.6886 | 0.8327 | 0.7655 |
| P0 (baseline) | plain segformer | 0.7030 | 0.5688 | 0.8372 | 0.6858 | 0.8272 | 0.7696 |
| **D1v2** (R7) | softmax sampling + clDice | 0.7022 | 0.5663 | 0.8382 | 0.7074 | 0.8461 | 0.7094 |
| D1 (R6) | loss reweight + clDice | 0.7004 | 0.5630 | 0.8379 | 0.6940 | 0.8593 | **0.8236** |
| **F1v2** (R7) | softmax sampling + C2 + clDice | 0.6987 | 0.5456 | **0.8517** | 0.6846 | 0.8545 | 0.7977 |
| D0 (R6) | loss reweight only | 0.6833 | 0.5536 | 0.8129 | 0.6759 | 0.8412 | 0.7758 |

---

## Round 6 vs Round 7 Head-to-Head

### D0 (difficulty only) → D0v2

| Metric | D0 (reweight) | D0v2 (sampling) | Delta |
|---|---|---|---|
| mIoU_fg | 0.6833 | **0.7038** | **+2.05%** |
| IoU_crack | 0.5536 | **0.5643** | **+1.06%** |
| IoU_spalling | 0.8129 | **0.8433** | **+3.04%** |
| BF1_fg | 0.6759 | **0.6886** | **+1.27%** |
| clDice_fg | **0.8412** | 0.8327 | -0.85% |
| ConnR_fg | **0.7758** | 0.7655 | **-1.03%** |

**Verdict: D0v2 wins on IoU/BF1, D0 wins on connectivity metrics. Softmax sampling is a clear upgrade for D0.**

### D1 (difficulty + clDice) → D1v2

| Metric | D1 (reweight) | D1v2 (sampling) | Delta |
|---|---|---|---|
| mIoU_fg | 0.7004 | **0.7022** | +0.18% |
| IoU_crack | 0.5630 | **0.5663** | +0.33% |
| IoU_spalling | 0.8379 | **0.8382** | +0.03% |
| BF1_fg | 0.6940 | **0.7074** | **+1.34%** |
| clDice_fg | **0.8593** | 0.8461 | **-1.32%** |
| ConnR_fg | **0.8236** | 0.7094 | **-11.42%** |

**Verdict: D1v2 marginally better on IoU/BF1, but ConnR_fg drops sharply (-11.4%). Softmax sampling hurts connectivity when combined with clDice.**

### F1 (full method) → F1v2

| Metric | F1 (reweight) | F1v2 (sampling) | Delta |
|---|---|---|---|
| mIoU_fg | **0.7091** | 0.6987 | **-1.05%** |
| IoU_crack | **0.5723** | 0.5456 | **-2.67%** |
| IoU_spalling | 0.8460 | **0.8517** | +0.58% |
| BF1_fg | **0.7169** | 0.6846 | **-3.22%** |
| clDice_fg | **0.8613** | 0.8545 | -0.68% |
| ConnR_fg | 0.7911 | **0.7977** | +0.66% |

**Verdict: F1 (Round 6) is clearly better. F1v2 drops on crack IoU and boundary F1. Softmax sampling hurts the full pipeline.**

---

## Training Convergence

| Experiment | Best Epoch | Val mIoU_fg (best) | Test mIoU_fg |
|---|---|---|---|
| F1 | 80 | 0.6758 | 0.7091 |
| P1 | 55 | 0.6848 | 0.7081 |
| C2 | 80 | 0.6949 | 0.7051 |
| D0v2 | 80 | 0.6937 | 0.7038 |
| P0 | 91 | 0.6786 | 0.7030 |
| D1v2 | 70 | 0.6819 | 0.7022 |
| D1 | 35 | 0.6874 | 0.7004 |
| F1v2 | 35 | 0.6802 | 0.6987 |
| D0 | 60 | 0.6815 | 0.6833 |

F1v2 converged at epoch 35 (vs F1 at epoch 80), suggesting instability or premature convergence with softmax sampling in the full method.

---

## Key Observations

1. **F1 (Round 6) remains the best overall model** with mIoU_fg = 0.7091, the highest among all experiments.

2. **Softmax sampling helps the simple case (D0→D0v2)** but **hurts the full pipeline (F1→F1v2)**. The more components are stacked, the less benefit sampling provides.

3. **Crack IoU is the bottleneck across all experiments** (~0.54-0.57), while spalling IoU is consistently strong (~0.84-0.85).

4. **F1v2 converged too early (epoch 35)** compared to F1 (epoch 80), suggesting instability or premature convergence with softmax sampling in the full method.

5. **Baselines are surprisingly competitive**: P1 (plain + clDice, mIoU_fg=0.7081) is only 0.1% behind F1, raising questions about whether the difficulty mechanism adds meaningful value over clDice alone.

6. **ConnR_fg is volatile**: D1v2 has a large ConnR drop vs D1 (-11.4%), suggesting softmax sampling + clDice combination may fragment predictions.

---

## Recommendation for Next Steps

The current best model is **F1 (Round 6, loss reweight)**. Options to consider:

1. **Stick with F1 as the final model** — it leads on mIoU_fg and BF1_fg.
2. **Investigate why P1 (plain + clDice) nearly matches F1** — if the difficulty mechanism doesn't add much over clDice, the paper's contribution narrative needs adjustment.
3. **Try a hybrid**: softmax sampling for D0-level, loss reweight for full pipeline.
4. **Address crack IoU**: this is the weakest link (~0.57 vs spalling ~0.85). Targeted improvements here would have the biggest impact.
