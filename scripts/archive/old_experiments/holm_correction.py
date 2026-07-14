"""Holm-Bonferroni correction for the 10 Wilcoxon p-values (Issue 11).

Reads the raw p-values from the paper and computes adjusted p-values.
"""

# Raw p-values from paper (row b vs row c, paired Wilcoxon signed-rank)
raw_pvalues = {
    "IoU_cr":     0.008,
    "Dice_cr":    0.007,
    "BF1_cr":     0.046,
    "clDice_cr":  0.001,   # reported as <0.001, use 0.001 conservatively
    "ConnR_cr":   0.040,   # significant negative
    "IoU_sp":     0.016,
    "Dice_sp":    0.016,
    "BF1_sp":     0.10,    # non-significant (approximate)
    "clDice_sp":  0.10,    # non-significant (approximate)
    "ConnR_sp":   0.10,    # non-significant (approximate)
}

# Holm-Bonferroni correction
n = len(raw_pvalues)
sorted_items = sorted(raw_pvalues.items(), key=lambda x: x[1])

print(f"Holm-Bonferroni correction (n={n} comparisons, alpha=0.05)")
print(f"{'Metric':<12} {'Raw p':>8} {'Threshold':>10} {'Adj p':>8} {'Sig?':>5}")
print("-" * 48)

adjusted = {}
max_adj = 0.0
for i, (metric, p) in enumerate(sorted_items):
    rank = i + 1
    threshold = 0.05 / (n - i)
    adj_p = min(p * (n - i), 1.0)
    # Enforce monotonicity: adjusted p must be >= previous
    adj_p = max(adj_p, max_adj)
    max_adj = adj_p
    sig = "YES" if adj_p < 0.05 else "no"
    adjusted[metric] = adj_p
    print(f"{metric:<12} {p:>8.4f} {threshold:>10.4f} {adj_p:>8.4f} {sig:>5}")

print()
print("Summary:")
sig_metrics = [m for m, p in adjusted.items() if p < 0.05]
nonsig_metrics = [m for m, p in adjusted.items() if p >= 0.05]
print(f"  Significant after Holm correction: {', '.join(sig_metrics)}")
print(f"  Non-significant after correction:  {', '.join(nonsig_metrics)}")
