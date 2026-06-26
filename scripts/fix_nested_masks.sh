#!/bin/bash
# Fix nested Mask/Mask directories created by scp
# Usage: bash scripts/fix_nested_masks.sh

set -e

BASE="/workspace/dam-segmentation-baselines/Dataset/DamSegment/Damage Segmentaion"

for tier in Easy Medium Hard; do
    nested="$BASE/$tier/Labels/Mask/Mask"
    target="$BASE/$tier/Labels/Mask"
    if [ -d "$nested" ]; then
        count=$(find "$nested" -type f -name "*.png" | wc -l)
        echo "$tier: found $count files in nested Mask/Mask/, moving up..."
        find "$nested" -type f -name "*.png" -exec cp -n {} "$target/" \;
        rm -rf "$nested"
        echo "$tier: done"
    else
        echo "$tier: no nesting found"
    fi
done

total=$(find "$BASE" -path "*/Labels/Mask/*" -type f | wc -l)
echo ""
echo "Total mask files: $total (expected: 1500)"

if [ "$total" -eq 1500 ]; then
    echo "OK - dataset is complete"
else
    echo "WARNING - expected 1500, got $total"
fi
