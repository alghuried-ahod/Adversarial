#!/usr/bin/env bash
# run_all.sh — End-to-end reproduction script
# Paper: "Adversarial Evasion of Smart-Contract Phishing Classifiers:
#         Attack Surface Analysis and a Certified Defense Framework"
# Applied Sciences, MDPI, 2026.
#
# ============================================================
# PREREQUISITES — download from Zenodo before running
# ============================================================
# Zenodo DOI: [INSERT FINAL DOI HERE BEFORE SUBMISSION]
#
# Required files not included in this package (too large):
#   data/bytecode/           — 22,000 EVM bytecode .hex files
#   data/features_144d.npy   — pre-extracted 144-d feature matrix
#                              (skip Step 2 if already downloaded)
#   classifiers/clf1_rf/model.pkl
#   classifiers/clf2_adaboost/model.pkl
#   classifiers/clf3_gcn/model_best.pt
#   classifiers/clf4_evullm/model_best.pt
#   attacks/a1_opcode_sub/attack.py   (and a2, a3, a4)
#   defense/ibp_certification/certify.py
#   defense/symbolic_execution/z3_approver.py
#   logs/per_seed/                     — five-seed training logs
#   logs/per_sample_predictions/       — per-sample prediction CSVs
#   symbolic_traces/delayed_trigger_427/  — 427 JSON Z3 trace files
#
# Contact: favadhassanjaskani@gmail.com for pre-publication access.
# ============================================================

set -e

# ---- Step 0: Check prerequisites ----
echo "=== Step 0: Checking prerequisites ==="
MISSING=0
for f in data/features_144d.npy data/class_labels.csv \
          data/splits/test_ids.txt \
          classifiers/clf1_rf/model.pkl \
          classifiers/clf2_adaboost/model.pkl \
          classifiers/clf3_gcn/model_best.pt \
          classifiers/clf4_evullm/model_best.pt \
          classifiers/extract_features.py \
          defense/ibp_certification/certify.py \
          defense/symbolic_execution/z3_approver.py; do
    if [ ! -f "$f" ]; then
        echo "  MISSING: $f"
        MISSING=1
    fi
done
if [ "$MISSING" -eq 1 ]; then
    echo ""
    echo "ERROR: Required files missing."
    echo "Download from Zenodo: [INSERT FINAL DOI HERE]"
    exit 1
fi
echo "  All required files present."

# ---- Step 1: Install dependencies ----
echo ""
echo "=== Step 1: Install dependencies ==="
pip install -r requirements.txt

# ---- Step 2: Extract 144-d opcode features ----
echo ""
echo "=== Step 2: Extract 144-d opcode features ==="
if [ -f "data/features_144d.npy" ]; then
    echo "  Skipping: data/features_144d.npy already present."
else
    python classifiers/extract_features.py \
        --bytecode_dir data/bytecode/ \
        --output       data/features_144d.npy
fi

# ---- Step 3: Reproduce Table 10 ----
echo ""
echo "=== Step 3: Reproduce Table 10 ==="
python evaluation/reproduce_table10.py \
    --features data/features_144d.npy \
    --labels   data/class_labels.csv \
    --splits   data/splits/ \
    --seeds    42 123 456 789 1011

# ---- Step 4: Reproduce Section 7.4 Integrated Defense ----
echo ""
echo "=== Step 4: Reproduce Section 7.4 Integrated Defense ==="
python evaluation/reproduce_section74.py \
    --features  data/features_144d.npy \
    --labels    data/class_labels.csv \
    --test_ids  data/splits/test_ids.txt

# ---- Step 5: IBP certified robustness ----
echo ""
echo "=== Step 5: IBP Certified Robustness (CLF-3, epsilon=0.01) ==="
python defense/ibp_certification/certify.py \
    --model    classifiers/clf3_gcn/model_best.pt \
    --features data/features_144d.npy \
    --labels   data/class_labels.csv \
    --test_ids data/splits/test_ids.txt \
    --epsilon  0.01 \
    --seeds    42 123 456 789 1011
# Expected: rho = 85.2% ± 1.3%

echo ""
echo "=== Done ==="
echo "All steps completed. Compare output against paper Table 10 and Section 7.4."
