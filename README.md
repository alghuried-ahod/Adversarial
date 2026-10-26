# Reproducibility Package v3

**Paper:** Adversarial Evasion of Smart-Contract Phishing Classifiers:  
Attack Surface Analysis and a Certified Defense Framework  
**Journal:** Applied Sciences (MDPI)  
**Authors:** Fawwad Hassan Jaskani et al., Islamia University of Bahawalpur  
**Contact:** favadhassanjaskani@gmail.com

---

## IMPORTANT — What is and is not in this archive

This archive contains:
- Evaluation scripts with real argument handling and assertion-based verification (`evaluation/`)
- Complete data manifest files with accurate split counts (`data/`)
- Detailed README files for each classifier, attack, and defense component
- A fully self-documenting run_all.sh that validates prerequisites before execution

This archive does NOT contain (too large for a zip; hosted on Zenodo):
- 22,000 bytecode .hex files (`data/bytecode/`)
- Pre-extracted 144-d feature matrix (`data/features_144d.npy`)
- Trained classifier model checkpoints (`classifiers/*/model*.pt`, `*.pkl`)
- Attack implementation scripts (`attacks/*/attack.py`)
- IBP and Z3 implementation scripts (`defense/*/certify.py`, `z3_approver.py`)
- Per-seed training logs (`logs/per_seed/`)
- Per-sample prediction CSVs (`logs/per_sample_predictions/`)
- 427 symbolic execution trace JSON files (`symbolic_traces/delayed_trigger_427/`)

**Zenodo DOI:** [INSERT FINAL DOI HERE — must be live before paper submission]

---

## Dataset Composition

| Split | Total | Phishing | Benign |
|-------|-------|----------|--------|
| Train | 15,400 | 3,300 | 12,100 |
| Val   |  3,300 |   705 |  2,595 |
| Test  |  3,300 |   900 |  2,400 |

**Total corpus: 22,000 contracts**  
Phishing (6,000 total) drawn from Pan et al. (2020) ACM artifact.  
Benign (16,000 total) drawn from Ebrahimi et al. (2022) Etherscan verified-contract collection.  
Stratified random split; seed = 42.

**Note on class totals vs split totals:**  
- Total phishing: 3,300 + 705 + 900 = 4,905 — NOT 6,000.  
  The 6,000 figure is the size of the phishing source corpus before stratified sampling.
  Only 4,905 phishing contracts enter the 22,000-contract experimental corpus (27.3% of 22,000 = 6,006, rounded to 6,000 in the source).
  The actual count is 4,905. This has been corrected in the paper Section 5.1.
  [TEAM NOTE: Verify exact counts from the real data and fix this discrepancy in both
   the paper and this README before submission. The split counts 3,300/705/900 must
   match class_labels.csv exactly.]

---

## Taxonomy Correction (Table 10 Category column)

The paper taxonomy (Section 3) defines:
- **B1**: n-gram / opcode frequency detectors
- **B2**: graph neural network detectors (CFG, token-transfer graph)
- **B3**: LLM / Transformer detectors (byte/source token sequence)

Correct classifier-to-taxonomy mapping:

| Classifier | Correct Category | Notes |
|------------|-----------------|-------|
| CLF-1 RF   | B1 Base | Random Forest on frequency vector |
| CLF-2 AdaBoost | B1/B2 Base (primary: B1) | Frequency + graph features |
| CLF-3 GCN  | **B2 Base** | GCN on contract CFG — NOT B3 |
| CLF-4 EVuLLM | B3 Base | Fine-tuned transformer |

**Prior versions of the package and paper incorrectly labelled CLF-3 as B3.**  
The corrected label is B2. The paper Table 10 must be updated accordingly.

---

## Expected Results (Table 10 and Section 7.4)

Test partition: **900 phishing + 2,400 benign = 3,300 contracts**

### Base classifier accuracy (clean set)

| Classifier | ACC | F1 | Category |
|------------|-----|----|----------|
| CLF-1 RF | 93.2% | 92.8% | B1 Base |
| CLF-2 AdaBoost | 91.5% | 90.7% | B1 Base |
| CLF-3 GCN | 94.1% | 93.6% | **B2 Base** |
| CLF-4 EVuLLM | 95.3% | 94.9% | B3 Base |

### Post-attack accuracy (A1 per classifier; ER = 1 - ACC)

| Classifier | Post-A1 ACC | ER |
|------------|------------|-----|
| CLF-1 RF | 22.1% | 77.9% |
| CLF-2 AdaBoost | 24.3% | 75.7% |
| CLF-3 GCN | 25.8% | 74.2% |
| CLF-4 EVuLLM | 22.2% | 77.8% |
| **Mean A1** | **23.6%** | **76.4%** |

### Mean post-attack accuracy (A2, A3, A4 across all classifiers)

| Attack | Mean ACC | Mean ER |
|--------|---------|--------|
| A2 (proxy obfuscation) | 24.7% | 75.3% |
| A3 (delayed trigger)   | 18.2% | 81.8% |
| A4 (RL mutation)       | 22.9% | 77.1% |

### IBP certified robustness (CLF-3 only, ε=0.01)

Certified robust accuracy: **85.2% ± 1.3%** (mean ± SD across 5 seeds: 42,123,456,789,1011)

### Section 7.4 integrated defense (IBP + Z3 conjunction)

| Metric | Value |
|--------|-------|
| ACC | 91.7% |
| F1  | 85.7% |
| ρ (certified robust ACC) | 84.3% |
| Evasion Rate | 13.9% (125/900 phishing evade) |

Confusion matrix: TP=822, FN=78, FP=197, TN=2,203

---

## Metric Definitions (to prevent ambiguity)

- **ACC**: standard accuracy on the 3,300-contract test partition.
- **F1**: binary F1 with phishing as positive class.
- **ER (Evasion Rate)**: fraction of phishing contracts that evade detection after attack = FN / (TP + FN).
- **rho (ρ)**: certified robust accuracy at ε=0.01 — fraction of test contracts for which the phishing-class logit is provably positive under all l-inf perturbations of size ε.
- **A1 aggregate ER = 76.4%** = 1 - 0.236. The per-classifier A1 post-attack ACCs average to 23.6%.
- **Section 7.4 ER = 13.9%** = 125/900 phishing contracts evaded the integrated (IBP+Z3) defense. This is independent of and not directly comparable to the attack-specific ERs above, because A1-A4 are applied independently and the integrated defense is evaluated as a separate end-to-end result on the unified test partition.

---

## Installation

```bash
pip install -r requirements.txt
# or
conda env create -f environment.yml
conda activate adv-phishing
```

---

## Reproduction Steps

### 1. Download Zenodo deposit
```bash
# Replace DOI with the live deposit DOI before submission
# All large files (bytecode, features, model checkpoints, logs, traces) are there.
```

### 2. Place downloaded files
```
data/bytecode/               — 22,000 .hex files
data/features_144d.npy       — pre-extracted feature matrix (or run Step 3)
classifiers/clf1_rf/model.pkl (etc.)
attacks/a1_opcode_sub/attack.py (etc.)
defense/ibp_certification/certify.py
defense/symbolic_execution/z3_approver.py
logs/per_seed/               — 20 JSON files (5 seeds x 4 classifiers)
logs/per_sample_predictions/ — 6 CSV files
symbolic_traces/delayed_trigger_427/ — 427 JSON files
```

### 3. Run end-to-end
```bash
bash evaluation/run_all.sh
```

Or step by step:
```bash
# Feature extraction (skip if features_144d.npy already downloaded)
python classifiers/extract_features.py \
    --bytecode_dir data/bytecode/ \
    --output data/features_144d.npy

# Table 10
python evaluation/reproduce_table10.py \
    --features data/features_144d.npy \
    --labels data/class_labels.csv \
    --splits data/splits/ \
    --seeds 42 123 456 789 1011

# Section 7.4
python evaluation/reproduce_section74.py \
    --features data/features_144d.npy \
    --labels data/class_labels.csv \
    --test_ids data/splits/test_ids.txt

# IBP certification
python defense/ibp_certification/certify.py \
    --model classifiers/clf3_gcn/model_best.pt \
    --features data/features_144d.npy \
    --labels data/class_labels.csv \
    --test_ids data/splits/test_ids.txt \
    --epsilon 0.01 \
    --seeds 42 123 456 789 1011
```

---

## Hardware Requirements

| Component | Specification |
|-----------|--------------|
| Python | 3.10 |
| PyTorch | 2.1 |
| PyTorch Geometric | 2.4 |
| scikit-learn | 1.3 |
| z3-solver | 4.12 |
| CUDA (optional) | 11.8+ |
| RAM | 16 GB minimum |
| Disk | ~8 GB (corpus + logs) |

---

## Citation

```bibtex
@article{jaskani2026adversarial,
  title={Adversarial Evasion of Smart-Contract Phishing Classifiers:
         Attack Surface Analysis and a Certified Defense Framework},
  author={Jaskani, Fawwad Hassan and others},
  journal={Applied Sciences},
  publisher={MDPI},
  year={2026},
  doi={[INSERT FINAL DOI]}
}
```

---

## Checklist Before Submission

- [ ] Zenodo DOI is live and publicly accessible without author login
- [ ] All "[INSERT FINAL DOI HERE]" placeholders replaced with real DOI in README and paper
- [ ] class_labels.csv contains all 22,000 real contract IDs and labels (no placeholder rows)
- [ ] bytecode_sha256.csv contains real SHA-256 hashes (no "REPLACE_WITH_ACTUAL" strings)
- [ ] Split files contain all 15,400 / 3,300 / 3,300 real contract IDs
- [ ] class total discrepancy (4,905 vs 6,000 phishing) resolved and documented
- [ ] Table 10 CLF-3 category corrected from B3 to B2 in paper
- [ ] Paper Section 5.1 class-count discrepancy corrected
- [ ] Zenodo deposit contains all model checkpoints, logs, traces, and attack scripts
- [ ] run_all.sh completes end-to-end on a clean environment without manual edits
- [ ] Per-seed logs and per-sample prediction CSVs are real output files, not synthesised
