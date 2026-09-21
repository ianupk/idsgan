# IDSGAN — Adversarial Attack Generation against Intrusion Detection Systems

[![Paper](https://img.shields.io/badge/arXiv-1809.02077v5-b31b1b.svg)](https://arxiv.org/abs/1809.02077)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13%2B-ee4c2c.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **College project** — MNNIT Allahabad, Department of Computer Science & Engineering
>
> Implementation of the paper:
> **"IDSGAN: Generative Adversarial Networks for Attack Generation against Intrusion Detection"**
> — Zilong Lin, Yong Shi, Zhi Xue (arXiv:1809.02077v5)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Project Structure](#3-project-structure)
4. [Getting Started](#4-getting-started)
   - [Prerequisites](#41-prerequisites)
   - [Installation](#42-installation)
   - [Download Dataset](#43-download-dataset)
5. [Usage](#5-usage)
   - [Step 1 – Train IDS models](#step-1--train-ids-models)
   - [Step 2 – Train IDSGAN](#step-2--train-idsgan)
   - [Step 3 – Evaluate](#step-3--evaluate)
   - [Step 4 – Baseline comparison](#step-4--baseline-comparison)
6. [Dataset](#6-dataset-nsl-kdd)
7. [Hyperparameters](#7-hyperparameters)
8. [Key Results](#8-key-results)
9. [Module Reference](#9-module-reference)
10. [Bug Fixes & Changes](#10-bug-fixes--changes)
11. [Citation](#11-citation)
12. [License](#12-license)

---

## 1. Overview

IDSGAN is a **Wasserstein GAN (WGAN)** framework that generates adversarial
malicious network traffic records capable of evading black-box intrusion detection
systems (IDS). Unlike white-box attacks, IDSGAN requires **no knowledge of the IDS
internals** — it learns to fool any IDS purely from its binary outputs (detected /
not detected).

### Key ideas

| Feature | Description |
|---|---|
| **Black-box attack** | Queries the IDS as an oracle; no gradient access required |
| **Restricted modification** | A per-attack functional-feature mask preserves features that define attack behaviour, so generated samples remain valid network packets |
| **Dynamic discriminator** | The critic learns the IDS decision boundary by consuming real-time IDS predictions, not static labels |
| **Multi-IDS evaluation** | Tested against 7 different scikit-learn classifiers (SVM, NB, MLP, LR, DT, RF, KNN) |

---

## 2. Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         IDSGAN Framework                         │
│                                                                  │
│  ┌────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │  Original  │───►│  Generator G │───►│  Adversarial         │  │
│  │  Malicious │    │  (5-layer    │    │  Traffic Records     │  │
│  │  Records M │    │   MLP)       │    └──────────┬───────────┘  │
│  └────────────┘    └──────────────┘               │              │
│       │                   ▲                       │              │
│  ┌────▼────┐              │ gradient              ▼              │
│  │ Noise N │              │              ┌─────────────────────┐  │
│  │ (dim=9) │         ┌────┴───────┐     │  Black-Box IDS      │  │
│  └─────────┘         │Discriminator│◄───│  (SVM/NB/MLP/LR/    │  │
│                      │  D (WGAN   │    │   DT/RF/KNN)        │  │
│                      │   critic)  │    └──────────┬───────────┘  │
│                      └────────────┘               │              │
│                                         Predicted labels (0/1)   │
└──────────────────────────────────────────────────────────────────┘
```

The generator receives an original malicious record **M** concatenated with a
uniform noise vector **N** (dim = 9). Its output is post-processed by the
**restricted modification mechanism**: functional features — those that define
the semantics of the attack — are always copied from the original record;
categorical features are never modified; binary features are rounded.

---

## 3. Project Structure

```
lucid-rutherford/
│
├── README.md                    ← this file
├── requirements.txt             ← Python dependencies
├── .gitignore
│
├── download_data.py             ← downloads NSL-KDD into data/
├── train_ids.py                 ← trains 7 black-box IDS classifiers
├── train_idsgan.py              ← trains IDSGAN generator/discriminator
├── evaluate.py                  ← reproduces Table 2, Figures 2–3
├── run_baselines.py             ← reproduces Table 3 (FGSM, JSMA, etc.)
│
├── data/                        ← NSL-KDD dataset (git-ignored; downloaded separately)
│   ├── KDDTrain+.txt
│   └── KDDTest+.txt
│
├── src/
│   ├── __init__.py
│   ├── config.py                ← all hyperparameters and filesystem paths
│   ├── data_preprocessing.py   ← load, encode, normalise, split NSL-KDD
│   ├── feature_utils.py        ← feature definitions, functional masks
│   ├── ids_models.py           ← factory / train / evaluate / save / load IDS
│   ├── generator.py            ← PyTorch Generator (5-layer MLP + restriction)
│   ├── discriminator.py        ← PyTorch Discriminator (WGAN critic)
│   ├── idsgan.py               ← IDSGAN training loop (Algorithm 1)
│   └── metrics.py              ← DR and EIR metric functions
│
└── results/
    ├── models/                  ← saved .pkl (IDS) and .pth (GAN) checkpoints
    └── figures/                 ← generated plots (PNG + PDF)
```

---

## 4. Getting Started

### 4.1 Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.10 or higher |
| pip | 23+ recommended |
| (Optional) CUDA | any version supported by your PyTorch build |

A virtual environment is strongly recommended.

### 4.2 Installation

```bash
# Clone or extract the project
cd idsgan

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate.bat     # Windows

# Install dependencies
pip install -r requirements.txt
```

> **GPU training** — the default `requirements.txt` installs a CPU-only PyTorch
> wheel. For GPU training, replace it with the appropriate CUDA wheel from
> <https://pytorch.org/get-started/locally/> before running the install command.

### 4.3 Download Dataset

```bash
python download_data.py
```

Downloads `KDDTrain+.txt` (~19 MB) and `KDDTest+.txt` (~3 MB) from the
`jmnwong/NSL-KDD-Dataset` GitHub mirror into the `data/` directory.
If the files already exist they are skipped.

---

## 5. Usage

Run each step from the project root directory with the virtual environment
active. All scripts are designed to be run in the order shown.

### Step 1 — Train IDS models

```bash
python train_ids.py
```

Trains SVM, NB, MLP, LR, DT, RF, and KNN on the first half of `KDDTrain+.txt`.
Saves seven `.pkl` checkpoints to `results/models/`. Prints a full
accuracy / precision / recall / F1 table plus per-category detection rates.

### Step 2 — Train IDSGAN

```bash
# Train all 14 combinations (2 attack groups × 7 IDS models)
python train_idsgan.py

# Single attack group against one IDS model
python train_idsgan.py --attack dos --ids_model MLP

# Custom epoch count
python train_idsgan.py --attack u2r_r2l --epochs 50
```

| `--attack` | Description |
|---|---|
| `dos` | DoS attack group only |
| `u2r_r2l` | U2R + R2L attack group only |
| `all` | Both groups (default) |

| `--ids_model` | Choices |
|---|---|
| `SVM NB MLP LR DT RF KNN` | Single model |
| `all` | All 7 models (default) |

Each trained pair is saved as `results/models/idsgan_<GROUP>_<MODEL>.pth`.
A quick validation report (original DR vs adversarial DR) is printed after
each training run.

> **Prerequisite**: Step 1 must be completed before Step 2.

### Step 3 — Evaluate

```bash
python evaluate.py
```

Reproduces the main paper results:

- **Table 2** — original DR, adversarial DR, and EIR for both attack groups
  across all 7 IDS models, under both the standard and robust evaluation
  scenarios.
- **Figure 2** — detection rate comparison bar charts (PNG + PDF).
- **Figure 3** — EIR robustness bar charts (PNG + PDF).

Figures are saved to `results/figures/`.

### Step 4 — Baseline comparison

```bash
python run_baselines.py
```

Reproduces **Table 3** — detection rates under six baseline adversarial methods
(JSMA, FGSM, DeepFool, CW, Unrestricted Static GAN, Static GAN) compared with
IDSGAN, all targeting the MLP IDS. DeepFool and CW operate on a 500-sample
subset for speed.

---

## 6. Dataset: NSL-KDD

NSL-KDD is the standard benchmark for network intrusion detection research.
It is a refined version of the 1999 KDD Cup dataset.

| Split | Records | Source |
|---|---|---|
| `KDDTrain+.txt` | 125,973 | Used for IDS training (first half) and GAN training (second half) |
| `KDDTest+.txt` | 22,544 | Held-out evaluation |

### Feature sets (41 raw features → 122 after one-hot encoding)

| Set | Features | Count |
|---|---|---|
| Intrinsic | Protocol, service, flag, bytes, land, fragments, urgent | 9 |
| Content | Login attempts, shells, file operations, guest/host login | 13 |
| Time-based traffic | Connection rate statistics over a 2-second window | 9 |
| Host-based traffic | Connection statistics over the last 100 connections | 10 |

### Attack categories

| Category | Examples |
|---|---|
| **DoS** | neptune, smurf, pod, teardrop, back, land |
| **Probe** | ipsweep, nmap, portsweep, satan |
| **U2R** | buffer_overflow, rootkit, perl, loadmodule |
| **R2L** | ftp_write, guess_passwd, imap, multihop, phf |

### Functional features (Table 1 of the paper)

The **restricted modification mechanism** only allows the GAN to modify
non-functional features — those that do not define the attack's behaviour.

| Attack | Intrinsic | Content | Time-based | Host-based |
|---|:---:|:---:|:---:|:---:|
| DoS | 🔒 | ✏️ | ✏️ | ✏️ |
| Probe | 🔒 | ✏️ | 🔒 | 🔒 |
| U2R | 🔒 | 🔒 | ✏️ | ✏️ |
| R2L | 🔒 | 🔒 | ✏️ | ✏️ |

🔒 = functional (preserved)   ✏️ = modifiable

---

## 7. Hyperparameters

All hyperparameters are centralised in `src/config.py`.

| Parameter | Value | Description |
|---|---|---|
| `BATCH_SIZE` | 64 | Mini-batch size |
| `EPOCHS` | 100 | Training epochs |
| `LR_GENERATOR` | 0.0001 | Generator learning rate |
| `LR_DISCRIMINATOR` | 0.0001 | Discriminator learning rate |
| `CLIP_VALUE` | 0.01 | WGAN weight clipping bound |
| `NOISE_DIM` | 9 | Noise vector dimension |
| `N_CRITIC` | 5 | Discriminator steps per generator step |
| `G_HIDDEN_DIM` | 256 | Generator hidden layer width |
| `D_HIDDEN_DIM` | 256 | Discriminator hidden layer width |
| `BINARY_THRESHOLD` | 0.5 | Rounding threshold for binary features |
| `RANDOM_SEED` | 42 | Reproducibility seed |

---

## 8. Key Results

### Table 2 — IDSGAN Detection Rates (from the paper)

| Attack | Metric | SVM | NB | MLP | LR | DT | RF | KNN |
|---|---|---|---|---|---|---|---|---|
| DoS | Original DR | ~82% | ~85% | ~83% | ~80% | ~75% | ~73% | ~77% |
| DoS | Adversarial DR | <1% | <1% | <1% | <1% | <1% | <1% | <1% |
| DoS | EIR | >99% | >99% | >99% | >99% | >99% | >99% | >99% |
| U2R & R2L | Original DR | ~1% | ~6% | ~5% | ~1% | ~13% | ~2% | ~6% |
| U2R & R2L | Adversarial DR | ~0% | ~0% | ~0% | ~0% | ~0% | ~0% | ~0% |

> Exact reproduced values depend on the random seed and hardware. Set
> `RANDOM_SEED = 42` (default) for the closest match.

### Table 3 — Baseline Comparison

IDSGAN consistently achieves the lowest adversarial detection rate compared
with FGSM, JSMA, DeepFool, CW, and both static GAN variants.

---

## 9. Module Reference

### `src/config.py`
Central configuration. Edit this file to change hyperparameters or paths
without touching any other source file.

### `src/feature_utils.py`
- `FEATURE_NAMES` — ordered list of all 41 NSL-KDD feature names.
- `CATEGORICAL_FEATURES` — protocol type (3 values), service (70 values), flag (11 values).
- `FUNCTIONAL_FEATURES_RAW` / `MODIFIABLE_FEATURES_RAW` — per-attack raw-index sets.
- `get_functional_feature_mask(attack)` → boolean array (122,).
- `get_modifiable_feature_mask(attack)` → complement of the above.
- `get_binary_feature_indices_encoded()` → indices of binary features in encoded space.
- `get_added_unmodified_features(attack, fraction)` → robustness mask (Section 4.3).

### `src/data_preprocessing.py`
- `load_raw_data(filepath)` — loads a KDD file, maps attack labels to categories.
- `preprocess_features(train_df, test_df)` — one-hot encodes categoricals, min-max scales numerics. Returns arrays + fitted scaler + encoder.
- `prepare_datasets()` — full pipeline; returns a dict with `ids_train_X/y`, `gan_normal_X`, `gan_attack_X`, `test_X/y`, etc.
- `save_preprocessed(data_dict, path)` / `load_preprocessed(path)` — disk caching via `.npz` + joblib.

### `src/generator.py` — `Generator(nn.Module)`
- `forward(original_records, noise)` → raw output (122,).
- `generate(original_records, noise, functional_mask)` → clamps output, applies restricted modification mechanism, rounds binary features using a straight-through estimator.

### `src/discriminator.py` — `Discriminator(nn.Module)`
Three-layer MLP with LeakyReLU activations. No sigmoid on the output — outputs raw WGAN scores.

### `src/idsgan.py` — `IDSGAN`
- `__init__(attack_category, ids_model)` — instantiates generator, discriminator, optimisers (RMSProp), and functional mask.
- `train(normal_data, attack_data, epochs, batch_size)` → loss history.
- `generate_adversarial(attack_data)` → `np.ndarray` of adversarial records.
- `save(path)` / `load(path)` — checkpoint I/O.

### `src/ids_models.py`
- `create_ids_model(name)` — factory returning a calibrated SVM, GaussianNB, MLPClassifier, LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, or KNeighborsClassifier.
- `train_ids_model`, `evaluate_ids`, `predict_ids`, `save_ids_model`, `load_ids_model`.

### `src/metrics.py`
- `detection_rate(y_true, y_pred)` — fraction of true attacks correctly detected.
- `evasion_increase_rate(original_dr, adversarial_dr)` — `1 − adv_dr / orig_dr`.
- `compute_all_metrics(original_labels, original_preds, adversarial_preds)`.

---

## 10. Bug Fixes & Changes

| File | Issue | Fix |
|---|---|---|
| `download_data.py` | Unnecessary `sys.path` manipulation made the script fragile when run from a different working directory | Removed; `src.config` is importable when running from the project root |
| `train_idsgan.py` | CLI argument `--attack u2r_r2l` was mapped to `["U2R_R2L"]` which matched the `ATTACK_GROUPS` key — correct — but the printed help text was inconsistent with internal naming | Consolidated via `_ATTACK_CLI_MAP`; added explicit `_PRIMARY_ATTACK` dict to avoid the implicit `"U2R"` assumption buried in the training loop |
| `evaluate.py` | `from src.generator import Generator` was imported but never used | Removed unused import |
| `evaluate.py` | `IDSGAN(ids_model=None)` — safe for inference, but no guard on missing checkpoint | Added `os.path.exists` check with a clear warning before falling back to an untrained generator |
| `run_baselines.py` | `deepfool_attack`: after `x_i = (...).detach(); x_i.requires_grad = True`, the gradient from the *previous* iteration could contaminate the new `x_i.grad` if the old grad buffer was not zeroed before `backward()`. With the reassignment the old tensor is discarded, but the new leaf has `grad = None` until `backward()` is called — so `x_i.grad.zero_()` was unconditionally called at the top of the loop on a potentially `None` grad | Fixed by removing the premature zero call and relying on the fresh leaf tensor; `backward()` now populates `x_i.grad` correctly each iteration |
| `run_baselines.py` | `deepfool_attack`: used `retain_graph=True` on a graph that is rebuilt from scratch each iteration (not needed and wastes memory) | Removed `retain_graph=True` |
| `requirements.txt` | `joblib` was used in `data_preprocessing.py` and `ids_models.py` but absent from `requirements.txt` | Added `joblib>=1.2.0` |
| `README.md` | Listed `notebooks/results_visualization.ipynb` in the project tree but this directory does not exist | Removed the non-existent path from the tree |
| `.gitignore` | Missing `__pycache__/` pattern under `src/` (it was listed but the `src/__pycache__` directory was committed in the zip) | Ensured the pattern is at the top level so all subdirectories match |

---

## 11. Citation

If you use this code for academic work, please cite the original paper:

```bibtex
@article{lin2022idsgan,
  title   = {IDSGAN: Generative Adversarial Networks for Attack Generation
             against Intrusion Detection},
  author  = {Lin, Zilong and Shi, Yong and Xue, Zhi},
  journal = {arXiv preprint arXiv:1809.02077},
  year    = {2022}
}
```

---

## 12. License

This project is for educational and research purposes only.
Use responsibly and in accordance with your institution's ethics guidelines.
