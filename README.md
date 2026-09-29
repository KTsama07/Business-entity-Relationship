# Business Entity Resolution — Amazon ML Challenge 2026

A machine learning based **Business Entity Resolution** system developed for the **Amazon ML Challenge 2026**.

The objective is to identify records across multiple business-data sources that refer to the same real-world business entity. The project explores multiple candidate-generation and machine-learning approaches, including semantic nearest-neighbor search, exact/token-based blocking, XGBoost, and LightGBM.

---

## Overview

Business Entity Resolution (BER) is the task of determining whether records from different data sources represent the same underlying business.

For this challenge:

* **Source 1** acts as the reference entity set.
* **Source 2** and **Source 3** contain potentially matching business records.
* Business names and addresses may contain:

  * Typos
  * Abbreviations
  * Transliteration differences
  * Missing information
  * Different word ordering
  * Address-format variations
  * Legal-name variations

The system therefore combines deterministic matching signals, text similarity, semantic similarity, and supervised machine-learning models.

The challenge evaluates predictions using the **macro-averaged F₀.₅ score**, which gives greater importance to precision than recall. This makes avoiding incorrect entity merges particularly important.

### High-Level Pipeline

```text
                    ┌─────────────────────┐
                    │   Source 1 Records  │
                    │   Reference Set     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Normalization /     │
                    │ Preprocessing       │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
        Exact / Token     Semantic NN      TF-IDF Blocking
          Blocking         Blocking          / Hash Joins
              │                │                │
              └────────────────┼────────────────┘
                               ▼
                    ┌─────────────────────┐
                    │ Candidate Pairs     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Pairwise Features   │
                    │                     │
                    │ Name similarity     │
                    │ Address similarity  │
                    │ Country match       │
                    │ Length differences  │
                    └──────────┬──────────┘
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
              XGBoost Model         LightGBM Model
                    │                     │
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │ Threshold           │
                    │ Calibration         │
                    │ using F₀.₅          │
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │ matching_results    │
                    │ candidate_pairs     │
                    └─────────────────────┘
```

---

## Repository Structure

```text
Business-entity-Relationship/
│
├── NN/
│   ├── Untitled.ipynb
│   └── Untitled1.ipynb
│
├── method1_xgb/
│   ├── model_pipeline.ipynb
│   └── threshold_config.json
│
├── method4_lgbm/
│   ├── LIGHTBGM/
│   │   ├── kartik_Lgbm.ipynb
│   │   └── pipeline_config.json
│   │
│   └── XGboost/
│       ├── xgboost.ipynb
│       ├── recall_fixing.ipynb
│       └── threshold_config.json
│
├── student_resource/
│   ├── README.md
│   ├── Documentation_template.md
│   ├── Untitled.ipynb
│   └── utils/
│       └── validate_submission.py
│
├── .gitignore
└── README.md
```

The repository currently contains the main experimental implementations as Jupyter notebooks rather than a single packaged Python CLI pipeline.

### Directory Description

| Directory           | Purpose                                                                                     |
| ------------------- | ------------------------------------------------------------------------------------------- |
| `NN/`               | Neural/semantic candidate generation using sentence embeddings and nearest-neighbor search  |
| `method1_xgb/`      | XGBoost-based entity matching pipeline                                                      |
| `method4_lgbm/`     | LightGBM and XGBoost experiments, including recall-oriented candidate generation            |
| `student_resource/` | Challenge-provided documentation, dataset instructions, and submission validation utilities |

---

## Installation / Requirements

### Recommended Environment

The notebooks were developed around a modern Python environment and make use of both CPU and GPU acceleration depending on the method.

Create an isolated environment:

```bash
python -m venv .venv
```

Activate it on Linux/macOS:

```bash
source .venv/bin/activate
```

On Windows:

```powershell
.venv\Scripts\activate
```

Install the core dependencies:

```bash
pip install \
    pandas \
    numpy \
    scikit-learn \
    scipy \
    joblib \
    jellyfish \
    rapidfuzz \
    xgboost \
    lightgbm \
    tqdm
```

### Neural / Semantic Blocking Dependencies

The `NN` implementation additionally uses:

```bash
pip install \
    torch \
    sentence-transformers \
    faiss-cpu \
    sparse-dot-topn
```

For a CUDA-enabled environment, install the appropriate PyTorch and FAISS GPU builds for the target CUDA version.

The semantic implementation uses the `BAAI/bge-small-en-v1.5` sentence-transformer model together with FAISS for nearest-neighbor search.

### Dataset

The challenge data consists of TSV files:

```text
student_resource/
└── dataset/
    ├── train/
    │   ├── train_source1.tsv
    │   ├── train_source2.tsv
    │   ├── train_source3.tsv
    │   └── train_ground_truth.tsv
    │
    └── test/
        ├── test_source1.tsv
        ├── test_source2.tsv
        └── test_source3.tsv
```

The challenge files must be read using a tab separator:

```python
import pandas as pd

df = pd.read_csv(
    "dataset/train/train_source1.tsv",
    sep="\t"
)
```

The official challenge documentation specifies TSV input/output and describes Source 1 as the reference set, with Source 2 and Source 3 containing potential matches.

> **Note:** The notebooks in this repository contain environment-specific paths such as `ML2026/cleaned_dataset/...`. Update those paths to match your local dataset layout before execution.

---

## How to Run

Because the repository is notebook-oriented, the easiest way to reproduce the experiments is with Jupyter.

### 1. Start Jupyter

```bash
jupyter notebook
```

or:

```bash
jupyter lab
```

---

### 2. Run the XGBoost Pipeline

Open:

```text
method1_xgb/model_pipeline.ipynb
```

The pipeline performs:

1. Candidate-pair loading
2. Ground-truth labeling
3. Name and address feature generation
4. Pairwise similarity calculation
5. Grouped train/validation splitting
6. XGBoost training
7. F₀.₅ threshold calibration
8. Model serialization

The main pairwise features include:

* Jaro-Winkler name similarity
* Transliterated-name similarity
* Name Jaccard similarity
* Exact-name match
* Name-length difference
* Address Levenshtein similarity
* Address Jaccard similarity
* Address-missing indicator
* Address-digit overlap
* Country match

The implementation uses a grouped split by Source-1 entity to reduce leakage between training and validation records.

The trained threshold is stored in:

```text
method1_xgb/threshold_config.json
```

---

### 3. Run the Neural / Semantic Candidate Generator

Open:

```text
NN/Untitled.ipynb
```

The neural approach performs semantic candidate generation using:

```text
SentenceTransformer
        │
        ▼
BGE-small embeddings
        │
        ▼
FAISS nearest-neighbor search
        │
        ▼
Top-k semantic candidates
```

The notebook also combines semantic candidates with deterministic blocking strategies such as:

* Exact normalized-name matching
* Exact transliterated-name matching
* Country + name matching
* Country + address matching
* Sorted-name matching
* Address-digit matching
* Acronym-based matching

This produces a broader candidate set before downstream classification.

---

### 4. Run the LightGBM Pipeline

Open:

```text
method4_lgbm/LIGHTBGM/kartik_Lgbm.ipynb
```

The LightGBM pipeline uses candidate pairs generated through TF-IDF/blocking methods and constructs pairwise features including:

```text
tfidf_similarity
jaro_name
lev_address
country_match
len_diff_name
```

The model uses:

* LightGBM classifier
* Grouped train/validation split
* Early stopping
* Class-imbalance handling
* F₀.₅ threshold calibration

The resulting model and supporting artifacts are saved as:

```text
lgbm_f05_model.pkl
tfidf_vectorizer.pkl
pipeline_config.json
```

---

### 5. Run the XGBoost Variant

Open:

```text
method4_lgbm/XGboost/xgboost.ipynb
```

This implementation focuses on high-throughput pairwise feature computation and XGBoost classification.

The feature set includes:

```text
jaro_name
trans_jaro
name_jaccard
exact_name_match
len_diff_name
lev_address
addr_jaccard
addr_missing
addr_digit_match
country_match
```

The trained model uses grouped validation and F₀.₅-oriented threshold selection.

The selected threshold is stored in:

```text
method4_lgbm/XGboost/threshold_config.json
```

---

## Model Methods Used

### 1. Exact and Token-Based Blocking

The first layer reduces the enormous Source-1 × Source-2/3 search space.

Candidate generation uses signals such as:

* Exact normalized business name
* Exact transliterated business name
* Country + name
* Country + token
* Phonetic representations
* Address tokens
* Token-frequency filtering

High-frequency/common tokens are filtered to prevent extremely large candidate groups.

This approach is computationally cheap and provides strong deterministic signals.

---

### 2. TF-IDF Similarity

Character-level and text-level TF-IDF representations are used to retrieve potentially similar business records.

The approach combines fields such as:

```text
business name
transliterated name
normalized address
```

and performs sparse similarity search to generate candidate pairs.

This is particularly useful for:

* Typos
* Partial matches
* Formatting differences
* Small character-level changes
* Token variations

---

### 3. Neural Semantic Blocking

The `NN` approach uses:

```text
BAAI/bge-small-en-v1.5
```

to generate dense semantic representations.

FAISS is then used for efficient nearest-neighbor retrieval.

Conceptually:

```text
Business Record
      │
      ▼
Sentence Transformer
      │
      ▼
Dense Embedding
      │
      ▼
FAISS Index
      │
      ▼
Top-K Similar Records
```

This allows the candidate generator to capture similarity that may not be obvious from exact string matching.

---

### 4. Pairwise Feature Engineering

Candidate pairs are converted into numerical features before classification.

#### Name Features

* Exact match
* Jaro-Winkler similarity
* Transliterated-name similarity
* Jaccard token similarity
* Name-length difference

#### Address Features

* Normalized Levenshtein similarity
* Address token Jaccard similarity
* Address digit overlap
* Missing-address indicator

#### Metadata Features

* Country equality

These features allow the classifier to combine multiple weak signals instead of relying on a single string-matching rule.

---

### 5. XGBoost

XGBoost is used as a supervised binary classifier for:

```text
match = 1
non-match = 0
```

The model is trained on generated candidate pairs labeled from the challenge ground truth.

The implementation uses:

* Histogram-based tree construction
* Class imbalance handling
* Grouped validation
* Early stopping
* Probability-based inference

The grouped split uses the Source-1 entity ID as the grouping variable so that records belonging to the same reference entity do not leak across train and validation partitions.

---

### 6. LightGBM

LightGBM provides an alternative gradient-boosted-tree classifier.

The LightGBM implementation uses a compact feature set combining:

```text
TF-IDF similarity
Jaro-Winkler similarity
Address Levenshtein similarity
Country match
Name-length difference
```

It uses early stopping and imbalance handling during training.

---

### 7. F₀.₅ Threshold Calibration

The challenge is precision-sensitive, so the classifier's default probability threshold is not necessarily used.

Instead, candidate thresholds are evaluated on the validation set:

```text
probability
    │
    ├── threshold 0.70
    ├── threshold 0.75
    ├── threshold 0.80
    ├── ...
    └── threshold 0.95+
```

For each threshold, precision, recall, and F₀.₅ are calculated.

The threshold producing the highest validation F₀.₅ is stored as a configuration artifact and used during inference.

The XGBoost experiment in `method1_xgb` records a selected threshold of approximately `0.98` in its configuration.

---

## Evaluation Metric

The competition uses the precision-heavy F₀.₅ metric:

```text
F₀.₅ =
(1.25 × Precision × Recall)
--------------------------------
(0.25 × Precision) + Recall
```

F₀.₅ weights precision more heavily than recall.

This is important for entity resolution because an incorrect merge can incorrectly associate two different businesses.

The official challenge evaluates the metric as a macro-average across Source-1 entities and includes singleton entities.

---

## Expected Outputs

The final pipeline should produce:

```text
output/
├── matching_results.tsv
└── candidate_pairs.tsv
```

### `matching_results.tsv`

Contains the final predicted matches:

```text
source1_entity_id    matched_entity_ids
```

Example:

```text
S1-00001    S2-00047,S2-00193,S3-00812
S1-00002    S3-00004
S1-00003
```

### `candidate_pairs.tsv`

Contains the candidate set passed to the final matching model.

The final matches should be a subset of these candidates.

The challenge provides `student_resource/utils/validate_submission.py` for validating the submission format.

Run:

```bash
python3 student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir student_resource/dataset/test
```

---

## Method Summary

| Method                  | Candidate Generation        | Model                     | Main Signals                                       |
| ----------------------- | --------------------------- | ------------------------- | -------------------------------------------------- |
| `NN`                    | Sentence embeddings + FAISS | Neural semantic retrieval | Semantic name/address similarity                   |
| `method1_xgb`           | Strict blocking             | XGBoost                   | Jaro, Jaccard, Levenshtein, country, digit overlap |
| `method4_lgbm/LIGHTBGM` | TF-IDF/blocking             | LightGBM                  | TF-IDF, Jaro, address similarity, country          |
| `method4_lgbm/XGboost`  | TF-IDF/blocking             | XGBoost                   | String similarity + address + metadata             |
| `student_resource`      | Challenge resources         | —                         | Dataset + validation                               |

---

## Reproducibility Notes

The repository currently stores experimental workflows primarily as Jupyter notebooks. Some notebooks reference external/local paths and pre-generated intermediate files, so they may require path adjustments before being run on another machine.

For a fully reproducible deployment, the recommended structure is:

```text
src/
├── preprocessing.py
├── blocking.py
├── features.py
├── train.py
├── inference.py
└── evaluate.py

requirements.txt
README.md
```

with configuration separated from code:

```text
configs/
├── xgb.json
└── lgbm.json
```

and generated artifacts kept outside the source tree:

```text
artifacts/
output/
```

This avoids the classic notebook archaeology problem where a future reader has to determine which cell was run three Tuesdays ago.

---

## Challenge Compliance

The official challenge rules require the final submission to contain:

```text
output/
├── matching_results.tsv
└── candidate_pairs.tsv

code/
└── business_entity_resolution/
    ├── src/
    ├── README.md
    └── requirements.txt

Documentation_template.md
```

The challenge also prohibits external business-identity lookup and external entity-resolution data augmentation.

---

## Repository

**GitHub:**
https://github.com/KTsama07/Business-entity-Relationship

**Challenge:** Amazon ML Challenge 2026 — Business Entity Resolution

---
