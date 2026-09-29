# Business Entity Relationship

## Overview

This repository contains experimentation pipelines for the **ML Challenge 2026: Business Entity Resolution** task.  
The objective is to identify records from Source 2 and Source 3 that correspond to each Source 1 business entity, despite noisy names and addresses.

The project includes multiple modeling tracks (XGBoost, LightGBM, and neural/embedding-based candidate generation) implemented primarily as Jupyter notebooks.

---

## Repository Structure

```text
Business-entity-Relationship/
├── method1_xgb/
│   ├── model_pipeline.ipynb
│   └── threshold_config.json
├── method4_lgbm/
│   ├── LIGHTBGM/
│   │   ├── kartik_Lgbm.ipynb
│   │   └── pipeline_config.json
│   └── XGboost/
│       ├── xgboost.ipynb
│       ├── recall_fixing.ipynb
│       └── threshold_config.json
├── NN/
│   ├── Untitled.ipynb
│   └── Untitled1.ipynb
└── student_resource/
    ├── README.md
    ├── Documentation_template.md
    └── utils/validate_submission.py
```

---

## Installation / Requirements

### Environment
- Python 3.10+ (3.12 also used in notebook outputs)
- Jupyter Notebook / JupyterLab

### Common Python dependencies used in this repository
- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost`
- `lightgbm`
- `jellyfish`
- `joblib`
- `tqdm`
- `torch`
- `sentence-transformers`
- `faiss` (GPU/CPU variant, depending on environment)
- `sparse-dot-topn`

Install baseline dependencies:

```bash
pip install pandas numpy scikit-learn xgboost lightgbm jellyfish joblib tqdm torch sentence-transformers sparse-dot-topn
```

> Notes:
> - Some notebooks assume GPU availability for faster embedding/candidate generation.
> - Data files are expected externally in challenge-specific folders referenced by notebooks (for example `ML2026/...`).

---

## How to Run

Because this repository is notebook-driven, execution is done per method notebook.

1. Launch Jupyter in the repository root:
   ```bash
   jupyter notebook
   ```
2. Open one of the pipelines:
   - `/home/runner/work/Business-entity-Relationship/Business-entity-Relationship/method1_xgb/model_pipeline.ipynb`
   - `/home/runner/work/Business-entity-Relationship/Business-entity-Relationship/method4_lgbm/LIGHTBGM/kartik_Lgbm.ipynb`
   - `/home/runner/work/Business-entity-Relationship/Business-entity-Relationship/method4_lgbm/XGboost/xgboost.ipynb`
   - `/home/runner/work/Business-entity-Relationship/Business-entity-Relationship/NN/Untitled.ipynb`
3. Update dataset paths inside notebook cells (if needed) to match your local data location.
4. Run cells sequentially to:
   - build candidate pairs,
   - engineer pairwise features,
   - train model(s),
   - calibrate decision threshold,
   - generate prediction outputs.
5. Validate submission-format files (from `student_resource/`):
   ```bash
   python3 utils/validate_submission.py \
       --matching output/matching_results.tsv \
       --candidate output/candidate_pairs.tsv \
       --test-dir dataset/test
   ```

---

## Model Methods Used

### 1) XGBoost classification pipeline
- Gradient-boosted classifier (`XGBClassifier`) on engineered similarity features.
- Feature set includes:
  - Jaro(-Winkler) name similarity
  - transliterated-name similarity
  - token Jaccard overlap (name/address)
  - normalized Levenshtein-based address similarity
  - country-match and length-difference signals
- Threshold tuning is performed for precision-heavy **F0.5** optimization.

### 2) LightGBM pipeline
- LightGBM classifier over candidate pairs with lexical similarity features.
- Uses blocking + candidate generation followed by supervised scoring.
- Stores calibrated threshold in `pipeline_config.json`.

### 3) Hybrid neural candidate generation (NN folder)
- Semantic blocking with sentence embeddings (`sentence-transformers` + `FAISS`).
- Combined with lexical/TF-IDF-based blocking signals.
- Designed to improve candidate recall before downstream classification.

### 4) Blocking and recall-improvement experiments
- Additional notebook experiments in `method4_lgbm/XGboost/recall_fixing.ipynb`
  focus on strengthening candidate recall through TF-IDF and strict blocking strategies.

