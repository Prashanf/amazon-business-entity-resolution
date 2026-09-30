# Business Entity Resolution Pipeline — Amazon ML Challenge 2026

An end-to-end, high-throughput solution for Multi-Source Business Entity Resolution across reference tables (Source 1) and noisy candidate pools (Source 2 & Source 3).

## System Architecture

1. **Preprocessing & Multilingual Standardization**:
   - Universal unidecode transliteration, casing, and punctuation removal.
   - Legal entity suffix removal (`corp`, `inc`, `pvt ltd`, `sarl`, `sa`, etc.).
   - Standardized street expansion dictionary across US, India, and France (`rue`, `ave`, `bd`, `pl`, `zi`, etc.).
   - Hierarchical Administrative Resolution: Maps French departments/cities to administrative regions (`Hauts-de-France`, `Nouvelle-Aquitaine`, `Pays de la Loire`, `Ile-de-France`).
   - Street / Flat / Plot house number extraction.

2. **Candidate Blocking (GPU Formula B)**:
   - Partitions candidate pools strictly by country (`US`, `India`, `France`).
   - Uses Dual Sparse Hashing Vectorizers (`token_pattern=r"\b\w\b"` for single-char initials vs `r"\b\w{2,}\b"` for core brand words).
   - Pre-vectorizes candidate chunks into Scipy CSR matrices on CPU.
   - Evaluates multi-signal similarity on GPU:
     $$\text{Score} = 1.0 \times \text{len1} + 4.0 \times \text{len>1} + 2.0 \times \text{addr} + 2.0 \times \text{state}$$
   - Retains Top-15 candidate pool per $S_1$ entity (87.65% Recall on Ground Truth).

3. **Pairwise Feature Engineering (19 Signals)**:
   - Blocking score, word count, Jaccard similarity, core exact match flag, substring flag.
   - State presence flags and exact match flag.
   - House number presence flags and exact match flag.
   - Address word count and address Jaccard similarity.
   - Source indicator (`S3`), cross-interactions (`name_x_addr_jaccard`, `both_exact_and_state`, `score_per_word`).

4. **Supervised GBDT Matching Model**:
   - 5-Fold GroupKFold LightGBM Classifier (grouped strictly by `s1_id` to eliminate data leakage).
   - Imbalance compensation with `scale_pos_weight = 3.9`.
   - Out-of-fold threshold optimization for macro $F_{0.5}$ (optimal threshold = `0.75`).

5. **Submission Generation**:
   - Streams $S_1$ inference in memory-bounded blocks directly to disk.
   - Outputs `matching_results.tsv` (leaderboard-scored) and `candidate_pairs.tsv`.
   - Singletons with no match are cleanly formatted as empty strings `""`.

---

## Installation & Requirements

```bash
pip install -r requirements.txt
```

Hardware requirement: Any CUDA-capable GPU (NVIDIA T4 / P100 / V100 / A100) or CPU fallback.

---

## Reproduction Instructions

### Step 1: Model Training
To train the LightGBM model from raw training data:
```python
from src.preprocessing import preprocess_dataset
from src.gpu_blocking import block_candidates_gpu
from src.feature_engineering import compute_pair_features
from src.train import train_lgbm_model

# Run preprocessing on train datasets
# Run candidate blocking and feature extraction
# Train model booster
models, best_thresh = train_lgbm_model(train_pairs_df, model_save_path="models/lgbm_entity_resolver_v1.txt")
```

### Step 2: Full Test Inference
To run full-scale streaming inference on the test set:
```bash
python -m src.inference \
    --test-dir dataset/test \
    --output-dir output \
    --cache-dir data/processed \
    --model-path models/lgbm_entity_resolver_v1.txt \
    --threshold 0.75 \
    --batch-size 500
```

### Step 3: Validate Outputs
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
