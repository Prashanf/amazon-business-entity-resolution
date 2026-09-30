# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** DataResolvers  
**Team Members:** Prashant Sharma  
**Submission Date:** September 2026  

---

## 1. Executive Summary
We present a high-throughput, memory-bounded Multi-Source Business Entity Resolution system designed to link noisy candidate records from Source 2 and Source 3 back to canonical reference entities in Source 1. Our architecture combines a GPU-accelerated sparse candidate blocking pipeline (Formula B) achieving 87.65% Recall @ Top-15 with a 5-fold GroupKFold LightGBM classifier trained on 19 composite pairwise comparison features. By incorporating French administrative region mapping, house-number verification, and out-of-fold threshold optimization for macro $F_{0.5}$, our solution balances precision and recall while scaling efficiently across millions of records.

---

## 2. Methodology

### 2.1 Problem Analysis
During exploratory data analysis across training and test sets, we identified several fundamental data characteristics:
- **Severe Class Imbalance**: In candidate pairs generated through blocking, true matches represent only ~20.3% of pairs (an imbalance ratio of 1 : 3.9).
- **Heterogeneous Noise Patterns**:
  - Name abbreviations, acronyms, and legal suffix variations (`Corp`, `Pvt Ltd`, `LLC`, `SARL`).
  - Street address contractions (`St`, `Rd`, `Ave`, `Blvd`, `Rue`, `Impasse`, `ZI`).
  - Transliteration inconsistencies and landmark-based address descriptions in India.
  - Test set domain shift: The addition of `France` in the test set, where reference addresses list administrative regions (e.g., `Nouvelle-Aquitaine`, `Hauts-de-France`) while candidate sources list constituent departments (e.g., `Gironde`, `Nord`) or cities (`Bordeaux`, `Lille`).
- **Singleton Prevalence**: ~5.6% to 6.2% of Source 1 entities have zero true matches in external candidate sources. Accurately outputting empty lists for singletons is crucial under the competition's macro-$F_{0.5}$ metric.

### 2.2 Solution Strategy
We adopt a **Two-Stage Hybrid Architecture**:
1. **Stage 1 (Candidate Generation / Blocking)**: Fast, candidate screening using PyTorch sparse matrix multiplications on GPU with multi-signal weighted scoring to isolate the Top-15 candidate pool per reference entity.
2. **Stage 2 (Supervised Matching & Classification)**: A supervised GBDT ensemble (LightGBM) trained on 19 rich pairwise comparison features with group-aware cross-validation and probability threshold tuning to maximize the precision-weighted macro $F_{0.5}$ objective.

**Approach Type:** High-Throughput GPU Candidate Blocking + 5-Fold Grouped GBDT Classifier  
**Core Innovation:** Bounded-VRAM GPU streaming blocking utilizing dual hashing vectorizers (isolating single-letter initials from multi-letter brand tokens) combined with hierarchical French department-to-region ontology resolution.

---

## 3. Candidate Generation (Blocking)

To reduce the quadratic comparison space ($1.73\text{M} \times 10\text{M} \approx 1.7 \times 10^{13}$ pairs) to a manageable set:
- **Country Partitioning**: Candidate pools are partitioned strictly by country (`US`, `India`, `France`), eliminating cross-border noise and reducing memory footprint.
- **Dual Hashing Vectorizers**: 
  - Token pattern `(?u)\b\w\b` isolates single-character acronyms/initials.
  - Token pattern `(?u)\b\w{2,}\b` isolates core brand tokens.
  - Address vectorizer isolates street and city tokens.
- **Formula B Multi-Signal Ranking**:
  $$\text{Score} = 1.0 \times \text{overlap}_{\text{len1}} + 4.0 \times \text{overlap}_{\text{gt1}} + 2.0 \times \text{addr\_overlap} + 2.0 \times \text{state\_match}$$
- **Recall Verification**: Evaluated on 10,000 reference entities with 34,752 ground truth targets, our Top-15 blocking retrieved **30,461 true matches (87.65% Recall)** while reducing the candidate space by **99.9998%**.
- **Bounded VRAM Streaming**: Candidate chunks are pre-vectorized once on CPU into Scipy CSR matrices, and query batches are streamed through GPU with automatic VRAM cache clearing (`< 1.0 GB` peak VRAM).

---

## 4. Matching Model

### 4.1 Feature Taxonomy (19 Comparison Features)
- **Name Signals**: `blocking_score`, `common_words_count`, `word_jaccard`, `is_exact_core_match`, `is_substring_match`, `score_per_word`.
- **Administrative Location Signals**: `has_state_s1`, `has_state_cand`, `both_have_state`, `state_match` (hierarchical state/region equivalence).
- **Granular Address Signals**: `extract_house_number`, `has_house_num_s1`, `has_house_num_cand`, `both_have_house_num`, `house_number_match`, `address_common_words_count`, `address_word_jaccard`.
- **Interaction & Source Features**: `source_is_s3`, `name_x_addr_jaccard`, `both_exact_and_state`.

### 4.2 Model Architecture & Training
- **Model Type**: LightGBM Gradient Boosted Decision Tree (5-Fold Ensemble).
- **Validation Scheme**: `GroupKFold` grouped strictly by `s1_id` across 150,000 candidate pairs. This prevents data leakage by ensuring that all candidate pairs for any given reference entity reside in either the training fold or the validation fold, never split across both.
- **Imbalance Handling**: Configured `scale_pos_weight = 3.9` to counter the 1:3.9 negative-to-positive candidate ratio.
- **Threshold Selection**: Calibrated on out-of-fold predictions across a grid from 0.10 to 0.90. An optimal decision cutoff of **`0.75`** was selected to favor high precision, directly aligning with the competition's $F_{0.5}$ metric (which weights precision 2x over recall).

---

## 5. Results & Error Analysis

- **Out-of-Fold Validation Metrics**:
  - ROC-AUC: **0.9624**
  - PR-AUC: **0.9018**
  - Precision: **88.42%**
  - Recall: **82.15%**
  - Optimal Threshold: **0.75**
- **Test Set Inference Statistics (Pilot 10,000 Entities)**:
  - Total Matches Predicted: **34,057**
  - Entities with $\ge 1$ Match: **93.8%** (matches Ground Truth distribution of 94.4%)
  - Singletons Identified: **6.2%** (matches Ground Truth distribution of 5.6%)
- **Error Analysis**:
  - *False Positives*: Commonly arise from franchise chains sharing identical brand names and city names but differing slightly in street suffix (e.g., "Subway Main St" vs "Subway North Main Ave"). Addressed by our `house_number_match` feature.
  - *False Negatives*: Arise from extreme spelling transliterations in Indian regional addresses where no common n-grams survived initial blocking.

---

## 6. Conclusion
Our entity resolution architecture demonstrates that coupling domain-tailored GPU sparse candidate blocking with gradient boosted decision trees enables scalable, highly accurate record linkage across millions of noisy commercial business listings. The combination of hierarchical administrative mapping, house number matching, and strict group-aware cross-validation provides strong generalization to unseen international test distributions.

---

## Appendix

### A. Code Artefacts
The reproducible codebase is organized under `code/business_entity_resolution/`:
- `src/preprocessing.py`: Multilingual text normalization, French department-to-region mapping, and street/house number parsers.
- `src/gpu_blocking.py`: GPU tensor blocking with Formula B multi-signal ranking and bounded VRAM streaming.
- `src/feature_engineering.py`: Pairwise 19-feature extraction.
- `src/train.py`: 5-fold GroupKFold LightGBM training and threshold calibration.
- `src/inference.py`: Full streaming test inference pipeline.
- `requirements.txt`: Pinned dependencies (`torch`, `lightgbm`, `scikit-learn`, `pandas`, `unidecode`).
- `README.md`: Step-by-step reproduction instructions.
