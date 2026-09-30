# Amazon ML Challenge 2026: Multi-Source Business Entity Resolution

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0%2B-green.svg)](https://lightgbm.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end, high-throughput machine learning solution designed for the **Amazon ML Challenge 2026**. This system resolves and deduplicates canonical reference businesses (Source 1) against high-volume, noisy, unindexed external candidate pools (Source 2 and Source 3) across **11.7M+ total records** and an unconstrained search space of $\approx 1.73 \times 10^{13}$ pairwise comparisons.

---

## 🚀 Key Highlights & Performance

* **99.9998% Search Space Reduction**: Bounded-VRAM GPU sparse candidate blocking (Formula B) retrieves candidate pools in milliseconds while maintaining **87.72% Ground Truth Recall @ Top-15**.
* **High-Throughput Streaming**: Achieves **650+ queries/second** on a single GPU ($< 1\text{ GB}$ VRAM, $< 2.8\text{ GB}$ RAM), streaming inference across **1.73M test entities in $\sim$40 minutes**.
* **Domain Feature Engineering**: 19 composite pairwise comparison features including cross-lingual French administrative region ontology resolution (`FRANCE_DEP_TO_REGION`), regex house-number verification, and brand token interaction metrics.
* **Metric Alignment (Macro $F_{0.5}$)**: Trained on a 5-fold `GroupKFold` LightGBM ensemble; calibrated decision threshold to **`0.75`** to penalize false merges 2× as harshly as missed links and safeguard 5.6% true singleton entities.
* **Session Resilience**: Automated Parquet checkpointing (`data/processed/`) protects long-running inference against kernel crashes and memory loss.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph Stage1["1. Multilingual Preprocessing & Normalization"]
        S1["Source 1 (Reference)"]
        S2["Source 2 (Candidate Pool)"]
        S3["Source 3 (Candidate Pool)"]
        Norm["Universal Cleaning & Text Normalization<br/>• Possessive Stripping ('s)<br/>• Standardized State Extraction (US/India)<br/>• French Dept-to-Region Resolution<br/>• Granular House Number Extraction"]
        S1 & S2 & S3 --> Norm
    end

    subgraph Stage2["2. High-Throughput GPU Candidate Blocking"]
        Part["Country Partitioning (US, India, France)"]
        DualVec["Dual Hashing Vectorizers (n=2^19)<br/>• Len-1: Initial/Acronym tokens (?u)\\b\\w\\b<br/>• GT-1: Brand tokens (?u)\\b\\w{2,}\\b<br/>• Addr: Street tokens (?u)\\b\\w{2,}\\b"]
        GPU["Vectorized GPU Sparse MM (Formula B)<br/>Score = 1.0*L1 + 4.0*GT1 + 2.0*Addr + 2.0*State<br/>Chunk Size = 500,000 | Batch Size = 500"]
        TopK["NumPy Vectorized Top-K Merge (k=15)<br/>Recall @ 15: 87.72% | Search Reduction: 99.9998%"]
        Norm --> Part --> DualVec --> GPU --> TopK
    end

    subgraph Stage3["3. Pairwise Feature Engineering"]
        Feat["19 Pairwise Comparison Features<br/>(Jaccard, House Number, State Match, Source S3,<br/>Name x Addr Interaction, Score per Word)"]
        TopK --> Feat
    end

    subgraph Stage4["4. Supervised Ranking & Inference"]
        LGBM["5-Fold GroupKFold LightGBM Ensemble<br/>(scale_pos_weight = 3.92, Grouped by s1_id)"]
        Thresh["Precision-Weighted Threshold Tuning (0.75)<br/>Val ROC-AUC: 0.9850 | PR-AUC: 0.9502"]
        Feat --> LGBM --> Thresh
    end

    subgraph Stage5["5. Streaming Export & Submission Validator"]
        Disk["Direct Streaming Append to TSVs<br/>• output/matching_results.tsv<br/>• output/candidate_pairs.tsv"]
        Val["student_resource/utils/validate_submission.py"]
        Zip["Official Package: DataResolvers_submission.zip"]
        Thresh --> Disk --> Val --> Zip
    end
```

---

## 📊 Empirical Blocking Benchmark (Formula B vs. Baseline)

Evaluated under strict empirical conditions on 500 reference entities across 501,692 candidate pool records:

| Metric | Initial Baseline (v2) | Formula B Engine | Net Improvement |
| :--- | :---: | :---: | :---: |
| **Top-15 Recall** | 83.30% | **88.36%** | **+5.06% (+90 recovered targets)** |
| **Full Entity Recovery** | 55.40% | **63.60%** | **+8.20% higher full linkage** |
| **Mean Reciprocal Rank (MRR)** | 0.8991 | **0.9077** | **+0.0086** |
| **Throughput Speed** | 41.3 queries/sec | **59.8 queries/sec** | **+44.8% faster** |

---

## 📁 Repository Structure

```text
├── LICENSE                              # MIT Open Source License
├── README.md                            # Comprehensive project overview & architecture
├── requirements.txt                     # Root dependencies (PyTorch, LightGBM, Scikit-learn, etc.)
├── Documentation_template.md            # Official methodology report
├── workflow.md                          # In-depth hypothesis testing, benchmarks & mathematical formulation
├── package_submission.py                # Automated packaging utility (<team_name>_submission.zip)
├── notebooks/
│   ├── eda_train.ipynb                  # Master interactive EDA, feature extraction & training notebook
│   └── entity_resolution_v2.ipynb       # Baseline blocking exploration notebook
├── code/
│   └── business_entity_resolution/      # Official deliverable package
│       ├── src/
│       │   ├── __init__.py
│       │   ├── preprocessing.py         # Universal text cleaning, state, house #, French ontologies
│       │   ├── gpu_blocking.py          # Vectorized GPU candidate blocking engine (Formula B)
│       │   ├── feature_engineering.py   # 19 composite pairwise comparison features
│       │   ├── train.py                 # 5-fold GroupKFold LightGBM trainer
│       │   └── inference.py             # Memory-bounded streaming inference runner
│       ├── README.md                    # Package-level instructions
│       └── requirements.txt             # Pinned production dependencies
├── utils/
│   └── validate_submission.py           # Standalone official competition format validator
└── output/                              # Generated submission TSVs (mock/samples tracked)
```

---

## 🛠️ Quickstart & Reproduction

### 1. Environment Setup

```bash
git clone https://github.com/Prashanf/amazon-business-entity-resolution.git
cd amazon-business-entity-resolution
pip install -r requirements.txt
```

### 2. Run Full Test Inference

```bash
python -m code.business_entity_resolution.src.inference \
    --test-dir dataset/test \
    --output-dir output \
    --cache-dir data/processed \
    --threshold 0.75 \
    --batch-size 500
```

### 3. Validate & Package Official Submission

```bash
# Validate format and integrity using standalone validator
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test

# Package deliverable ZIP
python package_submission.py
```

---

## 👤 Author

**Prashant Sharma**  
* Indian Institute of Technology Madras (IITM)  
* Email: [22f1000640@ds.study.iitm.ac.in](mailto:22f1000640@ds.study.iitm.ac.in)  
* GitHub: [@Prashanf](https://github.com/Prashanf)  

**Team:** DataResolvers  
*Amazon ML Challenge 2026*
