# Unsupervised Domain Adaptation from Data Lakes

**Asael Avia · April 2026**

Zero-shot binary classification by extracting labeled signal directly from an unlabeled data lake — no external labeled datasets, no human annotation.

---

## Problem

> **Given:** A large lake of unlabeled tables (GitTables, ~421K CSVs from GitHub) + an unlabeled target dataset.  
> **Goal:** Predict a binary label on the target, using only what the lake contains.

All labeled signal is extracted via **source repurposing**: scanning the lake for columns whose names semantically match the target label concept and treating those columns as proxy labels. The target dataset remains fully unlabeled at training time.

---

## Pipeline

```
INPUT: Data lake (unlabeled tables) + unlabeled target table
         │
         ▼
STEP 0: SOURCE REPURPOSING
         │  LLM expands target label → concept synonyms
         │  Scan all lake tables; cosine sim ≥ 0.70 → repurpose column as pseudo-label
         ▼
STEP 1: TABLE DISCOVERY
         │  Embed column names → bipartite matching → discovery score per source
         ▼
STEP 2: SCHEMA ALIGNMENT
         │  Hungarian matching → align source columns to target schema
         ▼
STEP 3: WEIGHTED MULTI-SOURCE ADAPTATION
         │  Baseline / L0 / L2 (pseudo-labeling) / L5 (DANN) / L5.5 (CDAN) / L6 (FTTA)
         ▼
STEP 4: EVALUATION
         └─ compare all levels + oracle (supervised upper bound)
```

---

## Key Results

### Best UDA AUC vs Oracle (GitTables lake, 10 targets)

| Target | Baseline | Best UDA | Oracle | % Gap Closed |
|--------|----------|----------|--------|-------------|
| Adult income | 0.521 | **0.760** | 0.927 | 59% |
| Bank marketing | 0.392 | **0.677** | 0.929 | 53% |
| Diabetes | 0.512 | **0.760** | 0.816 | 82% |
| Heart disease | 0.554 | **0.803** | 0.881 | 76% |
| Medical no-show | 0.424 | **0.694** | 0.723 | 90% |
| NY house price | 0.630 | **0.791** | 0.938 | 52% |
| County obesity | 0.693 | **0.793** | 0.899 | 48% |
| Employee turnover | 0.436 | **0.516** | 0.725 | 28% |
| Credit risk | 0.412 | **0.453** | 0.765 | 12% |
| Customer churn | **0.637** | 0.566 | 0.843 | −35% |

Best adapted model beats the equal-weight baseline on 9 of 10 targets.

### The lake can substitute for labeled data

On several targets, zero-shot UDA outperforms training on 0.1–1% of labeled target data:

| Target | UDA AUC | Beats labels-only through |
|--------|---------|--------------------------|
| Adult income | 0.760 | 0.1% labeled (~24 rows) |
| Diabetes | 0.760 | 1% labeled (~6 positives) |
| Heart disease | 0.803 | 5% labeled (~6 positives) |
| NY house price | 0.778 | 0.5% labeled |

### LLM concept expansion is a prerequisite, not an optimization

Repurposing with only the raw label name finds **0 sources** for every tested target. Real column names use domain-specific conventions (`heart_disease`, `CAD`, `troponin`) that require expansion.

| Target | Sources with LLM expansion | Sources without |
|--------|--------------------------|-----------------|
| Adult income | 256 | **0** (400K tables scanned) |
| Heart disease | 96 | **0** |
| Diabetes | 1,037 | **0** |

### Transferability prediction (PAS)

PCA-whitened centroid margin (PAS) predicts oracle gap closed with Spearman ρ = **+0.685** (p = 0.014) across 12 targets — formally significant with n = 12.

---

## Setup

```bash
conda env create -f environment.yml
conda activate uda-datalake

# Additional dependencies (not in environment.yml)
pip install torch xgboost matplotlib shap datasets
```

---

## Data

### Data Lakes

#### GitTables (primary lake — required for Act 5)

~421K CSV tables scraped from GitHub, stored as parquet. Download from Zenodo (~50 GB, resumable):

```bash
python gittables_lake.py --download-zenodo
python gittables_lake.py --stats    # check progress
```

Output: `data/gittables/` with parquet files and `manifest.json`.

#### GovData (US government open data — optional)

Fetches tables via the Socrata API. No authentication required.

```bash
python download_govdata.py --max-tables 10000
```

Output: `data/govdata/`

#### WikiTables (optional)

Streams from HuggingFace (`penfever/wikitables`, 1.65M Wikipedia tables).

```bash
pip install datasets
python download_wikitables.py --max-tables 50000
```

Output: `data/wikitables/`

#### OpenML lake (optional)

Downloads tabular datasets from OpenML via their API.

```bash
python download_openml_lake.py --max-tables 5000
```

Output: `data/openml_lake/`

---

### Target Datasets

Most targets are auto-downloaded on first run via the OpenML API or `folktables`.

| Target | How obtained |
|--------|-------------|
| Adult income | Auto-downloaded via `folktables` |
| Bank marketing | Auto-downloaded from OpenML (ID 1461) |
| Pima diabetes | Auto-downloaded from OpenML (ID 37) |
| German credit | Auto-downloaded from OpenML (ID 31) |
| Customer churn | Auto-downloaded from OpenML (ID 42178) |
| Heart disease | Auto-downloaded from OpenML (ID 53) |
| Employee turnover | Auto-downloaded from OpenML (ID 43551) |
| Communities & Crime | Auto-downloaded from OpenML (ID 43891) |
| NY house price | **Manual** — download from [Kaggle](https://www.kaggle.com/datasets/nelgiriyewithana/new-york-house-price-dataset), place CSV at `data/NY-Housing/nyhouse.csv` |
| Medical no-show | **Manual** — download `KaggleV2-May-2016.csv` from [Kaggle](https://www.kaggle.com/datasets/joniarroba/noshowappointments), place at `data/KaggleV2-May-2016.csv` |
| County obesity | **Manual** — download CDC PLACES 2023 county data from [CDC](https://data.cdc.gov/500-Cities-Places/PLACES-County-Data-GIS-Friendly-Format-2023-releas/i46a-9kgh), place at `data/cdc_obesity_county.csv` |

---

### LLM Concept Expansion (optional — cache included)

Concept lists for all 10 targets are pre-generated and cached in `data/llm_expansion_cache.json`. No LLM setup needed to reproduce results.

To regenerate or add a new target, install [Ollama](https://ollama.ai) and pull the model:

```bash
ollama pull qwen2.5:7b
# Concept generation runs automatically on the first scan of a new target
```

---

## Running Experiments

```bash
# Main experiment (Act 5 — GitTables unlabeled lake)
python act5_gittables_lake.py --target adult
python act5_gittables_lake.py --target heart
python act5_gittables_lake.py --target diabetes
# ... all 10 targets

# Semi-supervised extension (Act 6 — lake + small labeled set)
python act6_semi_supervised.py --target adult

# Multi-lake comparison (GitTables vs GovData vs OpenML vs WikiTables)
python compare_lakes.py --target adult

# Scalability analysis (AUC vs lake size from 5K to 421K tables)
python run_scalability.py
python scalability_analysis.py

# Transferability score analysis
python transferability_analysis.py

# SHAP feature importance
python shap_analysis.py --target adult
```

---

## File Structure

```
uda-datalake/
├── act5_gittables_lake.py       # Main experiment (unlabeled lake)
├── act6_semi_supervised.py      # Semi-supervised VLA extension
├── act4_openml_lake.py          # Labeled lake baseline
├── compare_lakes.py             # Multi-lake comparison runner
├── run_scalability.py           # Scalability sweep runner
├── scalability_analysis.py      # Scalability plots and tables
├── transferability_analysis.py  # PAS / transferability scores
├── shap_analysis.py             # SHAP feature importance
├── table_discovery.py           # Steps 0+1: repurposing + discovery
├── schema_alignment.py          # Step 2: Hungarian column matching
├── domain_adaptation.py         # Step 3: all adaptation levels
├── evaluation.py                # Step 4: metrics
├── gittables_lake.py            # GitTables downloader + loader
├── download_govdata.py          # GovData Socrata downloader
├── data/
│   ├── llm_expansion_cache.json # Pre-generated concept lists (all targets)
│   └── gittables/               # GitTables parquet files + manifest.json
└── results/
    ├── act5/                    # Per-target results (metrics.csv, plots)
    ├── scalability/             # Scalability analysis outputs
    └── transferability/         # Transferability analysis outputs
```

---

## Adaptation Methods

| Level | Method | Description |
|-------|--------|-------------|
| Baseline | Equal-weight pooling | All sources concatenated equally — demonstrates negative transfer |
| L0 | Direct transfer | Single best source by discovery score |
| L2 | Pseudo-labeling | Self-training from L0 over 5 rounds |
| L5 | DANN | Gradient reversal adversarial alignment |
| L5.5 | CDAN | Conditional DANN (features × softmax outer product) |
| L6 | FTTA | Feature-space test-time adaptation with KL anchor |
| Oracle | Supervised XGBoost | Trained on 80% of ground-truth target labels — upper bound only |

---

## Data Lakes

| Lake | Size | Description |
|------|------|-------------|
| **GitTables** | ~421K tables | CSVs from GitHub (primary lake) |
| **GovData** | ~5K tables | US government open data (Socrata API) |
| **OpenML** | ~9.8K tables | Curated ML datasets treated as unlabeled |
| **WikiTables** | ~10K tables | Wikipedia tables |

---

## Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Encoder | `all-MiniLM-L6-v2` (frozen) |
| Repurposing threshold | 0.70 cosine similarity |
| Top-K sources selected | 20 |
| DANN training epochs | 200 |
| XGBoost | default params, `eval_metric=logloss` |

---

## Full Results

See [RESULTS.md](RESULTS.md) for:
- Complete AUC / F1 / accuracy tables across all levels and targets
- Semi-supervised VLA results (Act 6) with mean ± std over 5 seeds
- Multi-lake comparison table
- Transferability score analysis and correlation table
- SHAP feature importance per target
- Concept expansion ablation
- Scalability analysis (AUC and runtime vs lake size)
- Limitations and related work

---

## Citation

If you use this work, please cite:

```bibtex
@misc{avia2026uda,
  author = {Asael Avia},
  title  = {Unsupervised Domain Adaptation from Data Lakes},
  year   = {2026},
  url    = {https://github.com/asaelavia/uda-datalake}
}
```
