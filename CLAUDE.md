# CLAUDE.md — Unsupervised Domain Adaptation in Data Lakes

## Problem Statement

**Input:** A true data lake (GitTables — ~1M unlabeled CSV tables from GitHub) + a target table (unlabeled, needs predictions).
**Output:** Predictions for the target table, using labeled sources extracted from the lake via source repurposing.

**Key constraint:** No external labeled datasets. All labeled signal must come from the lake itself via source repurposing (finding columns whose names semantically match the target label).

---

## Pipeline

```
INPUT: GitTables lake (~1M tables, all unlabeled) + unlabeled target table
         │
         ▼
STEP 0: SOURCE REPURPOSING
         │  scan all lake tables → find columns semantically matching target label
         │  → repurposed tables become labeled sources (feature col = pseudo-label)
         ▼
STEP 1: TABLE DISCOVERY
         │  embed columns → cosine similarity + bipartite matching → ranked scores
         ▼
STEP 2: SCHEMA ALIGNMENT
         │  Hungarian matching → aligned DataFrames per source
         ▼
STEP 3: WEIGHTED MULTI-SOURCE ADAPTATION
         │  Baseline / Level 0 / Level 1 / Level 2 / Level 3 / Level 5
         │  Non-repurposed lake tables used as unlabeled domain data in L3/L5
         ▼
STEP 4: EVALUATION
         └─ compare levels + naive baseline → show negative transfer
```

One module per step. Do not merge steps.

---

## Step Details

### Step 0 — Source Repurposing (`table_discovery.py`, `_stream_load_and_repurpose` in `act5_gittables_lake.py`)
- Scan all GitTables lake tables for columns whose name is semantically similar to the target label
- Concept expansion: LLM (Ollama/Qwen3.5) → KG (DBpedia fallback) → label-only fallback
- **Critical distinction:** concepts must be synonyms/proxy labels (columns that COULD BE the label), NOT predictive features. E.g. for "income": salary, wages ✓ — occupation, education ✗
- Concepts cached in `data/llm_expansion_cache.json` (pre-populated for all 8 targets via Claude Opus)
- Repurposing threshold: **0.70** (with expanded concept lists; 0.35 matches too many tables)
- Repurposed column is binarized (median split) to produce a binary label
- Streaming single-pass: reads each parquet, checks threshold, discards non-matches immediately
- Checkpointed every 5% to `data/stream_ckpt_*.json` — safe to pause/resume
- **Output:** subset of lake tables promoted to labeled sources, with assigned label column

### Step 1 — Table Discovery (`table_discovery.py`)
- Embed column names with `sentence-transformers` (frozen, no training)
- Compute pairwise table similarity: cosine similarity + bipartite matching score
- Rank lake tables by relevance to the target
- TOP_K = **20** tables selected for adaptation
- **Output:** `dict[table_id → similarity_score]`

### Step 2 — Schema Alignment (`schema_alignment.py`)
- For each relevant source table, map its columns to target columns via Hungarian algorithm (`scipy.optimize.linear_sum_assignment`)
- Drop unmatched columns, rename matched ones to target schema
- **Output:** aligned `DataFrame` per source table

### Step 3 — Weighted Multi-Source Adaptation (`domain_adaptation.py`)

| Level | Description |
|---|---|
| **Baseline** | Naive equal-weight combination of all sources (demonstrates negative transfer) |
| **Level 0** | Direct transfer from single best source (highest discovery score) |
| **Level 1** | Weighted multi-source: XGBoost trained with per-sample weights proportional to discovery scores |
| **Level 2** | Level 1 + pseudo-labeling: predict on target, add high-confidence pseudo-labeled rows, retrain |
| **Level 3** | Instance reweighting via domain classifier (source vs target); unlabeled lake tables augment source side |
| **Level 5** | DANN adversarial adaptation; unlabeled lake tables augment domain discriminator |

Removed: L4 (L3+pseudo, marginal) and L6 (prediction stacking, too few features).

- **Output:** predictions for the target table, one set per level

### Step 4 — Evaluation (`evaluation.py`)
- Compare all levels and the naive baseline
- Primary result: equal-weight baseline hurts (negative transfer); discovery-weighted approach prevents it
- **Output:** metrics table (accuracy / F1 / AUC) across levels, both default and calibrated thresholds

---

## Data Lake

**Primary lake: GitTables** (https://gittables.github.io)
- ~1M CSV tables extracted from GitHub repositories
- Stored locally as parquet files in `data/gittables/` with `manifest.json`
- Downloaded from Zenodo record 6517052 via `python gittables_lake.py --download-zenodo`
- Also includes ~9,807 tables from `target-benchmark/gittables-corpus` (HuggingFace)
- **All tables are unlabeled** — labeled signal extracted only via source repurposing
- GPU-accelerated repurposing scan: RTX 4060, batch_size=512 tables per encode call

**Previous lake (completed, for reference):** OpenML 787-lake
- 705 labeled binary classification datasets from OpenML
- Run via `act4_openml_lake.py`
- Clean results (contamination fixed): oracle > adapted for all 5 targets

---

## Experimental Targets

### Current 5 targets (act5, results available)

| Target | OpenML ID | Label | Sources found | Best discovery score | Notes |
|---|---|---|---|---|---|
| adult | 1590 | income >$50k | 256 | 0.630 | Best repurposing signal |
| diabetes | 37 | diabetes positive | 1,037 | 0.345 | Needs re-run with bmi/obesity added back |
| churn | 42178 | customer churn | 1,106 | 0.410 | New target, good signal |
| credit | 31 | credit good/bad | 666 | 0.544 | Moderate signal |
| bank | 1461 | term deposit | 770 | 0.212 | Hard — weak discovery scores |

### Planned targets (not yet run)

| Target | OpenML ID | Label | Positive value |
|---|---|---|---|
| nyhouse | — | price >$1M | — |
| heart | 53 | heart disease | `present` |
| turnover | 43551 | employee turnover | `Left` |

---

## Current Results (act5, AUC — calibrated threshold)

### AUC by level

| Target | Baseline | L0 | L1 | L2 | L3 | L5 | Oracle |
|---|---|---|---|---|---|---|---|
| **adult** | 0.456 | 0.720 | 0.454 | **0.737** | 0.457 | 0.409 | 0.926 |
| **diabetes** | 0.562 | 0.349 | **0.591** | 0.349 | 0.469 | 0.426 | 0.829 |
| **churn** | 0.439 | 0.531 | 0.461 | 0.531 | 0.452 | **0.621** | 0.843 |
| **credit** | 0.442 | 0.451 | 0.433 | 0.451 | 0.436 | **0.565** | 0.764 |
| **bank** | 0.411 | **0.482** | 0.414 | 0.416 | 0.433 | 0.431 | 0.929 |

### % of oracle gap closed (AUC)

| Target | L0 | L1 | L2 | L3 | L5 |
|---|---|---|---|---|---|
| **adult** | 56% | 0% | **60%** | 0% | -10% |
| **diabetes** | -80% | **11%** | -80% | -35% | -51% |
| **churn** | 23% | 6% | 23% | 3% | **45%** |
| **credit** | 3% | -3% | 3% | -2% | **38%** |
| **bank** | **14%** | 1% | 1% | 4% | 4% |

### Accuracy by level

| Target | Baseline | L0 | L1 | L2 | L3 | L5 | Oracle |
|---|---|---|---|---|---|---|---|
| **adult** | 0.643 | 0.697 | 0.648 | **0.722** | 0.642 | 0.599 | 0.868 |
| **diabetes** | 0.558 | 0.422 | **0.565** | 0.422 | 0.539 | 0.526 | 0.753 |
| **churn** | 0.594 | 0.619 | 0.601 | 0.619 | 0.610 | **0.632** | 0.785 |
| **credit** | 0.580 | **0.600** | 0.560 | **0.600** | 0.570 | 0.595 | 0.740 |
| **bank** | **0.792** | 0.760 | 0.780 | 0.788 | 0.793 | 0.792 | 0.905 |

### Key findings
- **L2 (pseudo-labeling)** best for adult — works when source quality is high
- **L5 (DANN)** best for churn and credit — most consistent positive performer
- **L1/L3** rarely help — close <11% of oracle gap in all cases
- **Diabetes** results are invalid — concept list missing bmi/obesity proxy labels; re-run pending
- **Bank** is genuinely hard — best discovery score only 0.212; imbalanced label makes accuracy misleading (use AUC)
- **Negative transfer confirmed**: equal-weight baseline AUC below 0.5 on adult and bank

---

## Source Repurposing — Concept Lists

Concepts are pre-generated by Claude Opus and cached in `data/llm_expansion_cache.json`.
**Rule:** include synonyms + proxy labels (columns that COULD BE the label); exclude predictive features.

| Target label | n_concepts | Example concepts |
|---|---|---|
| income above 50k | 36 | salary, wages, annual_income, household_income, IncomeLevel |
| diabetes diagnosis positive | 47 | diabetes, glucose, hba1c, fasting_glucose, bmi, obesity, waist_cm |
| customer churn | 35 | churn, churned, attrition, cancelled, Exited, subscription_status |
| credit risk good or bad | 35 | credit_risk, default, credit_score, loan_status, good_bad, fico |
| term deposit subscription | 30 | deposit, term_deposit, subscribed, campaign_outcome, conversion |
| heart disease diagnosis | 30 | heart_disease, cad, cvd, troponin, cholesterol, ejection_fraction |
| employee turnover | 30 | turnover, attrition, resigned, terminated, Attrition, LeaveOrNot |
| house price above 1 million | 35 | price, sale_price, property_value, SalePrice, appraised_value |

---

## Contamination Rules

Name patterns excluded from lake (applied at both manifest-build and load time):
- `adult`, `census`, `credit` (substring), `creditability`, `diabetes`, `pima`, `bank`, `marketing`, `fraud`

Note: GitTables lake has no name-based contamination filter in `gittables_lake.py` — only act4 OpenML lake applies these. Contamination for act5 targets is mitigated by the repurposing threshold and discovery scoring.

---

## Technology Stack

| Purpose | Library |
|---|---|
| Data manipulation | `pandas` |
| ML & metrics | `scikit-learn` |
| Column embeddings | `sentence-transformers` |
| Gradient boosting | `xgboost` |
| Hungarian matching | `scipy` |
| Deep learning (Level 5 only) | `torch` (CUDA — RTX 4060) |
| LLM concept expansion | Ollama + Qwen3.5 (local, no API cost) |
| HTTP downloads | `requests` |
| Python version | 3.10+ |

---

## Constraints

- **No neural network training** (except Level 5). `sentence-transformers` is used as a frozen encoder only.
- **No external labeled datasets.** All labeled signal must come from source repurposing within the lake.
- One module per pipeline step — `table_discovery`, `schema_alignment`, `domain_adaptation`, `evaluation`.
- Keep functions pure and stateless; avoid global state.
- Concept expansion must use LLM output — no manual curation of concept lists.

---

## Code Conventions

- Type hints on all function signatures.
- Each module has a small, explicit public API; internals prefixed with `_`.
- `pathlib.Path` for all file paths.
- `logging` module only — no `print` statements.
- Notebooks are exploratory scratch space, not deliverables.

---

## Directory Layout

```
uda-datalake/
├── CLAUDE.md
├── act4_openml_lake.py          # OpenML lake experiment (completed reference)
├── act5_gittables_lake.py       # GitTables lake experiment (main)
├── gittables_lake.py            # GitTables downloader + loader
├── table_discovery.py           # Step 0+1 (source repurposing + table discovery)
├── schema_alignment.py          # Step 2
├── domain_adaptation.py         # Step 3
├── evaluation.py                # Step 4
├── data/
│   ├── gittables/               # GitTables parquet cache + manifest.json
│   │   ├── manifest.json
│   │   └── gt_*.parquet
│   ├── llm_expansion_cache.json # Pre-generated concept lists (all 8 targets)
│   ├── stream_ckpt_*.json       # Repurposing scan checkpoints (auto-deleted on completion)
│   └── NY-Housing/
├── logs/
│   └── act5_*.log               # Per-target run logs
└── results/
    ├── act4/                    # OpenML lake results per target
    └── act5/                    # GitTables lake results per target
        ├── adult/
        ├── diabetes/
        ├── churn/
        ├── credit/
        └── bank/
```

---

## Running the Pipeline

```bash
# Download GitTables lake (resumable, run until complete):
python gittables_lake.py --download-zenodo

# Check download progress:
python gittables_lake.py --stats

# Run GitTables experiment (act5) — targets run sequentially:
python act5_gittables_lake.py --target adult
python act5_gittables_lake.py --target diabetes
python act5_gittables_lake.py --target churn
python act5_gittables_lake.py --target credit
python act5_gittables_lake.py --target bank

# Run all 5 sequentially in one command:
python act5_gittables_lake.py --target adult ; python act5_gittables_lake.py --target diabetes ; python act5_gittables_lake.py --target churn ; python act5_gittables_lake.py --target credit ; python act5_gittables_lake.py --target bank

# Run OpenML lake experiment (reference, all targets):
python act4_openml_lake.py --lake-size 787 --target adult
```

---

## Pending

1. **Re-run diabetes** — concept list updated to include bmi, obesity, waist_cm proxy labels
2. **Run heart, turnover** — new targets, concept lists ready in cache
3. **Algorithm analysis** — once diabetes re-run complete, compare L2 vs L5 as primary method
4. **Contamination** — add name-based filtering to GitTables lake for new targets (telco, heart_statlog)
