# Data Pipeline

This document walks through the full data flow from the OpenDota API to a
trained model, explaining each step and the design decisions behind it.

---

## Overview

```
OpenDota API
  └─ /parsedMatches or /proMatches      [scripts/fetch_data.py]
       └─ SQLite: data/matches.db        [src/dota_predictor/data/database.py]
            └─ Match / MatchPlayer        [src/dota_predictor/data/match.py]
                 └─ get_full_time_series() → numpy arrays
                      └─ .npy files        [scripts/process_data.py]
                           └─ DotaDataset  [src/dota_predictor/data/dataset.py]
                                └─ LSTMPredictor  [src/dota_predictor/models/lstm.py]
```

---

## Step 1: Fetching Match Data (`fetch_data.py`)

Match data is fetched from the [OpenDota API](https://docs.opendota.com/).

```bash
# Fetch 1000 matches and store in the SQLite database
python scripts/fetch_data.py --count 1000 --db

# Fetch pro matches specifically (higher parse rate)
python scripts/fetch_data.py --count 1000 --db --source pro
```

### Why parsed matches?

OpenDota distinguishes between **basic** and **parsed** matches:

- **Basic matches**: Only contain aggregate stats (kills, deaths, GPM, XPM).
  The per-minute time-series arrays (`gold_t`, `xp_t`, `lh_t`) are absent.
- **Parsed matches**: The replay has been processed by OpenDota's parser,
  producing the full minute-by-minute time series needed for training.

The `--source parsed` flag uses the `/parsedMatches` endpoint, which guarantees
time-series data. Pro matches (`/proMatches`) also have a higher parse rate
because they are parsed automatically by default.

When a match without time-series data is encountered during processing,
it is silently skipped.

### Match filtering criteria

A match is included in the training set only if it meets all of:

| Criterion | Reason |
|-----------|--------|
| `has_time_series == True` | Need `gold_t`, `xp_t`, `lh_t` arrays |
| Exactly 10 players | Incomplete match data is unreliable |
| Duration > 10 minutes | Very short games are outliers (e.g., abandoned) |
| Game mode: All Pick, Ranked, Turbo (configurable) | Consistent meta |

### Rate limiting

- Without an API key: ~30–60 requests/minute
- With an API key: 3000 requests/minute

Set `OPENDOTA_API_KEY` in your `.env` file (see `.env.example`). The async
client (`OpenDotaClient`) automatically respects these limits.

---

## Step 2: Data Storage (`database.py`)

Fetched matches are stored in a SQLite database at `data/matches.db`.

The database serves as a local cache: re-running `fetch_data.py` deduplicates
matches automatically by `match_id`. This allows incremental data collection
across multiple sessions without re-downloading matches.

Key tables:
- `matches`: One row per match (metadata + outcome label)
- `players`: One row per player per match (stats + time-series JSON)
- `objectives`: Objective events per match (towers, barracks, Roshan)

---

## Step 3: Processing into Training Arrays (`process_data.py`)

Converts the stored matches into numpy arrays optimised for fast loading
during training.

```bash
# Process with enhanced features (recommended)
python scripts/process_data.py --db --enhanced-features

# Process a raw JSON file instead of the database
python scripts/process_data.py --input data/raw/matches.json --enhanced-features
```

### What gets produced

All files are written to `data/processed/`:

| File | Shape | Description |
|------|-------|-------------|
| `train_features.npy` | `(N_train, 60, F)` | Time-series features, 60 timesteps, F features (8 or 20) |
| `train_heroes.npy` | `(N_train, 10)` | Hero IDs [5 Radiant + 5 Dire] |
| `train_labels.npy` | `(N_train,)` | 1.0 = Radiant win, 0.0 = Dire win |
| `train_masks.npy` | `(N_train, 60)` | 1.0 for valid timesteps, 0.0 for padding |
| `val_features.npy` | `(N_val, 60, F)` | Validation features |
| `val_heroes.npy` | `(N_val, 10)` | Validation hero IDs |
| `val_labels.npy` | `(N_val,)` | Validation labels |
| `val_masks.npy` | `(N_val, 60)` | Validation masks |
| `normalization.npz` | — | Per-feature mean and std (computed but not used in training) |

### Train/validation split

An 80/20 random split is applied at processing time. The split is consistent
across runs because it uses a fixed random seed.

### Padding

Matches have variable durations (typically 20–60 minutes). All sequences are
zero-padded to 60 timesteps. The validity mask records which timesteps are
real data vs. padding, and is passed to the model to exclude padded timesteps
from the loss calculation.

---

## Step 4: Dataset (`dataset.py`)

`DotaDataset` wraps the processed `.npy` files as a PyTorch `Dataset`.

```python
from dota_predictor.data.dataset import DotaDataset

dataset = DotaDataset("data/processed/", split="train")
features, heroes, label, mask = dataset[0]
# features: Tensor(60, 20)
# heroes:   Tensor(10,)   — hero IDs for embedding lookup
# label:    Tensor(1,)    — 0.0 or 1.0
# mask:     Tensor(60,)   — valid timestep mask
```

The model is trained on raw (unnormalized) features. While `normalization.npz`
is generated by `process_data.py`, it is not used during training or inference.

---

## Step 5: Model Training (`train.py`)

```bash
python scripts/train.py
```

The `LSTMPredictor` architecture:

```
Hero IDs (10,)
  └─ Embedding (140 → 32) × 10 heroes
       └─ Concatenate → hero_features (320,)

Time-series (60, 20)
  └─ LSTM (20 → 128, 2 layers, dropout=0.3)
       └─ Output sequence (60, 128)
            └─ concat hero_features at each step → (60, 448)
                 └─ Linear (448 → 64) + ReLU
                      └─ Linear (64 → 1) + Sigmoid
                           └─ Win probability at each minute
```

**Loss**: Binary cross-entropy on valid timesteps only (masked).

**Early stopping**: Training halts if validation loss does not improve for 10
consecutive epochs. Best checkpoint is saved to `models/checkpoints/model.pt`.

---

## Step 6: Calibration (`calibrate_per_minute.py`)

The raw model outputs are **not well-calibrated**: a predicted probability of
0.7 does not actually mean Radiant wins 70% of the time. The model's outputs
are compressed toward 0.5 especially in the early game.

Isotonic regression calibrators are fitted per game phase:

```bash
python scripts/calibrate_per_minute.py
```

This creates three phase-specific calibrators saved to `models/checkpoints/`:
- `calibrator_early.json` — minutes 1–10
- `calibrator_mid.json` — minutes 11–25
- `calibrator_late.json` — minutes 26+

`LivePredictor` and `predict_match.py` load these automatically via `calibrator_config.json`.

**Calibration results (on 50k matches)**:

| Metric | Before | After |
|--------|--------|-------|
| Brier score | 0.165 | 0.052 |
| ECE | 0.27 | 0.076 |

---

## Step 7: Evaluation (`evaluate.py`)

```bash
python scripts/evaluate.py --model models/checkpoints/model.pt --log
```

Produces:
- Per-minute accuracy table
- Calibration curve plot
- Experiment log entry in `experiments/`

**Results on 50k match dataset**:

| Game time | Accuracy |
|-----------|----------|
| 5 min | 49.3% (near random) |
| 10 min | 52.1% |
| 15 min | 62.2% |
| 20 min | 71.1% |
| 30 min | 73.7% |
| End of game | **94.11%** |

---

## Live Inference (`live_predict.py`)

At inference time, the `LivePredictor` receives per-minute snapshots from the
Dota 2 Game State Integration (GSI) server and runs the model on the
accumulated time series.

The GSI feature extraction maps live game state to the same feature vector
used during training. See [features.md](features.md#gsi-feature-alignment) for
the exact mapping.

```bash
# Start the GSI server (default port 3000)
python scripts/live_predict.py --port 3000
```

The terminal dashboard updates every minute with the current win probability,
calibrated to reflect true historical frequencies.
