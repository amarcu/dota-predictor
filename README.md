# Dota 2 Match Outcome Predictor

An LSTM-based neural network that predicts Dota 2 match outcomes in real time
by analysing the minute-by-minute evolution of game state. Trained on 50,000+
matches from the OpenDota API.

Based on the research paper
[*Machine learning models for DOTA 2 outcomes prediction*](https://arxiv.org/abs/2106.01782)
(Akhmedov & Phan, arXiv:2106.01782).

---

## Results

| Metric | Value |
|--------|-------|
| End-of-game accuracy | **94.11%** |
| Brier score | **0.052** |
| Expected calibration error | **0.076** |

Accuracy by game time:

| Minute | Accuracy |
|--------|----------|
| 5 | 49.3% (near random) |
| 10 | 52.1% |
| 15 | 62.2% |
| 20 | 71.1% |
| 30 | 73.7% |
| End | 94.11% |

---

## Project Structure

```
dota-predictor/
├── src/dota_predictor/
│   ├── api/           # Async OpenDota API client
│   ├── data/          # Match / MatchPlayer dataclasses, DotaDataset, SQLite DB
│   ├── models/        # LSTMPredictor, baseline models
│   ├── features/      # Feature extraction (8 and 20 feature variants)
│   ├── evaluation/    # Metrics, isotonic calibration, experiment tracker
│   ├── inference/     # GSI server, live predictor
│   ├── polymarket/    # Polymarket API client + Dota 2 market finder
│   └── utils/         # Config, training loop utilities
├── scripts/
│   ├── fetch_data.py           # Collect matches from OpenDota → SQLite
│   ├── process_data.py         # Transform matches → .npy training arrays
│   ├── train.py                # Train the LSTM model
│   ├── evaluate.py             # Evaluate with metrics, calibration, plots
│   ├── calibrate_per_minute.py # Fit per-phase isotonic calibrators
│   ├── predict_match.py        # Predict a historical match by ID
│   ├── live_predict.py         # Real-time GSI prediction dashboard
│   ├── find_games.py           # Show live/upcoming games with Polymarket odds
│   ├── spectate.py             # Find a game on Polymarket + launch Dota 2
│   ├── inspect_data.py         # Inspect processed .npy files
│   └── gsi_diagnostic.py       # Log raw GSI values for validation
├── docs/
│   ├── features.md                  # Feature engineering reference
│   ├── data_pipeline.md             # Full data flow documentation
│   └── polymarket_integration.md    # Polymarket API reference
├── models/checkpoints/
│   ├── model.pt                # Pre-trained LSTM (94.11% accuracy)
│   └── calibrator*.json        # Isotonic calibration files
├── tests/
├── notebooks/
├── data/examples/              # Sample match data
├── gamestate_integration_predictor.cfg
├── .env.example
├── Makefile
├── pyproject.toml
└── requirements.txt
```

---

## Installation

**Prerequisites**: Python 3.10+

```bash
git clone https://github.com/amarcu/dota-predictor.git
cd dota-predictor

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -e ".[dev]"
```

A pre-trained model (`models/checkpoints/model.pt`) is included in the repo,
so you can run predictions immediately without training from scratch.

Copy `.env.example` to `.env` and optionally add your OpenDota API key
(increases rate limits from ~30/min to 3000/min):

```bash
cp .env.example .env
# Edit .env and set OPENDOTA_API_KEY
# Get a key at: https://www.opendota.com/api-keys
```

---

## Quickstart

### 1. Fetch training data

```bash
# Fetch 1000 matches into the local SQLite database
python scripts/fetch_data.py --count 1000 --db
```

### 2. Process into training arrays

```bash
# Convert to .npy arrays (uses enhanced 20-feature set)
python scripts/process_data.py --db --enhanced-features
```

### 3. Train the model

```bash
python scripts/train.py
# Saves best checkpoint to models/checkpoints/model.pt
```

### 4. Evaluate

```bash
python scripts/evaluate.py --model models/checkpoints/model.pt --log

# Then fit calibrators for live prediction
python scripts/calibrate_per_minute.py
```

### 5. Predict a historical match

```bash
python scripts/predict_match.py --match-id 7892631234
```

Or use `make` to run the full pipeline:

```bash
make all       # process → train → evaluate
make live      # start the GSI prediction server
```

---

## Live Prediction (GSI)

The live predictor connects to Dota 2 via
[Game State Integration (GSI)](https://developer.valvesoftware.com/wiki/Dota_2_Workshop_Tools/Dota_2_Game_State_Integration)
and shows a real-time win-probability dashboard in your terminal.

### Setup

1. Copy the GSI config to your Dota 2 config directory:

   **macOS**:
   ```bash
   cp gamestate_integration_predictor.cfg \
     "$HOME/Library/Application Support/Steam/steamapps/common/dota 2 beta/game/dota/cfg/"
   ```

   **Windows**:
   ```
   Copy gamestate_integration_predictor.cfg to:
   C:\Program Files (x86)\Steam\steamapps\common\dota 2 beta\game\dota\cfg\
   ```

2. Add `-gamestateintegration` to Dota 2 launch options in Steam.

3. Start the prediction server:

   ```bash
   python scripts/live_predict.py --port 3000
   # or: make live
   ```

4. Open Dota 2 and start or spectate a match. The dashboard updates every minute
   with the current win probability.

### Spectating with Polymarket odds

To find a live pro match on Polymarket and launch Dota 2 alongside the predictor:

```bash
# List live and upcoming games with current market odds
python scripts/find_games.py

# Auto-find a live game, open Dota 2, and start the predictor
python scripts/spectate.py

# Spectate a specific game
python scripts/spectate.py --slug dota2-l1ga-vpp-2025-12-22

# Makefile shortcuts
make find-games
make spectate
make spectate-slug SLUG=dota2-l1ga-vpp-2025-12-22
```

For GSI diagnostics (validate that live features match training features):

```bash
python scripts/gsi_diagnostic.py --port 3000
```

---

## Model Architecture

```
Input: time-series (60 timesteps × 20 features)
       hero IDs (10 heroes)

Hero branch:
  Embedding (vocab = max_hero_id + 1, 32 dim) × 10 heroes
  → team averages + concatenate → hero_features (320,)

Sequence branch:
  LSTM (20 → 128 hidden, 2 layers, dropout=0.3)
  → output at each timestep (60, 128)

Combined (per timestep):
  concat(lstm_out, hero_features) → (448,)
  Linear 448→64 + ReLU + dropout
  Linear 64→1 + Sigmoid
  → win probability at each minute (60,)
```

Training uses binary cross-entropy masked to valid (non-padded) timesteps,
with early stopping on validation loss.

**Calibration**: Raw model outputs are passed through isotonic regression
calibrators fitted per game phase (early/mid/late) to produce well-calibrated
probabilities. See [docs/data_pipeline.md](docs/data_pipeline.md#step-6-calibration) for details.

---

## Documentation

- [**Feature Engineering**](docs/features.md) — full description of the 8 and 20
  feature sets, hero embeddings, and GSI alignment
- [**Data Pipeline**](docs/data_pipeline.md) — step-by-step walkthrough from
  API to trained model, including design decisions and calibration

---

## Development

```bash
make test       # pytest
make lint       # ruff check
make format     # ruff format
make typecheck  # mypy
```

---

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgments

- [OpenDota](https://www.opendota.com/) for the free, comprehensive API
- Akhmedov & Phan for the reference paper ([arXiv:2106.01782](https://arxiv.org/abs/2106.01782))
- [Polymarket](https://polymarket.com/) for the prediction market data
