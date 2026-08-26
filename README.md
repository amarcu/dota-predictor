# Dota 2 Match Outcome Predictor

A PyTorch LSTM that predicts the winner of a Dota 2 match from the
minute-by-minute game state, producing a Radiant win probability at every
minute of the game. The repository contains the full pipeline: an OpenDota
data collector, feature construction, training, evaluation and calibration,
a pre-trained checkpoint, an offline predictor for any parsed match, and a
live terminal dashboard fed by Dota 2's Game State Integration (GSI).

Personal side project. Python 3.10+, PyTorch, NumPy, scikit-learn, SQLite.

## At a glance

- **Model** — 2-layer LSTM (128 hidden units) over 20 per-minute features,
  combined with learned 32-dimensional embeddings of all 10 hero picks;
  244,866 trainable parameters.
- **Target** — Radiant win probability at *each* minute. Training uses a
  masked per-timestep binary cross-entropy, so one network serves early-,
  mid- and late-game predictions without retraining per cut-off.
- **Data** — parsed matches from the OpenDota API (per-player gold/XP/last-hit
  time series plus kill and objective logs), cached in SQLite and processed
  into NumPy arrays.
- **Shipped checkpoint** — `models/checkpoints/model.pt`; its embedded
  training history records **94.1% validation accuracy at the final minute**
  (see [Shipped model](#shipped-model) for exactly what is and isn't recorded).
- **Inference** — predict a historical match by ID, run a simulated demo, or
  attach to a live game through GSI. An optional Polymarket integration shows
  market odds next to the model's probability while spectating pro matches.

The setup follows Akhmedov & Phan, *Machine learning models for DOTA 2
outcomes prediction* ([arXiv:2106.01782](https://arxiv.org/abs/2106.01782)).

## How it works

### Data source

Match data comes from the [OpenDota API](https://docs.opendota.com/). Only
*parsed* matches are usable, because only those carry the per-minute
`gold_t`, `xp_t` and `lh_t` arrays for each player. `scripts/fetch_data.py`
collects match IDs from `/proMatches` (default) or `/parsedMatches`, fetches
each match from `/matches/{id}` (through an `aiohttp` client with a
concurrency cap for large batches), and keeps a match only if it has
time-series data, exactly 10 players and a duration of at least 10 minutes.
Matches are stored in a SQLite database (`data/matches.db`) that deduplicates
by match ID, so collection can be resumed across sessions.

The client rate-limits itself to 30 requests/minute without an API key and
3,000/minute with one (`OPENDOTA_API_KEY` in `.env`).

### Input features

`scripts/process_data.py` turns each match into a `(60, F)` array — one row
per game minute, zero-padded to 60 minutes with a validity mask — plus the 10
hero IDs and the label (`1` = Radiant win). An 80/20 train/validation split is
made with a fixed seed.

The basic 8 features are team sums of the OpenDota time series. The
`--enhanced-features` flag (the configuration the shipped model uses) adds 12
more, reconstructed per minute from the match's `kills_log` and `objectives`
events:

| Group | Features |
| --- | --- |
| Economy (8) | `radiant_gold`, `radiant_xp`, `dire_gold`, `dire_xp`, `gold_diff`, `xp_diff`, `radiant_lh`, `dire_lh` |
| Kills (3) | `radiant_kills`, `dire_kills`, `kill_diff` — cumulative |
| Towers (3) | `radiant_towers`, `dire_towers`, `tower_diff` — remaining, starting from 11 |
| Barracks (3) | `radiant_barracks`, `dire_barracks`, `barracks_diff` — remaining, starting from 6 |
| Roshan (3) | `radiant_roshan`, `dire_roshan`, `roshan_diff` — cumulative kills |

Features are fed to the model raw. `process_data.py` also writes z-score
statistics, but neither training nor inference uses them. Implementation:
`Match.get_full_time_series()` in `src/dota_predictor/data/match.py`.

### Model

`LSTMPredictor` in `src/dota_predictor/models/lstm.py`. The shapes below are
those of the shipped checkpoint.

```
hero IDs (10,)  ──► Embedding(146, 32) ──► flatten ──► (320,)
                                                         │  broadcast to every minute
features (60, 20) ──► LSTM(20 → 128, 2 layers, dropout 0.3) ──► (60, 128)
                                                         │
                                       concat ──► (60, 448) ──► Linear(448 → 1) ──► sigmoid
                                                         │
                                                         ▼
                                    Radiant win probability at each minute (60,)
```

Hero picks are learned embeddings rather than one-hot vectors; the vocabulary
is sized from the maximum hero ID in the training data, with index 0 reserved
for padding. The per-minute head is a single linear layer. The module also has
a deeper head (`448 → 64 → 32 → 1`) on the final hidden state for whole-match
classification, but every training and inference script uses the per-minute
output.

### Training

`scripts/train.py` with `Trainer` from `src/dota_predictor/utils/training.py`:

- Binary cross-entropy at every timestep, with the match label broadcast
  across the sequence and padded minutes masked out — the model learns to
  predict the final result from the state at each minute.
- Adam (learning rate 1e-3, weight decay 1e-5), batch size 32,
  `ReduceLROnPlateau` (halves the rate after 5 stagnant epochs), early
  stopping after 10 epochs without validation-loss improvement, at most
  50 epochs.
- The checkpoint is rewritten whenever validation loss improves. Accuracy is
  measured at each match's last valid minute.
- Runs on CUDA, Apple MPS or CPU, whichever is available.

### Calibration

Raw sigmoid outputs are not guaranteed to be calibrated probabilities.
`scripts/calibrate_per_minute.py` fits isotonic-regression calibrators on the
validation split for three game phases (minutes 1–10, 11–25 and 26+) and saves
them as JSON lookup tables in `models/checkpoints/`. `PhaseCalibrator` applies
the right one for the current minute, and `LivePredictor` accepts a calibrator
path — but the shipped entry points run uncalibrated; see
[Project status](#project-status).

## Shipped model

`models/checkpoints/model.pt` is a standard PyTorch checkpoint holding the
state dict, optimiser and scheduler state, and the per-epoch training history.
Reading it back gives:

- **Architecture**: 20 input features, LSTM 128 units × 2 layers, hero
  embedding 146 × 32, 244,866 trainable parameters.
- **Saved at** epoch 46 with validation loss 0.563 — the run's best.
- **Validation accuracy at the final minute: 94.11%** (training accuracy
  94.19% at the same epoch).
- Learning rate at save time 2.5e-4, i.e. reduced twice by the scheduler.

The checkpoint does not record which dataset it was trained on. The committed
exploration notebook (`notebooks/data_exploration.ipynb`) shows the processed
dataset at 39,812 training and 9,952 validation matches with the 20-feature
set, which is consistent with the checkpoint's input size.

The validation loss looks high for a 94%-accurate classifier because it is
averaged over *all* minutes of every match, including the opening minutes
where the outcome is close to a coin flip. Per-minute accuracy and calibration
metrics (Brier score, expected calibration error) are computed by
`scripts/evaluate.py` and `scripts/calibrate_per_minute.py`, but their output
goes to the git-ignored `experiments/` directory, so no numbers beyond the
checkpoint's own history are reproducible from this repository alone.

## Installation

Requires Python 3.10 or newer.

```bash
git clone https://github.com/amarcu/dota-predictor.git
cd dota-predictor
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"                             # or: make install
cp .env.example .env                                # optional: set OPENDOTA_API_KEY
```

An OpenDota API key is optional; it only raises the rate limit for data
collection. Keys are issued at <https://www.opendota.com/api-keys>.

## Usage

### Predict a historical match

Fetches a match from OpenDota and shows how the model's prediction evolves. It
works with the shipped checkpoint and no API key; the match must be parsed on
OpenDota.

```bash
python scripts/predict_match.py --match-id 8610327187
```

The script prints both lineups, the win probability at 5-minute intervals
(computed by zeroing out everything after that minute, so no future
information leaks in), and the final prediction against the actual result.
Output for the match above, a 35-minute Radiant win — note the model moving
to Radiant at minute 15 while Radiant is still slightly behind in gold:

```
Min 05 | Gold:     -903 | [██████████████████░░░░░░░░░░░░░░░░░░░░░░] | ⚪ 45.9%
Min 10 | Gold:   -1,438 | [█████████████████░░░░░░░░░░░░░░░░░░░░░░░] | 🔴 43.8%
Min 15 | Gold:     -177 | [██████████████████████░░░░░░░░░░░░░░░░░░] | 🟢 56.5%
Min 20 | Gold:     -462 | [███████████████████████████░░░░░░░░░░░░░] | 🟢 69.9%
Min 25 | Gold:   +3,080 | [███████████████████████████░░░░░░░░░░░░░] | 🟢 69.2%
Min 30 | Gold:   +5,499 | [█████████████████████████████░░░░░░░░░░░] | 🟢 73.5%
Min 35 | Gold:  +19,128 | [█████████████████████████████████████░░░] | 🟢 92.7%

Final Prediction: Radiant (92.7%)
Actual Result:    Radiant
```

### Run the demo (no Dota 2 required)

```bash
python scripts/live_predict.py --demo     # or: make demo
```

Simulates a 30-minute match in which Radiant builds an early lead and Dire
comes back, printing the probability minute by minute. No lineup is set, so
the hero branch sees only the padding embedding and the trajectory reflects
the economy features alone.

### Retrain from scratch

```bash
python scripts/fetch_data.py --count 1000 --db            # OpenDota → data/matches.db
python scripts/process_data.py --db --enhanced-features   # → data/processed/*.npy
python scripts/train.py                                   # → models/checkpoints/model.pt
python scripts/evaluate.py --model models/checkpoints/model.pt --log --calibration
python scripts/calibrate_per_minute.py                    # → calibrator_{early,mid,late}.json
```

`make all` runs process → train → evaluate and `make help` lists every target.
Collecting tens of thousands of matches needs an API key (`make fetch-large`
targets 50k).

### Live prediction from a running game (GSI)

Dota 2 can POST its game state to a local HTTP endpoint. The predictor runs
that endpoint, records one snapshot per game minute, and renders a terminal
dashboard with the current probability, an ASCII probability-over-time graph
and the score.

1. Copy `gamestate_integration_predictor.cfg` into your Dota 2 install at
   `.../steamapps/common/dota 2 beta/game/dota/cfg/gamestate_integration/`
   (create the folder if it doesn't exist).
   `python scripts/live_predict.py --generate-config` regenerates the file for
   a different port or auth token.
2. Add `-gamestateintegration` to Dota 2's launch options in Steam.
3. Start the server, then launch Dota 2 and play or spectate a match:

   ```bash
   python scripts/live_predict.py --port 3000    # or: make live
   ```

Payloads are processed only while the map state is
`DOTA_GAMERULES_STATE_GAME_IN_PROGRESS`, and `live_predict.py` checks each
payload's auth token against the config (`dota_predictor_secret` by default).
Both the spectator payload layout (`player.team2` / `player.team3`) and the
playing layout (`allplayers`) are handled. `scripts/gsi_diagnostic.py` logs
raw GSI payloads to a JSON file so live features can be compared with
OpenDota's values after the game.

### Spectate pro matches with Polymarket odds

`scripts/find_games.py` cross-references Polymarket's Dota 2 markets (Gamma
API, no credentials) with OpenDota's `/live` endpoint by team name and lists
games that are live or upcoming, with current market odds. `scripts/spectate.py`
picks a live one (or takes a `--slug`), opens Dota 2 through
`steam://run/570//+watch_server <id>`, copies the `watch_server` console
command to the clipboard, and starts the GSI dashboard with the market odds
displayed next to the model's probability.

```bash
python scripts/find_games.py --live
python scripts/spectate.py --slug dota2-l1ga-vpp-2025-12-22
```

Markets exist only for tournament matches, not public matchmaking. If
`gamma-api.polymarket.com` is geo-blocked where you are, `.env.example` shows
how to route only the Polymarket requests through a local SOCKS proxy
(`POLYMARKET_PROXY=socks5h://...`, e.g. Cloudflare WARP's local proxy) while
Dota 2 keeps its direct connection; the client also falls back to
DNS-over-HTTPS when system DNS can't resolve the host. The `PolymarketClient`
class also contains order-placement methods (behind the optional
`py-clob-client` package and wallet environment variables), but no script in
this repository calls them.

## Project structure

```
src/dota_predictor/
├── api/opendota.py         Sync and async OpenDota clients with rate limiting
├── data/                   Match/MatchPlayer dataclasses and feature construction,
│                           SQLite store, hero ID table, a PyTorch Dataset wrapper
├── models/                 LSTMPredictor, checkpoint loader, baseline models
├── features/extractor.py   Feature-extraction helper for the 8-feature path
├── utils/                  Config and .env loading, Trainer
├── evaluation/             Brier / log-loss / ECE metrics, isotonic and temperature
│                           calibration, JSON experiment tracker
├── inference/              LivePredictor (GSI payload → features) and the GSI HTTP
│                           server with terminal dashboard
└── polymarket/             Gamma/CLOB client and Polymarket ↔ OpenDota match linker
scripts/                    CLI entry points: fetch, process, train, evaluate, calibrate,
                            predict, live, find_games, spectate, diagnostics
models/checkpoints/         Pre-trained model.pt and per-phase isotonic calibrators
docs/                       features.md, data_pipeline.md, polymarket_integration.md
notebooks/                  data_exploration.ipynb (outputs committed)
data/examples/              Abbreviated sample match JSON and a match summary CSV
tests/                      pytest suite for the data models
```

## Development

```bash
make test        # pytest tests/
make lint        # ruff check src/ scripts/  (ruff is not in the dev extras; install it separately)
make typecheck   # mypy src/
```

## Project status

A working personal project at alpha quality
(`Development Status :: 3 - Alpha` in `pyproject.toml`), published as a single
release commit in March 2026. Everything described above runs from the code in
this repository, with these gaps worth knowing about:

- **Evaluation artifacts aren't committed.** The only recorded metric is the
  validation accuracy in the checkpoint's history; Brier/ECE and per-minute
  accuracy have to be regenerated with `scripts/evaluate.py` after collecting
  data again.
- **Calibration is shipped but switched off.** `live_predict.py` deliberately
  passes no calibrator (a code comment notes that raw outputs tracked Dota
  Plus's in-game estimate more closely), and `predict_match.py` never loads
  one, so the three calibrator files are unused by the entry points.
- **Live mode sees fewer signals than training.** From GSI the predictor fills
  gold (net worth), XP (approximated as XP-per-minute × elapsed minutes), last
  hits, kills and heroes. Tower, barracks and Roshan features stay at their
  starting values because the GSI config doesn't request building data, so
  live inputs are a subset of what the model was trained on.
- **Hero vocabulary is frozen by the training data.** The embedding covers hero
  IDs up to 145; a lineup containing a newer, higher-ID hero will fail lookup
  until the model is retrained on matches that include it.
- **Tests cover the data models only** (`tests/test_data_models.py`, 9 passing
  tests); the model, training loop and GSI parsing are untested.
- `LogisticRegressionBaseline`, `SimpleNNBaseline`, `LSTMWithAttention`,
  `FeatureExtractor` and `DotaDataset` exist in the package but no script uses
  them; the pipeline goes through `process_data.py`'s NumPy arrays instead.

## References

- Akhmedov, K., & Phan, A. H. (2021). *Machine learning models for DOTA 2
  outcomes prediction*. [arXiv:2106.01782](https://arxiv.org/abs/2106.01782)
- [OpenDota API](https://docs.opendota.com/)
- [Dota 2 Game State Integration](https://developer.valvesoftware.com/wiki/Dota_2_Workshop_Tools/Dota_2_Game_State_Integration)
- [Polymarket](https://polymarket.com/)

## License

MIT — see [LICENSE](LICENSE).
