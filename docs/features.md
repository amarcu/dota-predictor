# Feature Engineering

This document describes how raw OpenDota match data is transformed into the
features used by the LSTM model.

---

## Feature Sets

The model supports two feature configurations, selectable via `--enhanced-features`
in `scripts/process_data.py` and `scripts/train.py`.

### Basic Features (8 per timestep)

Derived directly from the per-minute time-series arrays provided by OpenDota.
These are the features described in the reference paper ([arXiv:2106.01782](https://arxiv.org/abs/2106.01782)).

| # | Name | Source field | Description |
|---|------|-------------|-------------|
| 0 | `radiant_gold` | `sum(player.gold_t[t])` for Radiant | Total team gold at minute `t` |
| 1 | `radiant_xp` | `sum(player.xp_t[t])` for Radiant | Total team experience at minute `t` |
| 2 | `dire_gold` | `sum(player.gold_t[t])` for Dire | Total team gold at minute `t` |
| 3 | `dire_xp` | `sum(player.xp_t[t])` for Dire | Total team experience at minute `t` |
| 4 | `gold_diff` | `radiant_gold - dire_gold` | Economic advantage (positive = Radiant ahead) |
| 5 | `xp_diff` | `radiant_xp - dire_xp` | Level advantage (positive = Radiant ahead) |
| 6 | `radiant_lh` | `sum(player.lh_t[t])` for Radiant | Total last hits (CS proxy) |
| 7 | `dire_lh` | `sum(player.lh_t[t])` for Dire | Total last hits (CS proxy) |

> **Note on XP**: The model uses actual `xp_t` values from the API — not the
> commonly used `xp_per_min * minute` approximation, which over-estimates XP
> in the early game.

### Enhanced Features (20 per timestep)

Extends the basic set with kill, objective, and structural state tracking.
These require parsed match data with `objectives` and `kills_log`.

| # | Name | Source | Description |
|---|------|--------|-------------|
| 0–7 | *(basic features)* | — | As above |
| 8 | `radiant_kills` | Reconstructed from `kills_log` | Cumulative Radiant kills at minute `t` |
| 9 | `dire_kills` | Reconstructed from `kills_log` | Cumulative Dire kills at minute `t` |
| 10 | `radiant_towers` | Reconstructed from `objectives` | Towers remaining for Radiant (max 11) |
| 11 | `dire_towers` | Reconstructed from `objectives` | Towers remaining for Dire (max 11) |
| 12 | `radiant_barracks` | Reconstructed from `objectives` | Barracks remaining for Radiant (max 6) |
| 13 | `dire_barracks` | Reconstructed from `objectives` | Barracks remaining for Dire (max 6) |
| 14 | `roshan_kills` | Reconstructed from `objectives` | Total Roshan kills up to minute `t` |
| 15 | `kills_diff` | `radiant_kills - dire_kills` | Kill lead |
| 16 | `towers_diff` | `radiant_towers - dire_towers` | Tower structure advantage |
| 17 | `barracks_diff` | `radiant_barracks - dire_barracks` | Barracks advantage |
| 18 | `structure_score` | weighted tower+barracks sum | Combined structural health metric |
| 19 | `momentum` | rolling kill delta | Short-term kill momentum (last 5 min) |

**Implementation**: `Match.get_full_time_series(enhanced=True)` in
[`src/dota_predictor/data/match.py`](../src/dota_predictor/data/match.py).

---

## Hero Embeddings

Hero picks are encoded as learned embeddings rather than one-hot vectors, which
allows the model to discover latent hero relationships (e.g., synergies and counters).

- **Vocabulary**: `max_hero_id + 1` slots (Dota 2 hero IDs are non-contiguous; slot 0 is reserved for padding)
- **Embedding dimension**: 32
- **Encoding**: Each match contributes 10 hero IDs (5 Radiant + 5 Dire). The embeddings are averaged per team and concatenated with the LSTM hidden state before the final classifier.

Controlled by `use_hero_embedding=True` in `LSTMPredictor`.

---

## Normalization

The model is trained on **raw (unnormalized) features**. While `process_data.py`
computes and saves z-score statistics to `data/processed/normalization.npz`, these
are not loaded during training or inference. The LSTM handles the raw feature
magnitudes directly through its learned gates.

If you wish to experiment with normalization, `DotaDataset` supports a
`normalize=True` option, but the shipped model expects raw features.

---

## Sequence Padding and Masking

Matches have variable durations (typically 20–60 minutes). All sequences are
padded to `max_minutes = 60` with zeros. A binary validity mask is stored
alongside the features:

```
mask[t] = 1.0  if t < actual_match_duration
mask[t] = 0.0  if t >= actual_match_duration (padded)
```

The LSTM processes all 60 timesteps, but the mask is used to select which
timesteps contribute to the training loss (only valid timesteps are included).

---

## GSI Feature Alignment

The live predictor (`LivePredictor`) reconstructs the same features in real
time from Dota 2 Game State Integration (GSI) data. The mapping is:

| Training feature | GSI source | Notes |
|-----------------|-----------|-------|
| `gold_t[t]` | `player.net_worth` | Net worth ≈ current gold value |
| `xp_t[t]` | `player.xp_per_min * elapsed_minutes` | `Hero.Experience` returns 0 in spectator mode |
| `lh_t[t]` | `player.last_hits` | Direct mapping |
| `kills_t[t]` | `player.kills` (cumulative) | Direct mapping |
| Tower states | `buildings` dict from GSI | Mapped to Radiant/Dire tower counts |
| Barracks | `buildings` dict from GSI | Mapped to Radiant/Dire barracks counts |
| Roshan | `roshan_killed` event | Counted from GSI events |

> **Spectator mode caveat**: `Hero.Experience` is always 0 in GSI spectator
> mode. XPM × elapsed time is used as a substitute. This introduces a small
> approximation in the early game (where XP accumulation is non-linear) but
> is accurate from mid-game onward.

---

## Feature Selection Rationale

The chosen features proxy the three primary dimensions of advantage in Dota 2:

1. **Economic** (`gold`, `gold_diff`): Controls itemization and overall power.
   Gold lead is the strongest single predictor of match outcome.

2. **Experience / Levels** (`xp`, `xp_diff`): Determines ability levels and
   stat growth. XP lead correlates strongly with gold lead but is independently
   informative, especially in the early game.

3. **Objectives** (`towers`, `barracks`, `roshan`): Structural control creates
   map pressure and eventual win conditions. Tower count is a lagging indicator
   that confirms momentum shifts already visible in gold/XP.

4. **Last hits** (`lh_t`): A proxy for farming efficiency. Useful for
   distinguishing teams that are winning via objectives vs. farming.

Hero picks are treated separately (embeddings) because their impact is
context-dependent and not expressible as a scalar at any given minute.

**Features deliberately excluded**:
- Deaths / assists (can be derived from kill events but add noise)
- Items (high cardinality, change rapidly, sparse signal per timestep)
- Hero levels (derivable from XP but not directly available in GSI)
- Denies (correlated with last hits, adds minimal signal)
