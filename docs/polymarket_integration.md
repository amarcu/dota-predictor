# Polymarket Integration

This document covers how the project reads Dota 2 market data from Polymarket
to display live odds alongside the model's predictions.

---

## Overview

Polymarket is a decentralised prediction market on Polygon. Two APIs are used:

1. **Gamma API** (`gamma-api.polymarket.com`) — fetches markets, events, and
   current odds (no authentication required)
2. **CLOB API** (`clob.polymarket.com`) — provides live order book data

The integration is read-only: no trading or order placement is performed.

---

## Fetching Dota 2 Markets (Gamma API)

```python
import requests

GAMMA_API = "https://gamma-api.polymarket.com"

# Fetch all active events
def get_active_events(limit=100):
    resp = requests.get(
        f"{GAMMA_API}/events",
        params={"limit": limit, "closed": "false", "order": "id", "ascending": "false"}
    )
    return resp.json()

# Fetch event by slug (from the Polymarket URL)
# URL: https://polymarket.com/sports/dota2/games/week/1/dota2-ty-ts8-2025-12-21
#                                                      ↑ slug
def get_event_by_slug(slug):
    resp = requests.get(f"{GAMMA_API}/events/slug/{slug}")
    return resp.json()

# Fetch markets by tag (Dota 2 has its own sports tag)
def get_markets_by_tag(tag_id, limit=50):
    resp = requests.get(
        f"{GAMMA_API}/markets",
        params={"tag_id": tag_id, "limit": limit, "closed": "false"}
    )
    return resp.json()
```

### Finding the Dota 2 tag

```python
def find_dota_markets():
    sports = requests.get("https://gamma-api.polymarket.com/sports").json()
    dota_tag = next((s for s in sports if "dota" in s.get("label", "").lower()), None)

    if dota_tag:
        tag_id = dota_tag.get("id")
        return requests.get(
            "https://gamma-api.polymarket.com/markets",
            params={"tag_id": tag_id, "closed": "false"}
        ).json()
    return []
```

### Reading odds from a live event

```python
event = get_event_by_slug("dota2-ty-ts8-2025-12-21")

if event:
    print(f"Match: {event['title']}")
    for market in event.get('markets', []):
        print(f"  {market['question']}")
        for token in market.get('tokens', []):
            print(f"    {token['outcome']}: {float(token['price'])*100:.1f}%")
```

### Key concepts

- **Token ID**: Long numeric string identifying each outcome (YES/NO per market)
- **Price**: 0.01–0.99, representing the implied probability
- **Volume**: Total USDC traded on a market — a proxy for liquidity

---

## Linking Polymarket Markets to Live Dota 2 Matches

`MatchLinker` cross-references Polymarket events with the OpenDota `/live`
endpoint to find the Dota 2 match corresponding to a given market.

```python
from dota_predictor.polymarket import MatchLinker

linker = MatchLinker()

# Find all currently live/upcoming matchable games
games = linker.find_matchable_games(limit=20)

for game in games:
    print(game.pm_title, game.pm_odds, game.dota_match)
```

Or use the `find_games.py` script:

```bash
python scripts/find_games.py          # all upcoming games
python scripts/find_games.py --live   # live games only
```

---

## Watching a Live Game

Once a match is linked, `spectate.py` can open Dota 2 directly to that game's
server, and display the GSI predictor alongside the current Polymarket odds:

```bash
python scripts/spectate.py --slug dota2-l1ga-vpp-2025-12-22
```

See the [README](../README.md#spectating-with-polymarket-odds) for the full
usage guide.

---

## Spectating Dota 2 Matches Programmatically

Dota 2 does not support direct match ID spectating from the command line.
The `spectate.py` script uses the `server_steam_id` returned by the
OpenDota `/live` endpoint to construct a Steam URL:

```python
import subprocess

server_steam_id = "90133068541296641"  # from OpenDota /live

# macOS
subprocess.Popen(["open", f"steam://run/570//+watch_server%20{server_steam_id}"])
# Windows
subprocess.Popen(["start", f"steam://run/570//+watch_server%20{server_steam_id}"], shell=True)
```

After Dota 2 opens, paste the `watch_server <id>` command into the in-game
console (`~`). The command is automatically copied to your clipboard.

### Recommended Dota 2 launch options for lower resource usage

```
-novid -nojoy -noaafonts -softparticlesdefaultoff
```

---

## Notes

- Polymarket Dota 2 markets are only available for major tournament matches,
  not public matchmaking games.
- Markets may close before the match ends; the `MatchLinker` handles this case
  gracefully.
- No API credentials are required for read-only market data access.
