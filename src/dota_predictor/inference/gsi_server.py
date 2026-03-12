"""
Game State Integration (GSI) Server for Dota 2.

This module provides a local HTTP server that receives real-time
game state updates from Dota 2 via the GSI feature.

To use GSI, you need to:
1. Create a config file in your Dota 2 game directory:
   Steam/steamapps/common/dota 2 beta/game/dota/cfg/gamestate_integration/

2. Name it something like: gamestate_integration_predictor.cfg

3. Contents should be:
   "Dota 2 Predictor Integration"
   {
       "uri"           "http://localhost:3000/"
       "timeout"       "5.0"
       "buffer"        "0.1"
       "throttle"      "0.1"
       "heartbeat"     "30.0"
       "data"
       {
           "provider"      "1"
           "map"           "1"
           "player"        "1"
           "hero"          "1"
           "abilities"     "1"
           "items"         "1"
       }
       "auth"
       {
           "token"         "your_secret_token"
       }
   }
"""

from __future__ import annotations

import json
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Callable

from dota_predictor.inference.predictor import LivePredictor


# ANSI escape codes for terminal control
CLEAR_SCREEN = "\033[2J"
CURSOR_HOME = "\033[H"
HIDE_CURSOR = "\033[?25l"
SHOW_CURSOR = "\033[?25h"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
DIM = "\033[2m"
RESET = "\033[0m"
BOLD = "\033[1m"

# Box width (visual characters inside the borders)
BOX_WIDTH = 60

import re
ANSI_PATTERN = re.compile(r'\033\[[0-9;]*m')


def visual_len(s: str) -> int:
    """Calculate visual length of string by stripping ANSI codes and accounting for wide chars."""
    # Strip ANSI codes
    clean = ANSI_PATTERN.sub('', s)
    # Count emojis as 2 chars wide (they render as double-width in most terminals)
    width = 0
    for char in clean:
        # Emoji ranges (simplified - covers common emojis)
        if ord(char) > 0x1F000:
            width += 2
        else:
            width += 1
    return width


def pad_line(content: str, width: int = BOX_WIDTH) -> str:
    """Create a bordered line with proper padding for ANSI content."""
    vis_len = visual_len(content)
    padding = max(0, width - vis_len)
    return f"{CYAN}║{RESET}{content}{' ' * padding}{CYAN}║{RESET}"


class Dashboard:
    """Single-screen dashboard for displaying game predictions."""
    
    # Graph settings
    GRAPH_WIDTH = 50  # Number of data points to show
    GRAPH_HEIGHT = 10  # Number of rows for the graph
    
    def __init__(
        self,
        match_id: str | None = None,
        market_odds: dict[str, float] | None = None,
        spectate_cmd: str | None = None,
    ) -> None:
        self.events_received = 0
        self.last_update = time.time()
        self.game_time = 0
        self.game_state = "Waiting for game..."
        self.radiant_score = 0
        self.dire_score = 0
        self.gold_advantage = 0
        self.radiant_prob = 0.5
        self.dire_prob = 0.5
        self.current_minute = 0
        self.prediction = "Even"
        self.confidence = 0.5
        self.last_prediction_minute = -1
        self._lock = threading.Lock()

        # Match info from bot
        self.match_id = match_id
        self.market_odds = market_odds or {}
        self.spectate_cmd = spectate_cmd
        
        # History for the graph (minute -> radiant_prob)
        self.history: dict[int, float] = {}
        
    def update_event(self) -> None:
        """Increment event counter."""
        with self._lock:
            self.events_received += 1
            self.last_update = time.time()
    
    def update_game_state(
        self,
        game_time: int,
        game_state: str,
        radiant_score: int = 0,
        dire_score: int = 0,
    ) -> None:
        """Update basic game state info."""
        with self._lock:
            self.game_time = game_time
            self.game_state = game_state
            self.radiant_score = radiant_score
            self.dire_score = dire_score
            self.current_minute = game_time // 60
    
    def update_prediction(self, summary: dict[str, Any]) -> None:
        """Update prediction data."""
        with self._lock:
            self.radiant_prob = summary["radiant_win_probability"]
            self.dire_prob = summary["dire_win_probability"]
            self.gold_advantage = summary["gold_advantage"]
            self.prediction = summary["prediction"]
            self.confidence = summary["confidence"]
            self.last_prediction_minute = summary["game_minute"]
            
            # Store in history
            minute = summary["game_minute"]
            self.history[minute] = self.radiant_prob
    
    def _render_graph(self) -> list[str]:
        """Render the win probability graph as ASCII art."""
        lines = []
        W = 44  # Graph width (visual characters)
        
        # Need at least 3 data points spanning at least 2 minutes to show a meaningful graph
        if len(self.history) < 3:
            remaining = 3 - len(self.history)
            lines.append(pad_line(""))
            lines.append(pad_line(f"  {DIM}Graph will appear after {remaining} more prediction(s)...{RESET}"))
            return lines
        
        max_min = max(self.history.keys())
        min_min = min(self.history.keys())
        
        # Also skip if range is too small (less than 2 minutes of data)
        if max_min - min_min < 2:
            lines.append(pad_line(""))
            lines.append(pad_line(f"  {DIM}Collecting data... graph needs more game time.{RESET}"))
            return lines
        
        # Title
        lines.append(pad_line(""))
        lines.append(pad_line(f"  {BOLD}WIN PROBABILITY OVER TIME{RESET}"))
        lines.append(pad_line(f"  {GREEN}▲ Radiant{RESET}"))
        
        # 6 rows of graph - row 0=100%, row 2=67%, row 3=50%, row 5=0%
        # Above 50% (rows 0-2): fill green where probability EXCEEDS that row's bottom
        # Below 50% (rows 3-5): fill red where probability DIPS BELOW that row's top
        for row in range(6):
            row_top = 1.0 - (row / 6)       # e.g., row 0 = 100%, row 3 = 50%
            row_bot = 1.0 - ((row + 1) / 6) # e.g., row 0 = 83.3%, row 3 = 33.3%
            
            lbl = "100" if row == 0 else (" 50" if row == 3 else ("  0" if row == 5 else "   "))
            brk = "├" if row == 3 else "│"
            end = "┤" if row == 3 else "│"
            
            chars = []
            for c in range(W):
                m = min_min + (c * (max_min - min_min) // max(1, W - 1)) if max_min > min_min else min_min
                p = self._get_prob_at_minute(m)
                
                # Rows above 50% line (rows 0, 1, 2): show green if prob > 50% and reaches this row
                if row < 3:
                    if p >= row_top:
                        # Probability exceeds this entire row
                        chars.append(f"{GREEN}█{RESET}")
                    elif p > row_bot and p > 0.5:
                        # Probability is within this row (partial fill)
                        chars.append(f"{GREEN}▄{RESET}")
                    else:
                        chars.append(" ")
                # Rows below 50% line (rows 3, 4, 5): show red if prob < 50% and dips this low
                else:
                    if p <= row_bot:
                        # Probability dips below this entire row
                        chars.append(f"{RED}█{RESET}")
                    elif p < row_top and p < 0.5:
                        # Probability is within this row (partial fill)
                        chars.append(f"{RED}▀{RESET}")
                    else:
                        chars.append(" ")
            
            graph = "".join(chars)
            lines.append(pad_line(f" {DIM}{lbl}%{RESET}{brk}{graph}{end}"))
        
        # Bottom label and axis (aligned with graph: 5 chars before data)
        lines.append(pad_line(f"  {RED}▼ Dire{RESET}"))
        lines.append(pad_line(f"     └{'─' * W}┘"))
        lines.append(pad_line(f"     {min_min:<2}{'─' * (W - 4)}min{max_min:>2}"))
        
        return lines
    
    def _get_prob_at_minute(self, minute: int) -> float:
        """Get probability at a minute, interpolating if needed."""
        if minute in self.history:
            return self.history[minute]
        
        if not self.history:
            return 0.5
        
        sorted_mins = sorted(self.history.keys())
        if minute < sorted_mins[0]:
            return self.history[sorted_mins[0]]
        if minute > sorted_mins[-1]:
            return self.history[sorted_mins[-1]]
        
        # Interpolate
        for i in range(len(sorted_mins) - 1):
            if sorted_mins[i] <= minute <= sorted_mins[i + 1]:
                m1, m2 = sorted_mins[i], sorted_mins[i + 1]
                t = (minute - m1) / (m2 - m1) if m2 != m1 else 0
                return self.history[m1] * (1 - t) + self.history[m2] * t
        
        return 0.5
    
    def render(self) -> None:
        """Render the dashboard to terminal."""
        with self._lock:
            lines = []
            
            # Fixed-width box characters
            TOP = f"{BOLD}{CYAN}╔{'═' * BOX_WIDTH}╗{RESET}"
            SEP = f"{CYAN}╠{'═' * BOX_WIDTH}╣{RESET}"
            BOT = f"{CYAN}╚{'═' * BOX_WIDTH}╝{RESET}"
            
            # Header
            lines.append(TOP)
            lines.append(pad_line(f"         🎮 DOTA 2 WIN PREDICTOR - LIVE DASHBOARD"))
            lines.append(SEP)
            
            # Connection status
            seconds_ago = time.time() - self.last_update
            if seconds_ago < 5:
                status = f"{GREEN}● Connected{RESET}"
            elif seconds_ago < 30:
                status = f"{YELLOW}● Idle{RESET}"
            else:
                status = f"{RED}● Disconnected{RESET}"
            
            timestamp = datetime.now().strftime("%H:%M:%S")
            lines.append(pad_line(f"  Status: {status}  |  Events: {self.events_received:,}  |  Time: {timestamp}"))
            lines.append(SEP)
            
            # Spectate hint — shown until game connects
            if self.spectate_cmd and self.events_received == 0:
                lines.append(pad_line(f"  {YELLOW}{BOLD}📺 TO SPECTATE: open Dota 2 console (~) and run:{RESET}"))
                lines.append(pad_line(f"  {BOLD}{self.spectate_cmd}{RESET}"))
                lines.append(pad_line(f"  {DIM}(Command was copied to clipboard){RESET}"))
                lines.append(SEP)

            # Game info
            state_short = self.game_state.replace("DOTA_GAMERULES_STATE_", "")
            minutes = self.game_time // 60
            seconds = self.game_time % 60
            lines.append(pad_line(f"  {BOLD}GAME STATE:{RESET} {state_short}"))
            
            # Game time + score + match ID
            game_info = f"  Game Time: {minutes:02d}:{seconds:02d}  |  Score: {GREEN}{self.radiant_score}{RESET} - {RED}{self.dire_score}{RESET}"
            if self.match_id:
                game_info += f"  |  Match: {self.match_id}"
            lines.append(pad_line(game_info))
            
            # Market odds if available
            if self.market_odds:
                odds_parts = [f"{k}: {v*100:.0f}%" for k, v in self.market_odds.items()]
                lines.append(pad_line(f"  {DIM}Market Odds: {' | '.join(odds_parts)}{RESET}"))
            
            lines.append(SEP)
            
            # Prediction section
            lines.append(pad_line(f"  {BOLD}CURRENT PREDICTION (Minute {self.last_prediction_minute:02d}):{RESET}"))
            lines.append(pad_line(""))
            
            # Progress bar
            bar_length = 40
            radiant_bars = int(self.radiant_prob * bar_length)
            dire_bars = bar_length - radiant_bars
            bar = f"{GREEN}{'█' * radiant_bars}{RED}{'░' * dire_bars}{RESET}"
            lines.append(pad_line(f"  Radiant [{bar}] Dire"))
            lines.append(pad_line(f"    {GREEN}🟢 Radiant: {self.radiant_prob:>6.1%}{RESET}        {RED}🔴 Dire: {self.dire_prob:>6.1%}{RESET}"))
            
            # Gold advantage
            if self.gold_advantage > 0:
                gold_str = f"{GREEN}+{self.gold_advantage:,.0f} Radiant{RESET}"
            elif self.gold_advantage < 0:
                gold_str = f"{RED}{self.gold_advantage:,.0f} Dire{RESET}"
            else:
                gold_str = "Even"
            
            # Prediction summary
            if self.radiant_prob > 0.6:
                color = GREEN
            elif self.radiant_prob < 0.4:
                color = RED
            else:
                color = YELLOW
            lines.append(pad_line(f"    {BOLD}→ {color}{self.prediction}{RESET} | Confidence: {self.confidence:.0%} | Gold: {gold_str}"))
            
            lines.append(SEP)
            
            # Add the graph
            graph_lines = self._render_graph()
            lines.extend(graph_lines)
            
            # Footer
            lines.append(SEP)
            lines.append(pad_line(f"  {DIM}Press Ctrl+C to stop  |  Updates every minute{RESET}"))
            lines.append(BOT)
            
            # Output
            output = CURSOR_HOME + "\n".join(lines) + "\n"
            sys.stdout.write(output)
            sys.stdout.flush()


@dataclass
class GSIConfig:
    """Configuration for the GSI server."""
    
    host: str = "localhost"
    port: int = 3000
    auth_token: str | None = None
    verbose: bool = True
    dashboard: Dashboard | None = None


class GSIHandler(BaseHTTPRequestHandler):
    """HTTP request handler for GSI payloads."""
    
    predictor: LivePredictor | None = None
    config: GSIConfig | None = None
    on_update: Callable[[dict[str, Any], float], None] | None = None
    
    def log_message(self, format: str, *args: Any) -> None:
        """Suppress default HTTP logging."""
        pass  # Silent - dashboard handles display
    
    def do_POST(self) -> None:
        """Handle POST requests from Dota 2 GSI."""
        try:
            # Read the payload
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode("utf-8"))
            
            # Track event
            if self.config.dashboard:
                self.config.dashboard.update_event()
            
            # Validate auth token if configured
            if self.config.auth_token:
                auth = data.get("auth", {}).get("token")
                if auth != self.config.auth_token:
                    self.send_error(401, "Invalid auth token")
                    return
            
            # Process the game state
            if self.predictor:
                try:
                    # Extract game info
                    map_data = data.get("map", {})
                    game_time = map_data.get("clock_time", -1)
                    game_state = map_data.get("game_state", "unknown")
                    radiant_score = map_data.get("radiant_score", 0)
                    dire_score = map_data.get("dire_score", 0)
                    match_id = map_data.get("matchid")  # GSI provides match ID
                    
                    # Update dashboard with game state
                    if self.config.dashboard:
                        # Set match_id from GSI if not already set
                        if match_id and not self.config.dashboard.match_id:
                            self.config.dashboard.match_id = str(match_id)
                        self.config.dashboard.update_game_state(
                            game_time, game_state, radiant_score, dire_score
                        )
                    
                    # Only process during actual gameplay
                    if game_state != "DOTA_GAMERULES_STATE_GAME_IN_PROGRESS":
                        if self.config.dashboard:
                            self.config.dashboard.render()
                        self._send_ok()
                        return
                    
                    updated = self.predictor.update_from_gsi(data)
                    
                    if updated:
                        summary = self.predictor.get_prediction_summary()
                        if self.config.dashboard:
                            self.config.dashboard.update_prediction(summary)
                    
                    # Always render dashboard
                    if self.config.dashboard:
                        self.config.dashboard.render()
                    
                    if updated and self.on_update:
                        prob = self.predictor.predict()
                        self.on_update(data, prob)
                        
                except Exception as e:
                    # Log error without breaking dashboard
                    import traceback
                    print(f"\n❌ Error in GSI handler: {e}")
                    traceback.print_exc()
            
            self._send_ok()
            
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
        except Exception as e:
            self.send_error(500, str(e))
    
    def _send_ok(self) -> None:
        """Send success response."""
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"OK")


class GSIServer:
    """
    Game State Integration server for real-time Dota 2 predictions.
    
    Example usage:
        from dota_predictor.inference import GSIServer, LivePredictor
        
        predictor = LivePredictor("models/checkpoints/model.pt")
        server = GSIServer(predictor, port=3000)
        
        print("Starting GSI server... Waiting for Dota 2 data.")
        server.start()  # Blocks until stopped
    """
    
    def __init__(
        self,
        predictor: LivePredictor,
        host: str = "localhost",
        port: int = 3000,
        auth_token: str | None = None,
        verbose: bool = True,
        on_update: Callable[[dict[str, Any], float], None] | None = None,
        match_id: str | None = None,
        market_odds: dict[str, float] | None = None,
        spectate_cmd: str | None = None,
    ) -> None:
        """
        Initialize the GSI server.

        Args:
            predictor: The LivePredictor instance to use
            host: Host to bind to
            port: Port to listen on
            auth_token: Expected auth token from GSI config
            verbose: Whether to print predictions to console
            on_update: Optional callback for each update (data, probability)
            match_id: Optional match ID to display in dashboard
            market_odds: Optional market odds from Polymarket {team: prob}
            spectate_cmd: watch_server command shown in dashboard until game connects
        """
        self.predictor = predictor
        self.dashboard = (
            Dashboard(match_id=match_id, market_odds=market_odds, spectate_cmd=spectate_cmd)
            if verbose else None
        )
        self.config = GSIConfig(
            host=host,
            port=port,
            auth_token=auth_token,
            verbose=verbose,
            dashboard=self.dashboard,
        )
        self.on_update = on_update
        self.server: HTTPServer | None = None
        self._thread: threading.Thread | None = None
    
    def start(self, blocking: bool = True) -> None:
        """
        Start the GSI server.
        
        Args:
            blocking: If True, blocks until server is stopped.
                     If False, runs in background thread.
        """
        # Configure the handler
        GSIHandler.predictor = self.predictor
        GSIHandler.config = self.config
        GSIHandler.on_update = self.on_update
        
        self.server = HTTPServer(
            (self.config.host, self.config.port),
            GSIHandler,
        )
        
        if self.dashboard:
            # Clear screen and show initial dashboard
            sys.stdout.write(CLEAR_SCREEN + HIDE_CURSOR)
            sys.stdout.flush()
            self.dashboard.render()
        else:
            print(f"🎮 GSI Server starting on http://{self.config.host}:{self.config.port}")
            print("📡 Waiting for Dota 2 game state updates...")
        
        if blocking:
            try:
                self.server.serve_forever()
            except KeyboardInterrupt:
                if self.dashboard:
                    sys.stdout.write(SHOW_CURSOR + "\n")
                    sys.stdout.flush()
                print("\n🛑 Server stopped.")
        else:
            self._thread = threading.Thread(target=self.server.serve_forever)
            self._thread.daemon = True
            self._thread.start()
    
    def stop(self) -> None:
        """Stop the GSI server."""
        if self.server:
            self.server.shutdown()
            if self.dashboard:
                sys.stdout.write(SHOW_CURSOR)
                sys.stdout.flush()
            print("🛑 GSI Server stopped.")
    
    def is_running(self) -> bool:
        """Check if the server is running."""
        return self._thread is not None and self._thread.is_alive()


def create_gsi_config(
    output_path: str,
    port: int = 3000,
    auth_token: str = "dota_predictor_secret",
) -> str:
    """
    Generate a GSI configuration file for Dota 2.
    
    Args:
        output_path: Where to save the config file
        port: Port the server will listen on
        auth_token: Auth token for security
        
    Returns:
        The path to the created config file
    """
    config_content = f'''"Dota 2 Predictor Integration"
{{
    "uri"           "http://localhost:{port}/"
    "timeout"       "5.0"
    "buffer"        "0.1"
    "throttle"      "0.5"
    "heartbeat"     "30.0"
    "data"
    {{
        "provider"      "1"
        "map"           "1"
        "player"        "1"
        "hero"          "1"
        "abilities"     "1"
        "items"         "1"
        "allplayers"    "1"
    }}
    "auth"
    {{
        "token"         "{auth_token}"
    }}
}}
'''
    
    with open(output_path, "w") as f:
        f.write(config_content)
    
    return output_path

