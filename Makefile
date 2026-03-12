# Dota Predictor - Development Workflow
# =====================================
#
# Development Loop:
#   1. make fetch        - Fetch new match data from OpenDota
#   2. make process      - Process raw JSON into training arrays
#   3. make train        - Train the model
#   4. make evaluate     - Evaluate and log results
#   5. make compare      - Compare experiments
#   6. make live         - Run live prediction server
#
# Quick Commands:
#   make all            - Full pipeline (process → train → evaluate)
#   make retrain        - Retrain with existing data
#   make test-live      - Run a demo without Dota 2

.PHONY: all install fetch process train evaluate compare live demo test lint format clean help spectate find-games

# Default Python
PYTHON ?= python3

# Directories
DATA_DIR = data
MODELS_DIR = models/checkpoints
EXPERIMENTS_DIR = experiments

# Colors for output
GREEN = \033[0;32m
YELLOW = \033[0;33m
BLUE = \033[0;34m
RED = \033[0;31m
NC = \033[0m # No Color

#------------------------------------------------------------------------------
# Main Targets
#------------------------------------------------------------------------------

all: process train evaluate
	@echo "$(GREEN)✅ Full pipeline complete!$(NC)"

retrain: train evaluate
	@echo "$(GREEN)✅ Retrain complete!$(NC)"

#------------------------------------------------------------------------------
# Setup
#------------------------------------------------------------------------------

install:
	@echo "$(BLUE)📦 Installing dependencies...$(NC)"
	pip install -e ".[dev]"

#------------------------------------------------------------------------------
# Data Pipeline
#------------------------------------------------------------------------------

fetch:
	@echo "$(BLUE)📥 Fetching match data from OpenDota → database...$(NC)"
	@echo "$(YELLOW)⚠️  This uses API calls - check your quota!$(NC)"
	$(PYTHON) scripts/fetch_data.py --count 1000 --db

fetch-large:
	@echo "$(BLUE)📥 Fetching large dataset (50k matches) → database...$(NC)"
	@echo "$(YELLOW)⚠️  This will cost ~$10 in API calls!$(NC)"
	$(PYTHON) scripts/fetch_data.py --count 50000 --db

process:
	@echo "$(BLUE)🔄 Processing matches from database...$(NC)"
	$(PYTHON) scripts/process_data.py --db --enhanced-features
	@echo "$(GREEN)✅ Data processed to $(DATA_DIR)/processed/$(NC)"

process-json:
	@echo "$(BLUE)🔄 Processing raw JSON data (legacy)...$(NC)"
	$(PYTHON) scripts/process_data.py --input data/raw/matches_50k.json --enhanced-features

inspect:
	@echo "$(BLUE)🔍 Inspecting dataset...$(NC)"
	$(PYTHON) scripts/inspect_data.py

#------------------------------------------------------------------------------
# Training
#------------------------------------------------------------------------------

train:
	@echo "$(BLUE)🧠 Training model...$(NC)"
	$(PYTHON) scripts/train.py
	@echo "$(GREEN)✅ Model saved to $(MODELS_DIR)/model.pt$(NC)"

train-quick:
	@echo "$(BLUE)🧠 Quick training (10 epochs)...$(NC)"
	$(PYTHON) scripts/train.py --epochs 10

#------------------------------------------------------------------------------
# Evaluation & Experiments
#------------------------------------------------------------------------------

evaluate:
	@echo "$(BLUE)📊 Evaluating model...$(NC)"
	$(PYTHON) scripts/evaluate.py --model $(MODELS_DIR)/model.pt --log

evaluate-calibrate:
	@echo "$(BLUE)📊 Evaluating model and building calibrators...$(NC)"
	$(PYTHON) scripts/evaluate.py --model $(MODELS_DIR)/model.pt --log
	$(PYTHON) scripts/calibrate_per_minute.py

calibrate-phases:
	@echo "$(BLUE)📊 Building per-phase calibrators for live prediction...$(NC)"
	$(PYTHON) scripts/calibrate_per_minute.py

#------------------------------------------------------------------------------
# Polymarket / Live Spectating
#------------------------------------------------------------------------------

find-games:
	@echo "$(BLUE)Finding live/upcoming games on Polymarket + OpenDota...$(NC)"
	$(PYTHON) scripts/find_games.py

find-games-live:
	@echo "$(BLUE)Finding LIVE games only...$(NC)"
	$(PYTHON) scripts/find_games.py --live

spectate:
	@echo "$(BLUE)Auto-finding a live game and starting GSI predictor...$(NC)"
	$(PYTHON) scripts/spectate.py

spectate-slug:
	@echo "$(BLUE)Starting spectator for a specific game...$(NC)"
	@echo "$(YELLOW)Usage: make spectate-slug SLUG=dota2-l1ga-vpp-2025-12-22$(NC)"
	$(PYTHON) scripts/spectate.py --slug $(SLUG)

compare:
	@echo "$(BLUE)📈 Comparing experiments...$(NC)"
	$(PYTHON) scripts/evaluate.py --compare

best:
	@echo "$(BLUE)🏆 Finding best experiment...$(NC)"
	$(PYTHON) scripts/evaluate.py --best

summary:
	@echo "$(BLUE)📋 Experiment summary...$(NC)"
	$(PYTHON) scripts/evaluate.py --summary

#------------------------------------------------------------------------------
# Live Prediction
#------------------------------------------------------------------------------

live:
	@echo "$(BLUE)🎮 Starting live prediction server on port 3000...$(NC)"
	@echo "$(YELLOW)Make sure Dota 2 GSI is configured!$(NC)"
	$(PYTHON) scripts/live_predict.py --port 3000

demo:
	@echo "$(BLUE)🎬 Running prediction demo...$(NC)"
	$(PYTHON) scripts/live_predict.py --demo

gsi-debug:
	@echo "$(BLUE)🔍 Running GSI diagnostic tool...$(NC)"
	$(PYTHON) scripts/gsi_diagnostic.py --port 3000

#------------------------------------------------------------------------------
# Code Quality
#------------------------------------------------------------------------------

test:
	@echo "$(BLUE)🧪 Running tests...$(NC)"
	pytest tests/ -v

lint:
	@echo "$(BLUE)🔎 Linting code...$(NC)"
	ruff check src/ scripts/

format:
	@echo "$(BLUE)✨ Formatting code...$(NC)"
	ruff format src/ scripts/

typecheck:
	@echo "$(BLUE)📝 Type checking...$(NC)"
	mypy src/

#------------------------------------------------------------------------------
# Cleanup
#------------------------------------------------------------------------------

clean:
	@echo "$(YELLOW)🧹 Cleaning temporary files...$(NC)"
	rm -rf __pycache__ .pytest_cache .mypy_cache
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -delete

clean-data:
	@echo "$(YELLOW)🧹 Cleaning processed data (keeps raw)...$(NC)"
	rm -rf $(DATA_DIR)/processed/*.npy
	rm -rf $(DATA_DIR)/processed/*.npz

clean-experiments:
	@echo "$(YELLOW)🧹 Cleaning experiments...$(NC)"
	rm -rf $(EXPERIMENTS_DIR)/*

#------------------------------------------------------------------------------
# Help
#------------------------------------------------------------------------------

help:
	@echo ""
	@echo "$(GREEN)Dota Predictor - Development Workflow$(NC)"
	@echo "======================================="
	@echo ""
	@echo "$(BLUE)📋 DEVELOPMENT LOOP:$(NC)"
	@echo "  1. make fetch        Fetch matches from OpenDota → database"
	@echo "  2. make process      Process database → .npy training arrays"
	@echo "  3. make train        Train the LSTM model"
	@echo "  4. make evaluate     Evaluate and log to experiment tracker"
	@echo "  5. make compare      Compare all experiments"
	@echo "  6. make live         Start live prediction server"
	@echo ""
	@echo "$(BLUE)⚡ QUICK COMMANDS:$(NC)"
	@echo "  make all             Full pipeline (process → train → evaluate)"
	@echo "  make retrain         Retrain and evaluate with existing data"
	@echo "  make demo            Run demo without Dota 2"
	@echo ""
	@echo "$(BLUE)📊 EXPERIMENTS:$(NC)"
	@echo "  make evaluate        Evaluate current model"
	@echo "  make compare         Show experiment comparison table"
	@echo "  make best            Show best performing experiment"
	@echo "  make summary         Show experiment summary"
	@echo ""
	@echo "$(BLUE)POLYMARKET / SPECTATING:$(NC)"
	@echo "  make find-games               Find live/upcoming games on Polymarket"
	@echo "  make find-games-live          Find only currently live games"
	@echo "  make spectate                 Auto-find game, open Dota 2, start GSI"
	@echo "  make spectate-slug SLUG=...   Spectate a specific game by slug"
	@echo ""
	@echo "$(BLUE)CODE QUALITY:$(NC)"
	@echo "  make test            Run pytest"
	@echo "  make lint            Run ruff linter"
	@echo "  make format          Format code with ruff"
	@echo "  make typecheck       Run mypy type checker"
	@echo ""
	@echo "$(BLUE)CLEANUP:$(NC)"
	@echo "  make clean           Remove temp files"
	@echo "  make clean-data      Remove processed data (keeps raw)"
	@echo "  make clean-experiments  Remove all experiments"
	@echo ""

