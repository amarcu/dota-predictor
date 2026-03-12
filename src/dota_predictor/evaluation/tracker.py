"""
Experiment tracking for model development.

This provides a lightweight alternative to MLflow for tracking experiments,
storing metrics, and comparing model versions.
"""

from __future__ import annotations

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any
from dataclasses import dataclass, asdict


@dataclass
class Experiment:
    """A single experiment/model run."""
    
    id: str
    name: str
    timestamp: str
    
    # Model configuration
    model_config: dict[str, Any]
    
    # Training configuration
    training_config: dict[str, Any]
    
    # Data configuration
    data_config: dict[str, Any]
    
    # Metrics
    metrics: dict[str, float]
    
    # Path to model checkpoint
    model_path: str | None = None
    
    # Notes
    notes: str = ""
    
    # Tags for filtering
    tags: list[str] | None = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class ExperimentTracker:
    """
    Track and compare ML experiments.
    
    Usage:
        tracker = ExperimentTracker("experiments/")
        
        # Log an experiment
        exp_id = tracker.log_experiment(
            name="baseline_lstm",
            model_config={"hidden_size": 128, "num_layers": 2},
            training_config={"epochs": 100, "lr": 0.001},
            data_config={"n_matches": 50000, "enhanced_features": True},
            metrics={"brier_score": 0.18, "accuracy": 0.72},
            model_path="models/checkpoints/model.pt",
            notes="First baseline with enhanced features",
            tags=["baseline", "lstm"]
        )
        
        # Compare experiments
        tracker.compare(["exp_abc123", "exp_def456"])
        
        # Get best experiment by metric
        best = tracker.get_best("brier_score", lower_is_better=True)
    """
    
    def __init__(self, experiments_dir: str | Path = "experiments"):
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.experiments_dir / "index.json"
        self._load_index()
    
    def _load_index(self) -> None:
        """Load the experiments index."""
        if self.index_path.exists():
            with open(self.index_path) as f:
                self._index = json.load(f)
        else:
            self._index = {"experiments": []}
    
    def _save_index(self) -> None:
        """Save the experiments index."""
        with open(self.index_path, "w") as f:
            json.dump(self._index, f, indent=2)
    
    def _generate_id(self, name: str) -> str:
        """Generate a unique experiment ID."""
        timestamp = datetime.now().isoformat()
        hash_input = f"{name}_{timestamp}"
        short_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        return f"exp_{short_hash}"
    
    def log_experiment(
        self,
        name: str,
        model_config: dict[str, Any],
        training_config: dict[str, Any],
        data_config: dict[str, Any],
        metrics: dict[str, float],
        model_path: str | None = None,
        notes: str = "",
        tags: list[str] | None = None,
    ) -> str:
        """
        Log a new experiment.
        
        Returns:
            The experiment ID
        """
        exp_id = self._generate_id(name)
        timestamp = datetime.now().isoformat()
        
        experiment = Experiment(
            id=exp_id,
            name=name,
            timestamp=timestamp,
            model_config=model_config,
            training_config=training_config,
            data_config=data_config,
            metrics=metrics,
            model_path=model_path,
            notes=notes,
            tags=tags or [],
        )
        
        # Save experiment file
        exp_path = self.experiments_dir / f"{exp_id}.json"
        with open(exp_path, "w") as f:
            json.dump(asdict(experiment), f, indent=2)
        
        # Update index
        self._index["experiments"].append({
            "id": exp_id,
            "name": name,
            "timestamp": timestamp,
            "primary_metric": metrics.get("brier_score"),
            "tags": tags or [],
        })
        self._save_index()
        
        print(f"✅ Logged experiment: {exp_id}")
        print(f"   Name: {name}")
        brier = metrics.get("brier_score")
        acc = metrics.get("accuracy")
        print(f"   Brier Score: {brier:.4f}" if brier is not None else "   Brier Score: N/A")
        print(f"   Accuracy: {acc:.2%}" if acc is not None else "   Accuracy: N/A")
        
        return exp_id
    
    def get_experiment(self, exp_id: str) -> Experiment | None:
        """Load an experiment by ID."""
        exp_path = self.experiments_dir / f"{exp_id}.json"
        if not exp_path.exists():
            return None
        
        with open(exp_path) as f:
            data = json.load(f)
        
        return Experiment(**data)
    
    def list_experiments(
        self, 
        tags: list[str] | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """List experiments, optionally filtered by tags."""
        experiments = self._index["experiments"]
        
        if tags:
            experiments = [
                e for e in experiments
                if any(t in e.get("tags", []) for t in tags)
            ]
        
        # Sort by timestamp (newest first)
        experiments = sorted(
            experiments, 
            key=lambda x: x["timestamp"], 
            reverse=True
        )
        
        return experiments[:limit]
    
    def get_best(
        self, 
        metric: str = "brier_score",
        lower_is_better: bool = True,
        tags: list[str] | None = None,
    ) -> Experiment | None:
        """Get the best experiment by a metric."""
        experiments = self.list_experiments(tags=tags, limit=1000)
        
        best_exp = None
        best_value = float("inf") if lower_is_better else float("-inf")
        
        for exp_info in experiments:
            exp = self.get_experiment(exp_info["id"])
            if exp is None:
                continue
            
            value = exp.metrics.get(metric)
            if value is None:
                continue
            
            if lower_is_better and value < best_value:
                best_value = value
                best_exp = exp
            elif not lower_is_better and value > best_value:
                best_value = value
                best_exp = exp
        
        return best_exp
    
    def compare(
        self, 
        exp_ids: list[str] | None = None,
        metrics: list[str] | None = None,
    ) -> str:
        """
        Compare experiments and return a formatted table.
        
        Args:
            exp_ids: List of experiment IDs to compare (default: last 5)
            metrics: List of metrics to show (default: all)
            
        Returns:
            Formatted comparison table
        """
        if exp_ids is None:
            recent = self.list_experiments(limit=5)
            exp_ids = [e["id"] for e in recent]
        
        if not exp_ids:
            return "No experiments to compare."
        
        experiments = [self.get_experiment(eid) for eid in exp_ids]
        experiments = [e for e in experiments if e is not None]
        
        if not experiments:
            return "No valid experiments found."
        
        if metrics is None:
            metrics = ["brier_score", "log_loss", "accuracy", "ece"]
        
        # Build comparison table
        lines = []
        lines.append("=" * 80)
        lines.append("EXPERIMENT COMPARISON")
        lines.append("=" * 80)
        
        # Header
        header = f"{'Metric':<20}"
        for exp in experiments:
            header += f" {exp.name[:12]:<12}"
        lines.append(header)
        lines.append("-" * 80)
        
        # Metrics rows
        for metric in metrics:
            row = f"{metric:<20}"
            values = []
            for exp in experiments:
                val = exp.metrics.get(metric)
                if val is not None:
                    values.append(val)
                    row += f" {val:<12.4f}"
                else:
                    row += f" {'N/A':<12}"
            
            # Highlight best value
            lines.append(row)
        
        lines.append("-" * 80)
        
        # Show timestamps
        lines.append(f"{'Timestamp':<20}")
        for exp in experiments:
            ts = exp.timestamp[:10]  # Just date
            lines.append(f"  {exp.name[:12]}: {ts}")
        
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def summary(self) -> str:
        """Get a summary of all experiments."""
        experiments = self.list_experiments(limit=100)
        
        if not experiments:
            return "No experiments logged yet."
        
        lines = []
        lines.append("=" * 70)
        lines.append("EXPERIMENT SUMMARY")
        lines.append("=" * 70)
        lines.append(f"Total experiments: {len(experiments)}")
        lines.append("")
        
        # Best by Brier Score
        best = self.get_best("brier_score", lower_is_better=True)
        if best:
            lines.append(f"🏆 Best Brier Score: {best.metrics['brier_score']:.4f}")
            lines.append(f"   Model: {best.name} ({best.id})")
        
        lines.append("")
        lines.append("Recent experiments:")
        lines.append("-" * 70)
        lines.append(f"{'ID':<15} {'Name':<20} {'Brier':<10} {'Accuracy':<10} {'Date':<12}")
        lines.append("-" * 70)
        
        for exp_info in experiments[:10]:
            exp = self.get_experiment(exp_info["id"])
            if exp:
                brier = exp.metrics.get("brier_score", 0)
                acc = exp.metrics.get("accuracy", 0)
                date = exp.timestamp[:10]
                lines.append(f"{exp.id:<15} {exp.name[:20]:<20} {brier:<10.4f} {acc:<10.2%} {date:<12}")
        
        lines.append("=" * 70)
        
        return "\n".join(lines)


