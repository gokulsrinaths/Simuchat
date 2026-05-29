"""
MetricsTracker: Per-episode and aggregate metrics for SocialConsensusEnv.
Supports CSV export, JSON export, and matplotlib plotting.
"""
import csv
import json
import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional


class MetricsTracker:
    """
    Tracks training and evaluation metrics across episodes.

    Per-episode metrics tracked:
    - consensus_reached (bool)
    - average_trust (float)
    - polarization_score (float)
    - episode_reward (float)
    - time_to_consensus (int or None): steps until consensus, if reached
    - n_rounds (int): total rounds in episode
    - duration (float): wall-clock seconds for episode
    - timestamp (str): ISO 8601 timestamp

    Aggregate summary:
    - total_episodes
    - consensus_rate (%)
    - mean_reward / std_reward
    - mean_trust / mean_polarization
    - mean_rounds
    - mean_time_to_consensus (only for episodes where consensus was reached)
    """

    def __init__(self):
        self.episodes: List[Dict[str, Any]] = []
        self._start_time = time.time()

    def record_episode(
        self,
        consensus_reached: bool,
        average_trust: float,
        polarization_score: float,
        episode_reward: float,
        time_to_consensus: Optional[int],
        n_rounds: int,
        duration: float,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record metrics for one completed episode.

        Args:
            consensus_reached: whether consensus was achieved
            average_trust: mean off-diagonal trust at episode end
            polarization_score: std of agreement scores at episode end
            episode_reward: total reward accumulated
            time_to_consensus: number of steps until consensus (None if not reached)
            n_rounds: number of rounds completed
            duration: wall-clock duration in seconds
            extra: optional dict of additional metrics
        """
        entry = {
            "episode": len(self.episodes),
            "consensus_reached": bool(consensus_reached),
            "average_trust": float(average_trust),
            "polarization_score": float(polarization_score),
            "episode_reward": float(episode_reward),
            "time_to_consensus": int(time_to_consensus) if time_to_consensus is not None else None,
            "n_rounds": int(n_rounds),
            "duration": float(duration),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if extra:
            entry.update(extra)
        self.episodes.append(entry)

    def get_summary(self) -> Dict[str, Any]:
        """
        Compute and return aggregate summary statistics.

        Returns:
            dict with aggregate metrics
        """
        if not self.episodes:
            return {
                "total_episodes": 0,
                "consensus_rate": 0.0,
                "mean_reward": 0.0,
                "std_reward": 0.0,
                "mean_trust": 0.0,
                "mean_polarization": 0.0,
                "mean_rounds": 0.0,
                "mean_time_to_consensus": None,
            }

        total = len(self.episodes)
        consensus_eps = [e for e in self.episodes if e["consensus_reached"]]
        consensus_rate = 100.0 * len(consensus_eps) / total

        rewards = [e["episode_reward"] for e in self.episodes]
        trusts = [e["average_trust"] for e in self.episodes]
        polarizations = [e["polarization_score"] for e in self.episodes]
        rounds = [e["n_rounds"] for e in self.episodes]

        ttc_values = [
            e["time_to_consensus"]
            for e in self.episodes
            if e["time_to_consensus"] is not None
        ]

        import statistics

        def safe_mean(lst):
            return sum(lst) / len(lst) if lst else 0.0

        def safe_std(lst):
            if len(lst) < 2:
                return 0.0
            mean = safe_mean(lst)
            variance = sum((x - mean) ** 2 for x in lst) / (len(lst) - 1)
            return variance ** 0.5

        return {
            "total_episodes": total,
            "consensus_rate": round(consensus_rate, 2),
            "mean_reward": round(safe_mean(rewards), 4),
            "std_reward": round(safe_std(rewards), 4),
            "min_reward": round(min(rewards), 4),
            "max_reward": round(max(rewards), 4),
            "mean_trust": round(safe_mean(trusts), 4),
            "mean_polarization": round(safe_mean(polarizations), 4),
            "mean_rounds": round(safe_mean(rounds), 2),
            "mean_time_to_consensus": (
                round(safe_mean(ttc_values), 2) if ttc_values else None
            ),
            "n_consensus_episodes": len(consensus_eps),
            "total_wall_time_s": round(time.time() - self._start_time, 2),
        }

    def to_csv(self, filepath: str) -> None:
        """
        Write all episode records to a CSV file.

        Args:
            filepath: path to output CSV file
        """
        if not self.episodes:
            return

        fieldnames = list(self.episodes[0].keys())

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for episode in self.episodes:
                writer.writerow(episode)

    def to_json(self, filepath: str) -> None:
        """
        Write summary and all episode records to a JSON file.

        Args:
            filepath: path to output JSON file
        """
        output = {
            "summary": self.get_summary(),
            "episodes": self.episodes,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, default=str)

    def plot_learning_curve(self, filepath: str, window: int = 10) -> None:
        """
        Plot episode rewards and rolling mean to a PNG file.

        Args:
            filepath: path to output PNG file
            window: rolling average window size (default 10)
        """
        try:
            import matplotlib
            matplotlib.use("Agg")  # Non-interactive backend
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            print("[MetricsTracker] matplotlib not available. Skipping plot.")
            return

        if not self.episodes:
            print("[MetricsTracker] No episodes recorded. Skipping plot.")
            return

        rewards = [e["episode_reward"] for e in self.episodes]
        episodes_x = list(range(1, len(rewards) + 1))

        # Compute rolling mean
        rolling_mean = []
        for i in range(len(rewards)):
            start = max(0, i - window + 1)
            rolling_mean.append(float(np.mean(rewards[start:i + 1])))

        consensus_episodes = [
            i + 1 for i, e in enumerate(self.episodes) if e["consensus_reached"]
        ]

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Plot 1: Episode rewards + rolling mean
        ax1 = axes[0]
        ax1.plot(episodes_x, rewards, alpha=0.3, color="steelblue", linewidth=0.8, label="Episode Reward")
        ax1.plot(episodes_x, rolling_mean, color="darkorange", linewidth=2.0, label=f"Rolling Mean (w={window})")

        if consensus_episodes:
            consensus_rewards = [rewards[i - 1] for i in consensus_episodes]
            ax1.scatter(
                consensus_episodes, consensus_rewards,
                color="green", s=20, alpha=0.5, zorder=5, label="Consensus Reached"
            )

        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Reward")
        ax1.set_title("SocialConsensusEnv — Learning Curve")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)

        # Plot 2: Trust and polarization over time
        ax2 = axes[1]
        trusts = [e["average_trust"] for e in self.episodes]
        polarizations = [e["polarization_score"] for e in self.episodes]

        ax2.plot(episodes_x, trusts, color="green", linewidth=1.0, alpha=0.7, label="Avg Trust")
        ax2.plot(episodes_x, polarizations, color="red", linewidth=1.0, alpha=0.7, label="Polarization")
        ax2.axhline(y=0.75, color="green", linestyle="--", alpha=0.5, linewidth=0.8, label="Consensus Threshold")

        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Score")
        ax2.set_title("Trust and Polarization Over Training")
        ax2.legend(loc="upper left")
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[MetricsTracker] Learning curve saved to: {filepath}")

    def get_recent_episodes(self, n: int = 10) -> List[Dict[str, Any]]:
        """Return the last n episodes."""
        return self.episodes[-n:] if len(self.episodes) >= n else list(self.episodes)

    def clear(self) -> None:
        """Clear all recorded episodes."""
        self.episodes = []
        self._start_time = time.time()

    def __len__(self) -> int:
        return len(self.episodes)

    def __repr__(self) -> str:
        summary = self.get_summary()
        return (
            f"MetricsTracker("
            f"episodes={summary['total_episodes']}, "
            f"consensus_rate={summary['consensus_rate']:.1f}%, "
            f"mean_reward={summary['mean_reward']:.3f})"
        )
