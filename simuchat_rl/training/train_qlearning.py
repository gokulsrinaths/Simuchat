"""
Tabular Q-Learning baseline for SocialConsensusEnv.
State is discretized from continuous observations.

Usage:
    python train_qlearning.py --episodes 500 --max_rounds 20 --save_dir checkpoints/qlearning
"""
import numpy as np
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.social_consensus_env import SocialConsensusEnv, N_ACTIONS
from metrics.tracker import MetricsTracker

TOPICS = [
    "climate change",
    "artificial intelligence ethics",
    "universal basic income",
    "space exploration",
    "social media regulation",
    "nuclear energy",
    "vaccine mandates",
    "cryptocurrency",
    "remote work",
    "immigration policy",
]


def discretize_observation(obs: dict, n_bins: int = 5) -> tuple:
    """
    Convert continuous observation dict to discrete state tuple.

    State dimensions:
    - trust_mean: mean of all off-diagonal trust values (5 bins)
    - trust_std: std of trust values (5 bins)
    - agreement_mean: mean agreement score (5 bins)
    - agreement_std: std of agreement scores (5 bins)
    - current_round_bin: round // 4, capped at 4 (5 bins)
    - current_agent: agent index 0-2 (3 values)

    Total state space: 5^4 * 5 * 3 = 9375 states
    """
    import numpy as np_

    trust_flat = np_.array(obs["trust_matrix"]).flatten()
    agreement = np_.array(obs["agreement_scores"])

    # Exclude diagonal (self-trust = 1.0) from mean/std
    n = int(np_.sqrt(len(trust_flat)))
    mask = ~np_.eye(n, dtype=bool).flatten()
    off_diag = trust_flat[mask] if np_.any(mask) else trust_flat

    trust_mean = float(np_.mean(off_diag))
    trust_std = float(np_.std(off_diag))
    agreement_mean = float(np_.mean(agreement))
    agreement_std = float(np_.std(agreement))
    current_round_bin = min(int(obs["current_round"]) // 4, 4)
    current_agent = int(obs["current_agent"])

    def bin_value(v: float, n: int = n_bins) -> int:
        return min(int(v * n), n - 1)

    return (
        bin_value(trust_mean),
        bin_value(trust_std),
        bin_value(agreement_mean),
        bin_value(agreement_std),
        current_round_bin,
        current_agent,
    )


class QLearningAgent:
    """
    Tabular Q-Learning agent for SocialConsensusEnv.

    Uses a dictionary-based Q-table (sparse representation) since
    the state space is large but most states may not be visited.
    """

    def __init__(
        self,
        n_actions: int = N_ACTIONS,
        lr: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.05,
    ):
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table: dict = {}
        self.n_updates: int = 0
        self._total_steps: int = 0

    def get_q_values(self, state: tuple) -> np.ndarray:
        """Get or initialize Q-values for a state."""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.n_actions, dtype=np.float64)
        return self.q_table[state]

    def select_action(self, obs: dict) -> int:
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return int(np.random.randint(0, self.n_actions))
        state = discretize_observation(obs)
        q_vals = self.get_q_values(state)
        # Break ties randomly
        max_q = np.max(q_vals)
        best_actions = np.where(q_vals == max_q)[0]
        return int(np.random.choice(best_actions))

    def update(
        self,
        obs: dict,
        action: int,
        reward: float,
        next_obs: dict,
        done: bool,
    ) -> float:
        """
        Apply Q-learning update rule:
            Q(s, a) += lr * (r + gamma * max_a' Q(s', a') - Q(s, a))

        Returns:
            td_error: absolute temporal difference error
        """
        state = discretize_observation(obs)
        next_state = discretize_observation(next_obs)

        q = self.get_q_values(state)
        next_q = self.get_q_values(next_state)

        target = reward + (0.0 if done else self.gamma * float(np.max(next_q)))
        td_error = target - q[action]
        q[action] += self.lr * td_error

        self.n_updates += 1
        self._total_steps += 1

        return abs(td_error)

    def decay_epsilon(self) -> None:
        """Apply epsilon decay after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath: str) -> None:
        """Save Q-table and metadata to JSON."""
        data = {
            "q_table": {str(k): v.tolist() for k, v in self.q_table.items()},
            "epsilon": self.epsilon,
            "n_updates": self.n_updates,
            "total_steps": self._total_steps,
            "hyperparams": {
                "n_actions": self.n_actions,
                "lr": self.lr,
                "gamma": self.gamma,
                "epsilon_min": self.epsilon_min,
                "epsilon_decay": self.epsilon_decay,
            },
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str) -> None:
        """Load Q-table and metadata from JSON."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.q_table = {
            eval(k): np.array(v, dtype=np.float64)
            for k, v in data["q_table"].items()
        }
        self.epsilon = data["epsilon"]
        self.n_updates = data["n_updates"]
        self._total_steps = data.get("total_steps", 0)

    def get_stats(self) -> dict:
        """Return current agent statistics."""
        all_q_vals = []
        for q_arr in self.q_table.values():
            all_q_vals.extend(q_arr.tolist())
        return {
            "n_states_visited": len(self.q_table),
            "n_updates": self.n_updates,
            "epsilon": round(self.epsilon, 4),
            "mean_q": round(float(np.mean(all_q_vals)) if all_q_vals else 0.0, 4),
            "max_q": round(float(np.max(all_q_vals)) if all_q_vals else 0.0, 4),
        }


def train(
    n_episodes: int = 500,
    max_rounds: int = 20,
    save_dir: str = "checkpoints/qlearning",
    eval_every: int = 50,
    render_every: int = 100,
) -> tuple:
    """
    Train Q-Learning agent on SocialConsensusEnv.

    Args:
        n_episodes: number of training episodes
        max_rounds: maximum rounds per episode
        save_dir: directory to save checkpoints and metrics
        eval_every: print stats every N episodes
        render_every: render env every N episodes

    Returns:
        (agent, tracker)
    """
    os.makedirs(save_dir, exist_ok=True)

    config = {
        "max_rounds": max_rounds,
        "consensus_threshold": 0.7,
        "use_llm": False,  # Use template responses for speed during RL training
    }
    env = SocialConsensusEnv(config=config)
    agent = QLearningAgent()
    tracker = MetricsTracker()

    print(f"{'=' * 60}")
    print(f"  Q-Learning Training — SocialConsensusEnv")
    print(f"{'=' * 60}")
    print(f"  Episodes:     {n_episodes}")
    print(f"  Max rounds:   {max_rounds}")
    print(f"  Actions:      {N_ACTIONS}")
    print(f"  State bins:   5^4 x 5 x 3 ≈ 9375 max states")
    print(f"  Save dir:     {save_dir}")
    print(f"{'=' * 60}\n")

    episode_rewards = []

    for episode in range(n_episodes):
        topic = TOPICS[episode % len(TOPICS)]
        obs, info = env.reset(options={"topic": topic})

        episode_reward = 0.0
        episode_td_errors = []
        done = False

        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            td_error = agent.update(obs, action, reward, next_obs, done)
            episode_td_errors.append(td_error)
            episode_reward += reward
            obs = next_obs

        agent.decay_epsilon()
        episode_rewards.append(episode_reward)

        # Record metrics
        trust_matrix = np.array(info.get("trust_matrix", [[0.5] * 3] * 3))
        n = trust_matrix.shape[0]
        mask = ~np.eye(n, dtype=bool)
        avg_trust = float(np.mean(trust_matrix[mask]))

        tracker.record_episode(
            consensus_reached=bool(info.get("consensus_reached", False)),
            average_trust=avg_trust,
            polarization_score=float(info.get("polarization_score", 0.0)),
            episode_reward=episode_reward,
            time_to_consensus=None if not info.get("consensus_reached") else info.get("step", None),
            n_rounds=int(info.get("round", max_rounds)),
            duration=0.0,
        )

        if (episode + 1) % eval_every == 0:
            recent_rewards = episode_rewards[-eval_every:]
            summary = tracker.get_summary()
            agent_stats = agent.get_stats()
            print(
                f"Ep {episode + 1:4d}/{n_episodes} | "
                f"Reward: {np.mean(recent_rewards):+.3f} ± {np.std(recent_rewards):.3f} | "
                f"Eps: {agent.epsilon:.3f} | "
                f"Q-states: {agent_stats['n_states_visited']:,} | "
                f"TD-err: {np.mean(episode_td_errors):.3f} | "
                f"Consensus: {summary.get('consensus_rate', 0):.1f}%"
            )
            agent.save(os.path.join(save_dir, f"qlearning_ep{episode + 1}.json"))

        if (episode + 1) % render_every == 0:
            env.render()

    # Final save
    agent.save(os.path.join(save_dir, "qlearning_final.json"))
    tracker.to_csv(os.path.join(save_dir, "metrics.csv"))
    tracker.to_json(os.path.join(save_dir, "metrics.json"))
    try:
        tracker.plot_learning_curve(os.path.join(save_dir, "learning_curve.png"))
    except Exception as e:
        print(f"[Warning] Could not plot learning curve: {e}")

    summary = tracker.get_summary()
    print(f"\n{'=' * 60}")
    print(f"  Training complete — Final Summary")
    print(f"{'=' * 60}")
    print(json.dumps(summary, indent=2))

    return agent, tracker


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Q-Learning on SocialConsensusEnv")
    parser.add_argument("--episodes", type=int, default=500, help="Number of training episodes")
    parser.add_argument("--max_rounds", type=int, default=20, help="Max rounds per episode")
    parser.add_argument("--save_dir", type=str, default="checkpoints/qlearning", help="Checkpoint directory")
    parser.add_argument("--eval_every", type=int, default=50, help="Evaluation interval")
    parser.add_argument("--render_every", type=int, default=100, help="Render interval")
    args = parser.parse_args()

    train(
        n_episodes=args.episodes,
        max_rounds=args.max_rounds,
        save_dir=args.save_dir,
        eval_every=args.eval_every,
        render_every=args.render_every,
    )
