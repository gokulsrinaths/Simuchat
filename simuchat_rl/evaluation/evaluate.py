"""
Evaluation utilities for SocialConsensusEnv agents.

Provides:
- evaluate_agent(): Run an agent for N episodes and collect metrics
- run_evaluation_suite(): Compare multiple agents head-to-head
- compare_action_distributions(): Plot action distribution comparison

Usage:
    from evaluation.evaluate import evaluate_agent, run_evaluation_suite
    results = evaluate_agent(my_agent, n_episodes=100)
    suite = run_evaluation_suite({"DQN": dqn_agent, "Random": random_agent})
"""
import os
import sys
import json
import numpy as np
from typing import Dict, Any, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.social_consensus_env import SocialConsensusEnv, N_ACTIONS, ACTIONS
from metrics.tracker import MetricsTracker

EVAL_TOPICS = [
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

ACTION_NAMES = list(ACTIONS.values())


def evaluate_agent(
    agent,
    n_episodes: int = 100,
    topics: Optional[List[str]] = None,
    config: Optional[Dict[str, Any]] = None,
    render: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate an agent for n_episodes and collect comprehensive metrics.

    The agent must implement:
        select_action(observation: dict) -> int

    Args:
        agent: agent with select_action(obs) -> int interface
        n_episodes: number of evaluation episodes
        topics: list of topics (cycles through if fewer than n_episodes)
        config: environment config dict
        render: if True, render the last episode
        verbose: if True, print progress

    Returns:
        summary dict with:
        - consensus_rate (%)
        - mean_reward, std_reward
        - mean_trust, mean_polarization
        - mean_rounds
        - action_distribution: {action_name: count}
        - per_topic_results: {topic: {consensus_rate, mean_reward, n_episodes}}
    """
    topics = topics or EVAL_TOPICS
    config = config or {}
    eval_config = {
        "max_rounds": config.get("max_rounds", 20),
        "consensus_threshold": config.get("consensus_threshold", 0.75),
        "use_llm": False,
        "verbose": verbose,
    }

    env = SocialConsensusEnv(config=eval_config)
    tracker = MetricsTracker()

    all_action_counts = np.zeros(N_ACTIONS, dtype=np.int64)
    per_topic_data: Dict[str, Dict[str, Any]] = {}

    for episode in range(n_episodes):
        topic = topics[episode % len(topics)]

        if hasattr(agent, "reset"):
            agent.reset()

        obs, info = env.reset(options={"topic": topic})
        episode_reward = 0.0
        episode_actions = []
        done = False

        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated

            if hasattr(agent, "update"):
                agent.update(obs, action, reward, next_obs, done)

            episode_reward += reward
            episode_actions.append(action)
            obs = next_obs

        # Accumulate action distribution
        for a in episode_actions:
            all_action_counts[a] += 1

        # Record metrics
        trust_matrix = np.array(step_info.get("trust_matrix", [[0.5] * 3] * 3))
        mask = ~np.eye(trust_matrix.shape[0], dtype=bool)
        avg_trust = float(np.mean(trust_matrix[mask]))
        consensus_reached = bool(step_info.get("consensus_reached", False))

        tracker.record_episode(
            consensus_reached=consensus_reached,
            average_trust=avg_trust,
            polarization_score=float(step_info.get("polarization_score", 0.0)),
            episode_reward=episode_reward,
            time_to_consensus=step_info.get("step") if consensus_reached else None,
            n_rounds=int(step_info.get("round", eval_config["max_rounds"])),
            duration=0.0,
        )

        # Per-topic tracking
        if topic not in per_topic_data:
            per_topic_data[topic] = {
                "consensus_count": 0,
                "total_reward": 0.0,
                "n_episodes": 0,
            }
        per_topic_data[topic]["n_episodes"] += 1
        per_topic_data[topic]["total_reward"] += episode_reward
        if consensus_reached:
            per_topic_data[topic]["consensus_count"] += 1

        if verbose and (episode + 1) % 10 == 0:
            partial_summary = tracker.get_summary()
            print(
                f"  [{episode + 1}/{n_episodes}] "
                f"Consensus: {partial_summary.get('consensus_rate', 0):.1f}% | "
                f"Reward: {partial_summary.get('mean_reward', 0):.3f}"
            )

    if render:
        env.render()

    # Build summary
    summary = tracker.get_summary()

    # Action distribution
    action_distribution = {
        ACTION_NAMES[i]: int(all_action_counts[i]) for i in range(N_ACTIONS)
    }
    total_actions = sum(all_action_counts)
    action_distribution_pct = {
        name: round(100.0 * count / max(total_actions, 1), 2)
        for name, count in action_distribution.items()
    }

    # Per-topic results
    per_topic_results = {}
    for topic, data in per_topic_data.items():
        n = data["n_episodes"]
        per_topic_results[topic] = {
            "consensus_rate": round(100.0 * data["consensus_count"] / max(n, 1), 1),
            "mean_reward": round(data["total_reward"] / max(n, 1), 3),
            "n_episodes": n,
        }

    return {
        **summary,
        "action_distribution": action_distribution,
        "action_distribution_pct": action_distribution_pct,
        "per_topic_results": per_topic_results,
        "agent_class": type(agent).__name__,
    }


def run_evaluation_suite(
    agents_dict: Dict[str, Any],
    n_episodes: int = 50,
    config: Optional[Dict[str, Any]] = None,
    save_path: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Compare multiple agents across the same set of evaluation episodes.

    Args:
        agents_dict: dict of {agent_name: agent_instance}
        n_episodes: number of evaluation episodes per agent
        config: optional environment config
        save_path: if provided, save comparison results to JSON

    Returns:
        dict of {agent_name: evaluation_results}
    """
    print(f"\n{'=' * 70}")
    print(f"  Evaluation Suite: {len(agents_dict)} agents × {n_episodes} episodes")
    print(f"{'=' * 70}")

    comparison = {}

    for agent_name, agent in agents_dict.items():
        print(f"\n  Evaluating: {agent_name}...")
        results = evaluate_agent(
            agent=agent,
            n_episodes=n_episodes,
            config=config,
            verbose=False,
        )
        comparison[agent_name] = results
        print(
            f"  → Consensus: {results.get('consensus_rate', 0):.1f}% | "
            f"Reward: {results.get('mean_reward', 0):.3f} ± {results.get('std_reward', 0):.3f} | "
            f"Trust: {results.get('mean_trust', 0):.3f} | "
            f"Polarization: {results.get('mean_polarization', 0):.3f}"
        )

    # Print comparison table
    print(f"\n{'=' * 70}")
    print(f"  Comparison Table (n={n_episodes} episodes each)")
    print(f"{'=' * 70}")

    headers = ["Agent", "Consensus%", "Mean Reward", "Std Reward", "Mean Trust", "Polarization", "Mean Rounds"]
    col_widths = [22, 12, 13, 12, 12, 14, 13]

    header_line = "  " + " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print(header_line)
    print("  " + "-" * (sum(col_widths) + 3 * (len(headers) - 1)))

    for agent_name, results in comparison.items():
        row = [
            agent_name[:20],
            f"{results.get('consensus_rate', 0):.1f}%",
            f"{results.get('mean_reward', 0):+.3f}",
            f"{results.get('std_reward', 0):.3f}",
            f"{results.get('mean_trust', 0):.3f}",
            f"{results.get('mean_polarization', 0):.3f}",
            f"{results.get('mean_rounds', 0):.1f}",
        ]
        row_line = "  " + " | ".join(v.ljust(w) for v, w in zip(row, col_widths))
        print(row_line)

    print(f"{'=' * 70}\n")

    # Best agent by consensus rate
    best_name = max(comparison, key=lambda k: comparison[k].get("consensus_rate", 0))
    print(f"  Best agent (consensus): {best_name} "
          f"({comparison[best_name].get('consensus_rate', 0):.1f}%)")

    best_reward_name = max(comparison, key=lambda k: comparison[k].get("mean_reward", 0))
    print(f"  Best agent (reward):    {best_reward_name} "
          f"({comparison[best_reward_name].get('mean_reward', 0):+.3f})")

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2, default=str)
        print(f"\n  Results saved to: {save_path}")

    return comparison


def compare_action_distributions(
    agents_dict: Dict[str, Dict[str, Any]],
    n_episodes: int = 20,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot action distribution comparison across multiple agents.

    Args:
        agents_dict: dict of {agent_name: agent_instance} OR
                     pre-computed {agent_name: evaluate_agent_results}
        n_episodes: episodes to evaluate (only used if agents are instances)
        save_path: optional output PNG path
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[compare_action_distributions] matplotlib not available.")
        return

    # If values are agent instances, evaluate them
    results = {}
    for name, obj in agents_dict.items():
        if hasattr(obj, "select_action"):
            print(f"  Evaluating {name} for action distribution...")
            results[name] = evaluate_agent(obj, n_episodes=n_episodes)
        elif isinstance(obj, dict) and "action_distribution" in obj:
            results[name] = obj
        else:
            print(f"  Skipping {name}: unrecognized type")

    if not results:
        print("[compare_action_distributions] No valid agents to plot.")
        return

    action_names = list(ACTIONS.values())
    n_agents = len(results)
    x = np.arange(len(action_names))
    width = 0.8 / max(n_agents, 1)

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, n_agents))

    for i, (agent_name, res) in enumerate(results.items()):
        dist = res.get("action_distribution_pct", res.get("action_distribution", {}))
        values = [dist.get(name, 0) for name in action_names]
        offset = (i - n_agents / 2 + 0.5) * width
        ax.bar(x + offset, values, width * 0.9, label=agent_name, color=colors[i], alpha=0.85)

    ax.set_xlabel("Action")
    ax.set_ylabel("Frequency (%)")
    ax.set_title("Action Distribution Comparison Across Agents")
    ax.set_xticks(x)
    ax.set_xticklabels(action_names, rotation=30, ha="right")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[compare_action_distributions] Saved to: {save_path}")
    else:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    """
    Main evaluation block: compare baseline agents and optionally load
    DQN/PPO checkpoints.
    """
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from agents.baseline_agents import RandomAgent, ConsensusSeekingAgent, GreedyTrustAgent, AdversarialAgent

    print("Initializing baseline agents...")

    agents = {
        "Random": RandomAgent(agent_idx=0),
        "GreedyTrust": GreedyTrustAgent(agent_idx=0),
        "ConsensusSeeking": ConsensusSeekingAgent(agent_idx=0),
        "Adversarial": AdversarialAgent(agent_idx=0),
    }

    # Try to load DQN checkpoint
    dqn_path = os.path.join(
        os.path.dirname(__file__), "..", "checkpoints", "dqn", "dqn_final.pt"
    )
    if os.path.exists(dqn_path):
        try:
            from training.train_dqn import DQNAgent, flatten_observation

            class DQNEvalAgent:
                def __init__(self, dqn_agent):
                    self._agent = dqn_agent

                def select_action(self, obs):
                    flat = flatten_observation(obs)
                    return self._agent.select_action(flat)

                def reset(self):
                    pass

            dqn_agent = DQNAgent()
            dqn_agent.load(dqn_path)
            dqn_agent.epsilon = 0.0  # greedy evaluation
            agents["DQN"] = DQNEvalAgent(dqn_agent)
            print(f"Loaded DQN checkpoint from: {dqn_path}")
        except Exception as e:
            print(f"[Warning] Could not load DQN: {e}")

    # Try to load PPO checkpoint
    ppo_path = os.path.join(
        os.path.dirname(__file__), "..", "checkpoints", "ppo", "ppo_final.pt"
    )
    if os.path.exists(ppo_path):
        try:
            from training.train_ppo import PPOAgent
            from training.train_dqn import flatten_observation

            class PPOEvalAgent:
                def __init__(self, ppo_agent):
                    self._agent = ppo_agent

                def select_action(self, obs):
                    flat = flatten_observation(obs)
                    action, _, _ = self._agent.select_action(flat)
                    return action

                def reset(self):
                    pass

            ppo_agent = PPOAgent()
            ppo_agent.load(ppo_path)
            agents["PPO"] = PPOEvalAgent(ppo_agent)
            print(f"Loaded PPO checkpoint from: {ppo_path}")
        except Exception as e:
            print(f"[Warning] Could not load PPO: {e}")

    # Run evaluation
    comparison = run_evaluation_suite(agents, n_episodes=50, save_path="results/evaluation_comparison.json")

    # Plot action distributions
    compare_action_distributions(
        agents,
        n_episodes=20,
        save_path="results/action_distributions.png",
    )
