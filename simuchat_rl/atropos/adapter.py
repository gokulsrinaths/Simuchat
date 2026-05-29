"""
Atropos compatibility layer for SocialConsensusEnv.

Atropos is an RL training framework that uses GRPO/PPO to fine-tune language models.
It expects rollouts as (prompt, completion, reward) tuples, either at token level
or sequence level.

This adapter converts SocialConsensusEnv episode rollouts into Atropos format.
"""
import json
import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class AtroposRollout:
    """
    Single turn rollout in Atropos format.

    Represents one agent turn: the prompt that was shown to the LLM,
    the completion it generated, and the reward assigned.

    Atropos uses these tuples to compute policy gradients via GRPO.
    """

    prompt: str
    completion: str
    reward: float
    agent_name: str
    action: str
    round_num: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "prompt": self.prompt,
            "completion": self.completion,
            "reward": self.reward,
            "agent_name": self.agent_name,
            "action": self.action,
            "round_num": self.round_num,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AtroposRollout":
        """Deserialize from dict."""
        return cls(
            prompt=d["prompt"],
            completion=d["completion"],
            reward=float(d["reward"]),
            agent_name=d.get("agent_name", ""),
            action=d.get("action", ""),
            round_num=int(d.get("round_num", 0)),
            metadata=d.get("metadata", {}),
        )

    def __repr__(self) -> str:
        return (
            f"AtroposRollout("
            f"agent={self.agent_name}, "
            f"action={self.action}, "
            f"round={self.round_num}, "
            f"reward={self.reward:.3f}, "
            f"prompt_len={len(self.prompt)}, "
            f"completion_len={len(self.completion)})"
        )


class AtroposAdapter:
    """
    Adapts SocialConsensusEnv rollouts to Atropos training format.

    Atropos expects:
    - A list of rollout groups (one per episode)
    - Each group is a list of AtroposRollout objects
    - Rewards can be token-level or sequence-level

    This adapter supports sequence-level rewards (one reward per completion).
    For token-level rewards, the sequence reward is broadcast uniformly to
    all tokens in the completion.

    Usage:
        adapter = AtroposAdapter(reward_scale=1.0, normalize_rewards=True)
        rollouts = adapter.convert_episode(env.get_rollout_for_atropos())
        normalized = adapter.normalize_episode_rewards(rollouts)
        trainer_batch = adapter.format_for_trainer([normalized])
    """

    def __init__(
        self,
        reward_scale: float = 1.0,
        normalize_rewards: bool = True,
        token_level_rewards: bool = False,
    ):
        """
        Args:
            reward_scale: scale factor applied to all rewards
            normalize_rewards: if True, normalize rewards to zero mean, unit variance
            token_level_rewards: if True, broadcast sequence reward to all tokens
        """
        self.reward_scale = reward_scale
        self.normalize_rewards = normalize_rewards
        self.token_level_rewards = token_level_rewards

    def convert_episode(
        self, env_rollout: List[Dict[str, Any]]
    ) -> List[AtroposRollout]:
        """
        Convert a list of env rollout dicts to AtroposRollout objects.

        Args:
            env_rollout: list of dicts from env.get_rollout_for_atropos()
                Each dict has: prompt, completion, reward, agent, action, round, metadata

        Returns:
            list of AtroposRollout objects
        """
        rollouts = []
        for entry in env_rollout:
            rollout = AtroposRollout(
                prompt=entry.get("prompt", ""),
                completion=entry.get("completion", ""),
                reward=float(entry.get("reward", 0.0)) * self.reward_scale,
                agent_name=entry.get("agent", ""),
                action=entry.get("action", ""),
                round_num=int(entry.get("round", 0)),
                metadata=entry.get("metadata", {}),
            )
            rollouts.append(rollout)
        return rollouts

    def normalize_episode_rewards(
        self, rollouts: List[AtroposRollout]
    ) -> List[AtroposRollout]:
        """
        Normalize rewards within an episode to zero mean, unit variance.

        This is standard practice in GRPO/PPO to stabilize training.
        Returns new AtroposRollout objects (does not mutate in place).

        If all rewards are identical (std=0), returns rewards as-is.
        """
        if not rollouts:
            return rollouts

        rewards = [r.reward for r in rollouts]
        mean_r = sum(rewards) / len(rewards)
        variance = sum((r - mean_r) ** 2 for r in rewards) / max(len(rewards) - 1, 1)
        std_r = math.sqrt(variance)

        if std_r < 1e-8:
            # All rewards identical — return as-is
            return rollouts

        normalized = []
        for r in rollouts:
            normalized_reward = (r.reward - mean_r) / std_r
            new_rollout = AtroposRollout(
                prompt=r.prompt,
                completion=r.completion,
                reward=normalized_reward,
                agent_name=r.agent_name,
                action=r.action,
                round_num=r.round_num,
                metadata={**r.metadata, "original_reward": r.reward, "normalized": True},
            )
            normalized.append(new_rollout)

        return normalized

    def format_for_trainer(
        self, episodes: List[List["AtroposRollout"]]
    ) -> Dict[str, Any]:
        """
        Format rollouts for the Atropos GRPO trainer.

        Returns a dict compatible with Atropos trainer.add_batch() interface:
        {
            "prompts": List[str],           # one per turn
            "completions": List[str],        # one per turn
            "rewards": List[float],          # sequence-level rewards
            "token_rewards": List[List[float]] | None,  # token-level if requested
            "metadata": List[Dict],          # per-turn metadata
            "episode_ids": List[int],        # which episode each turn belongs to
            "agent_names": List[str],        # which agent generated each turn
            "actions": List[str],            # action for each turn
        }

        Compatible with Atropos trainer.add_batch() interface.
        """
        prompts = []
        completions = []
        rewards = []
        token_rewards = [] if self.token_level_rewards else None
        metadata_list = []
        episode_ids = []
        agent_names = []
        actions = []

        for ep_idx, episode in enumerate(episodes):
            ep_rollouts = episode
            if self.normalize_rewards:
                ep_rollouts = self.normalize_episode_rewards(ep_rollouts)

            for rollout in ep_rollouts:
                prompts.append(rollout.prompt)
                completions.append(rollout.completion)
                rewards.append(rollout.reward)
                metadata_list.append(rollout.metadata)
                episode_ids.append(ep_idx)
                agent_names.append(rollout.agent_name)
                actions.append(rollout.action)

                if self.token_level_rewards and rollout.completion:
                    # Broadcast sequence reward uniformly to each token
                    # Token count is approximated by word count + 1
                    n_tokens = max(1, len(rollout.completion.split()))
                    token_rewards.append([rollout.reward / n_tokens] * n_tokens)

        return {
            "prompts": prompts,
            "completions": completions,
            "rewards": rewards,
            "token_rewards": token_rewards,
            "metadata": metadata_list,
            "episode_ids": episode_ids,
            "agent_names": agent_names,
            "actions": actions,
            "n_episodes": len(episodes),
            "n_turns": len(prompts),
        }

    def save_rollouts(
        self,
        episodes: List[List["AtroposRollout"]],
        filepath: str,
    ) -> None:
        """
        Save rollout episodes to a JSON file.

        Args:
            episodes: list of episodes, each a list of AtroposRollout
            filepath: output file path
        """
        data = {
            "n_episodes": len(episodes),
            "adapter_config": {
                "reward_scale": self.reward_scale,
                "normalize_rewards": self.normalize_rewards,
                "token_level_rewards": self.token_level_rewards,
            },
            "episodes": [
                [rollout.to_dict() for rollout in episode]
                for episode in episodes
            ],
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"[AtroposAdapter] Saved {len(episodes)} episodes to {filepath}")

    def load_rollouts(self, filepath: str) -> List[List["AtroposRollout"]]:
        """
        Load rollout episodes from a JSON file.

        Args:
            filepath: input file path

        Returns:
            list of episodes, each a list of AtroposRollout
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        episodes = []
        for ep_data in data.get("episodes", []):
            episode = [AtroposRollout.from_dict(r) for r in ep_data]
            episodes.append(episode)

        print(f"[AtroposAdapter] Loaded {len(episodes)} episodes from {filepath}")
        return episodes

    def get_reward_statistics(
        self, episodes: List[List["AtroposRollout"]]
    ) -> Dict[str, float]:
        """Compute reward statistics across all episodes."""
        all_rewards = [
            rollout.reward
            for episode in episodes
            for rollout in episode
        ]
        if not all_rewards:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n": 0}

        mean = sum(all_rewards) / len(all_rewards)
        variance = sum((r - mean) ** 2 for r in all_rewards) / max(len(all_rewards) - 1, 1)
        std = math.sqrt(variance)

        return {
            "mean": round(mean, 4),
            "std": round(std, 4),
            "min": round(min(all_rewards), 4),
            "max": round(max(all_rewards), 4),
            "n": len(all_rewards),
            "n_episodes": len(episodes),
        }

    def __repr__(self) -> str:
        return (
            f"AtroposAdapter("
            f"reward_scale={self.reward_scale}, "
            f"normalize={self.normalize_rewards}, "
            f"token_level={self.token_level_rewards})"
        )


def format_rollout_for_atropos(
    env: Any,
    episode_rollout: List[Dict[str, Any]],
) -> List[AtroposRollout]:
    """
    Module-level convenience function to convert an episode rollout to Atropos format.

    Args:
        env: SocialConsensusEnv instance (unused, kept for API compatibility)
        episode_rollout: list of rollout dicts from env.get_rollout_for_atropos()

    Returns:
        list of AtroposRollout objects
    """
    adapter = AtroposAdapter()
    return adapter.convert_episode(episode_rollout)
