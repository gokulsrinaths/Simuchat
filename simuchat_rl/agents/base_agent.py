"""
Abstract base class for all agents in SocialConsensusEnv.
"""
from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, List, Optional


class BaseAgent(ABC):
    """
    Abstract base class for SocialConsensusEnv agents.

    All agents must implement select_action(observation) -> int.
    The update() method records experience and can be overridden for learning agents.
    """

    def __init__(self, agent_idx: int, n_actions: int = 8):
        """
        Args:
            agent_idx: index of this agent (0=Alice, 1=Bob, 2=Charlie)
            n_actions: number of possible actions (default 8)
        """
        self.agent_idx = agent_idx
        self.n_actions = n_actions
        self.action_history: List[int] = []
        self.reward_history: List[float] = []
        self.total_reward: float = 0.0
        self._episode_count: int = 0

    @abstractmethod
    def select_action(self, observation: Dict[str, Any]) -> int:
        """
        Select an action given the current observation.

        Args:
            observation: dict with keys trust_matrix, emotion_vectors,
                         agreement_scores, current_round, current_agent

        Returns:
            action: integer in [0, n_actions)
        """
        ...

    def update(
        self,
        observation: Dict[str, Any],
        action: int,
        reward: float,
        next_observation: Dict[str, Any],
        done: bool,
    ) -> None:
        """
        Record experience. Override in learning agents to update policy.

        Args:
            observation: state before action
            action: action taken
            reward: reward received
            next_observation: state after action
            done: whether episode ended
        """
        self.action_history.append(action)
        self.reward_history.append(reward)
        self.total_reward += reward

    def reset(self) -> None:
        """Reset per-episode tracking."""
        self.action_history = []
        self.reward_history = []
        self.total_reward = 0.0
        self._episode_count += 1

    def get_stats(self) -> Dict[str, Any]:
        """Return agent statistics."""
        action_dist = (
            np.bincount(self.action_history, minlength=self.n_actions).tolist()
            if self.action_history
            else [0] * self.n_actions
        )
        return {
            "agent_idx": self.agent_idx,
            "total_reward": self.total_reward,
            "n_actions": len(self.action_history),
            "mean_reward": float(np.mean(self.reward_history)) if self.reward_history else 0.0,
            "std_reward": float(np.std(self.reward_history)) if self.reward_history else 0.0,
            "action_distribution": action_dist,
            "episode_count": self._episode_count,
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"agent_idx={self.agent_idx}, "
            f"total_reward={self.total_reward:.3f}, "
            f"n_actions={len(self.action_history)})"
        )
