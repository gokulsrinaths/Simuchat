"""
Baseline agents for SocialConsensusEnv.
Implements rule-based and simple learning agents for comparison.
"""
import numpy as np
from typing import Dict, Any, Optional

from .base_agent import BaseAgent

# Action indices
AGREE = 0
DISAGREE = 1
PERSUADE = 2
QUESTION = 3
SUPPORT = 4
CHALLENGE = 5
PROVIDE_EVIDENCE = 6
SEEK_CONSENSUS = 7

N_AGENTS = 3


class RandomAgent(BaseAgent):
    """
    Uniformly random action selection baseline.
    Useful as a lower bound for performance comparison.
    """

    def __init__(self, agent_idx: int, n_actions: int = 8, seed: Optional[int] = None):
        super().__init__(agent_idx=agent_idx, n_actions=n_actions)
        self.rng = np.random.default_rng(seed)

    def select_action(self, observation: Dict[str, Any]) -> int:
        """Select uniformly random action."""
        return int(self.rng.integers(0, self.n_actions))

    def __repr__(self) -> str:
        return f"RandomAgent(agent_idx={self.agent_idx}, n_actions={self.n_actions})"


class GreedyTrustAgent(BaseAgent):
    """
    Rule-based agent that selects actions based on current trust levels.

    Strategy:
    - Low average trust (< 0.5): use AGREE to build trust
    - Medium trust (0.5-0.7): use SUPPORT to reinforce progress
    - High trust (> 0.7): use SEEK_CONSENSUS to close the deal
    """

    LOW_TRUST_THRESHOLD = 0.5
    HIGH_TRUST_THRESHOLD = 0.7

    def __init__(self, agent_idx: int, n_actions: int = 8):
        super().__init__(agent_idx=agent_idx, n_actions=n_actions)

    def select_action(self, observation: Dict[str, Any]) -> int:
        """Select action based on trust levels."""
        trust_matrix = observation.get("trust_matrix")
        agreements = observation.get("agreement_scores", [0.5, 0.5, 0.5])

        if trust_matrix is None:
            return AGREE

        # Calculate average trust from others toward this agent
        avg_trust = 0.0
        count = 0
        for j in range(N_AGENTS):
            if j != self.agent_idx:
                try:
                    avg_trust += float(trust_matrix[j][self.agent_idx])
                    count += 1
                except (IndexError, TypeError):
                    pass

        if count > 0:
            avg_trust /= count
        else:
            avg_trust = 0.5

        # Agreement context
        try:
            my_agreement = float(agreements[self.agent_idx])
        except (IndexError, TypeError):
            my_agreement = 0.5

        # Strategy based on trust level
        if avg_trust < self.LOW_TRUST_THRESHOLD:
            # Build trust first
            return AGREE
        elif avg_trust < self.HIGH_TRUST_THRESHOLD:
            # Reinforce progress
            if my_agreement < 0.5:
                return SUPPORT
            else:
                return PROVIDE_EVIDENCE
        else:
            # Close toward consensus
            return SEEK_CONSENSUS

    def __repr__(self) -> str:
        return (
            f"GreedyTrustAgent("
            f"agent_idx={self.agent_idx}, "
            f"total_reward={self.total_reward:.3f})"
        )


class ConsensusSeekingAgent(BaseAgent):
    """
    Epsilon-greedy agent that learns which actions lead to highest reward.
    Tracks per-action cumulative rewards and exploits the best action most of the time.

    This is essentially an online multi-armed bandit (MAB) approach.
    """

    def __init__(
        self,
        agent_idx: int,
        n_actions: int = 8,
        epsilon: float = 0.1,
        seed: Optional[int] = None,
    ):
        super().__init__(agent_idx=agent_idx, n_actions=n_actions)
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)

        # Per-action reward tracking (bandit-style)
        self.action_rewards = np.zeros(n_actions, dtype=np.float64)
        self.action_counts = np.zeros(n_actions, dtype=np.int64)

        # Initialize with slight preference for consensus-building actions
        # based on domain knowledge
        self.action_rewards[AGREE] = 0.5
        self.action_rewards[SUPPORT] = 0.5
        self.action_rewards[SEEK_CONSENSUS] = 0.8
        self.action_rewards[PROVIDE_EVIDENCE] = 0.6

    def select_action(self, observation: Dict[str, Any]) -> int:
        """
        Epsilon-greedy action selection.
        With probability epsilon, explore randomly.
        Otherwise, exploit the action with highest estimated reward.
        """
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.n_actions))

        # Exploit: pick action with highest estimated mean reward
        # Break ties randomly
        mean_rewards = np.where(
            self.action_counts > 0,
            self.action_rewards / np.maximum(self.action_counts, 1),
            0.0,
        )
        # Add small noise to break ties
        noise = self.rng.uniform(0, 1e-6, size=self.n_actions)
        return int(np.argmax(mean_rewards + noise))

    def update(
        self,
        observation: Dict[str, Any],
        action: int,
        reward: float,
        next_observation: Dict[str, Any],
        done: bool,
    ) -> None:
        """Update action reward estimates (online mean tracking)."""
        super().update(observation, action, reward, next_observation, done)

        # Update cumulative reward and count for this action
        self.action_rewards[action] += reward
        self.action_counts[action] += 1

    def get_action_means(self) -> Dict[str, float]:
        """Return estimated mean reward per action."""
        from ..envs.social_consensus_env import ACTIONS
        means = {}
        for idx, name in ACTIONS.items():
            if self.action_counts[idx] > 0:
                means[name] = float(self.action_rewards[idx] / self.action_counts[idx])
            else:
                means[name] = 0.0
        return means

    def reset(self) -> None:
        """Reset episode state (but keep learned action estimates)."""
        super().reset()
        # Note: we intentionally do NOT reset action_rewards/counts
        # so the agent retains cross-episode learning

    def __repr__(self) -> str:
        best_action_idx = int(np.argmax(
            self.action_rewards / np.maximum(self.action_counts, 1)
        ))
        from ..envs.social_consensus_env import ACTIONS
        best_action = ACTIONS.get(best_action_idx, "UNKNOWN")
        return (
            f"ConsensusSeekingAgent("
            f"agent_idx={self.agent_idx}, "
            f"epsilon={self.epsilon}, "
            f"best_action={best_action}, "
            f"total_reward={self.total_reward:.3f})"
        )


class AdversarialAgent(BaseAgent):
    """
    Adversarial agent designed to stress-test the environment.

    Strategy:
    - When trust is high (> 0.6): use DISAGREE or CHALLENGE to disrupt consensus
    - When trust is medium/low: use QUESTION to sow doubt
    - Occasionally uses PERSUADE to test the environment's robustness

    This agent is useful for:
    1. Testing environment stability under adversarial conditions
    2. Evaluating how well cooperative agents recover from disruption
    3. Stress-testing reward function sensitivity
    """

    HIGH_TRUST_THRESHOLD = 0.6
    DISRUPTION_PROBABILITY = 0.8  # probability of choosing disruptive action

    def __init__(
        self,
        agent_idx: int,
        n_actions: int = 8,
        seed: Optional[int] = None,
        disruption_level: float = 0.8,
    ):
        super().__init__(agent_idx=agent_idx, n_actions=n_actions)
        self.disruption_level = disruption_level
        self.rng = np.random.default_rng(seed)

    def select_action(self, observation: Dict[str, Any]) -> int:
        """
        Select action to maximize disruption to emerging consensus.
        """
        trust_matrix = observation.get("trust_matrix")
        agreements = observation.get("agreement_scores", [0.5, 0.5, 0.5])

        # Calculate average trust in this agent from others
        avg_trust = 0.5
        if trust_matrix is not None:
            trusts = []
            for j in range(N_AGENTS):
                if j != self.agent_idx:
                    try:
                        trusts.append(float(trust_matrix[j][self.agent_idx]))
                    except (IndexError, TypeError):
                        pass
            if trusts:
                avg_trust = float(np.mean(trusts))

        # Check if consensus is converging (high mean agreement)
        try:
            mean_agreement = float(np.mean([float(a) for a in agreements]))
        except (TypeError, ValueError):
            mean_agreement = 0.5

        # With probability (1 - disruption_level), act randomly (less predictable)
        if self.rng.random() > self.disruption_level:
            return int(self.rng.integers(0, self.n_actions))

        # Adversarial strategy
        if avg_trust > self.HIGH_TRUST_THRESHOLD:
            # High trust: disrupt with DISAGREE or CHALLENGE
            return int(self.rng.choice([DISAGREE, CHALLENGE], p=[0.5, 0.5]))
        elif mean_agreement > 0.6:
            # Convergence happening: challenge or disagree
            return int(self.rng.choice([CHALLENGE, DISAGREE, QUESTION], p=[0.4, 0.4, 0.2]))
        else:
            # Low trust: sow doubt with QUESTION
            return QUESTION

    def __repr__(self) -> str:
        return (
            f"AdversarialAgent("
            f"agent_idx={self.agent_idx}, "
            f"disruption_level={self.disruption_level}, "
            f"total_reward={self.total_reward:.3f})"
        )
