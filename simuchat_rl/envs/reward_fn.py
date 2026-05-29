"""
Reward function for SocialConsensusEnv.
Computes shaped rewards based on trust dynamics, consensus, and conversation quality.
"""
import numpy as np
from typing import Optional


# Reward constants
REWARD_TRUST_BUILD = 1.0     # per agent whose trust increased by > 0.01
REWARD_CONSENSUS = 5.0       # when consensus_reached flips True
REWARD_EVIDENCE = 2.0        # PROVIDE_EVIDENCE that increased agreement
PENALTY_CONFLICT = -1.0      # per agent whose trust decreased > 0.01
PENALTY_POLARIZATION = -3.0  # std(agreement_scores) increased by > 0.05

# Thresholds
TRUST_CHANGE_THRESHOLD = 0.01
AGREEMENT_CHANGE_THRESHOLD = 0.02
POLARIZATION_CHANGE_THRESHOLD = 0.05


class RewardFunction:
    """
    Computes shaped rewards for the SocialConsensusEnv.

    Reward components:
    - Trust building: +1.0 per agent whose trust in acting agent increased > 0.01
    - Conflict penalty: -1.0 per agent whose trust decreased > 0.01
    - Consensus bonus: +5.0 one-time when consensus is first reached
    - Evidence reward: +2.0 for PROVIDE_EVIDENCE that improved mean agreement
    - Polarization penalty: -3.0 if std(agreement_scores) increased by > 0.05
    """

    def __init__(
        self,
        trust_build_reward: float = REWARD_TRUST_BUILD,
        consensus_reward: float = REWARD_CONSENSUS,
        evidence_reward: float = REWARD_EVIDENCE,
        conflict_penalty: float = PENALTY_CONFLICT,
        polarization_penalty: float = PENALTY_POLARIZATION,
        trust_threshold: float = TRUST_CHANGE_THRESHOLD,
        agreement_threshold: float = AGREEMENT_CHANGE_THRESHOLD,
        polarization_threshold: float = POLARIZATION_CHANGE_THRESHOLD,
    ):
        self.trust_build_reward = trust_build_reward
        self.consensus_reward = consensus_reward
        self.evidence_reward = evidence_reward
        self.conflict_penalty = conflict_penalty
        self.polarization_penalty = polarization_penalty
        self.trust_threshold = trust_threshold
        self.agreement_threshold = agreement_threshold
        self.polarization_threshold = polarization_threshold

    def compute(
        self,
        action: int,
        action_name: str,
        prev_trust: np.ndarray,
        new_trust: np.ndarray,
        prev_agreements: np.ndarray,
        new_agreements: np.ndarray,
        agent_idx: int,
        consensus_reached: bool,
    ) -> float:
        """
        Compute the reward for a single agent turn.

        Args:
            action: integer action index
            action_name: string name of the action
            prev_trust: (3, 3) trust matrix before the action
            new_trust: (3, 3) trust matrix after the action
            prev_agreements: (3,) agreement scores before the action
            new_agreements: (3,) agreement scores after the action
            agent_idx: index of the acting agent
            consensus_reached: True if consensus was newly achieved this turn

        Returns:
            total_reward: float
        """
        total_reward = 0.0
        n_agents = prev_trust.shape[0]

        # --- Trust build / conflict penalty ---
        # Look at how other agents' trust in the acting agent changed
        for j in range(n_agents):
            if j != agent_idx:
                # Trust of agent j in the acting agent
                trust_change = new_trust[j][agent_idx] - prev_trust[j][agent_idx]
                if trust_change > self.trust_threshold:
                    total_reward += self.trust_build_reward
                elif trust_change < -self.trust_threshold:
                    total_reward += self.conflict_penalty

        # --- Consensus bonus ---
        if consensus_reached:
            total_reward += self.consensus_reward

        # --- Evidence reward ---
        if action_name == "PROVIDE_EVIDENCE":
            mean_prev = float(np.mean(prev_agreements))
            mean_new = float(np.mean(new_agreements))
            if mean_new > mean_prev + self.agreement_threshold:
                total_reward += self.evidence_reward

        # --- Polarization penalty ---
        std_prev = float(np.std(prev_agreements))
        std_new = float(np.std(new_agreements))
        if std_new > std_prev + self.polarization_threshold:
            total_reward += self.polarization_penalty

        return total_reward

    def compute_episode_bonus(
        self,
        final_trust: np.ndarray,
        final_agreements: np.ndarray,
        n_rounds_taken: int,
        max_rounds: int,
        consensus_reached: bool,
    ) -> float:
        """
        Optional episode-level bonus/penalty.
        Rewards efficiency (reaching consensus faster) and penalizes deadlock.
        """
        bonus = 0.0

        if consensus_reached and max_rounds > 0:
            # Efficiency bonus: more reward for reaching consensus earlier
            efficiency = 1.0 - (n_rounds_taken / max_rounds)
            bonus += 2.0 * efficiency

        # Small bonus for overall trust level at episode end
        n_agents = final_trust.shape[0]
        mask = ~np.eye(n_agents, dtype=bool)
        mean_trust = float(np.mean(final_trust[mask]))
        bonus += mean_trust * 0.5

        return bonus

    def __repr__(self) -> str:
        return (
            f"RewardFunction("
            f"trust_build={self.trust_build_reward}, "
            f"consensus={self.consensus_reward}, "
            f"evidence={self.evidence_reward}, "
            f"conflict={self.conflict_penalty}, "
            f"polarization={self.polarization_penalty})"
        )
