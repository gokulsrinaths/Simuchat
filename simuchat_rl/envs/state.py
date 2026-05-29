"""
Environment state management for SocialConsensusEnv.
Tracks trust matrices, emotion vectors, agreement scores, and conversation history.
"""
import numpy as np
from typing import List, Dict, Any, Optional
import copy


AGENT_NAMES = ["Alice", "Bob", "Charlie"]
EMOTION_NAMES = ["joy", "trust", "fear", "anger", "optimism", "calmness", "interest", "confidence"]
N_AGENTS = 3
N_EMOTIONS = 8

# Emotion indices
JOY_IDX = 0
TRUST_IDX = 1
FEAR_IDX = 2
ANGER_IDX = 3
OPTIMISM_IDX = 4
CALMNESS_IDX = 5
INTEREST_IDX = 6
CONFIDENCE_IDX = 7

# Agent personalities
AGENT_PERSONALITIES = {
    "Alice": {
        "empathy": 0.9,
        "stability": 0.7,
        "boldness": 0.3,
        "base_emotions": {
            "joy": 0.7,
            "trust": 0.8,
            "fear": 0.2,
            "anger": 0.1,
            "optimism": 0.5,
            "calmness": 0.6,
            "interest": 0.5,
            "confidence": 0.4,
        },
    },
    "Bob": {
        "empathy": 0.4,
        "stability": 0.9,
        "boldness": 0.5,
        "base_emotions": {
            "joy": 0.4,
            "trust": 0.5,
            "fear": 0.2,
            "anger": 0.2,
            "optimism": 0.5,
            "calmness": 0.7,
            "interest": 0.8,
            "confidence": 0.7,
        },
    },
    "Charlie": {
        "empathy": 0.5,
        "stability": 0.5,
        "boldness": 0.9,
        "base_emotions": {
            "joy": 0.5,
            "trust": 0.4,
            "fear": 0.1,
            "anger": 0.3,
            "optimism": 0.7,
            "calmness": 0.4,
            "interest": 0.7,
            "confidence": 0.8,
        },
    },
}

# Trust update deltas per action
ACTION_TRUST_DELTAS = {
    "AGREE": 0.02,
    "DISAGREE": -0.015,
    "PERSUADE": 0.01,
    "QUESTION": 0.0,
    "SUPPORT": 0.02,
    "CHALLENGE": -0.015,
    "PROVIDE_EVIDENCE": 0.03,
    "SEEK_CONSENSUS": 0.02,
}

# Agreement update deltas per action
ACTION_AGREEMENT_DELTAS = {
    "AGREE": 0.05,
    "DISAGREE": -0.05,
    "PERSUADE": 0.02,
    "QUESTION": 0.0,
    "SUPPORT": 0.03,
    "CHALLENGE": -0.05,
    "PROVIDE_EVIDENCE": 0.03,
    "SEEK_CONSENSUS": 0.05,
}

# Emotion influence per action: {emotion_idx: delta}
ACTION_EMOTION_INFLUENCE = {
    "AGREE": {JOY_IDX: 0.03, TRUST_IDX: 0.02, ANGER_IDX: -0.02, CALMNESS_IDX: 0.01},
    "DISAGREE": {ANGER_IDX: 0.04, FEAR_IDX: 0.02, JOY_IDX: -0.02, TRUST_IDX: -0.02},
    "PERSUADE": {CONFIDENCE_IDX: 0.03, OPTIMISM_IDX: 0.02, INTEREST_IDX: 0.02},
    "QUESTION": {INTEREST_IDX: 0.03, FEAR_IDX: 0.01},
    "SUPPORT": {JOY_IDX: 0.02, TRUST_IDX: 0.03, CALMNESS_IDX: 0.02},
    "CHALLENGE": {CONFIDENCE_IDX: 0.04, ANGER_IDX: 0.03, FEAR_IDX: 0.02, CALMNESS_IDX: -0.03},
    "PROVIDE_EVIDENCE": {CONFIDENCE_IDX: 0.03, TRUST_IDX: 0.02, INTEREST_IDX: 0.03},
    "SEEK_CONSENSUS": {JOY_IDX: 0.02, CALMNESS_IDX: 0.04, OPTIMISM_IDX: 0.03, ANGER_IDX: -0.03},
}

EMOTION_DECAY_RATE = 0.05  # per turn
EMOTIONAL_CONTAGION_WEIGHT = 0.3  # weight for cross-agent emotional contagion


class AgentState:
    """Represents the current state of a single agent."""

    def __init__(self, name: str):
        self.name = name
        personality = AGENT_PERSONALITIES[name]
        self.empathy = personality["empathy"]
        self.stability = personality["stability"]
        self.boldness = personality["boldness"]

        base = personality["base_emotions"]
        self.emotions = np.array(
            [base[e] for e in EMOTION_NAMES], dtype=np.float32
        )

    def copy(self) -> "AgentState":
        new_state = AgentState.__new__(AgentState)
        new_state.name = self.name
        new_state.empathy = self.empathy
        new_state.stability = self.stability
        new_state.boldness = self.boldness
        new_state.emotions = self.emotions.copy()
        return new_state

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "empathy": self.empathy,
            "stability": self.stability,
            "boldness": self.boldness,
            "emotions": {EMOTION_NAMES[i]: float(self.emotions[i]) for i in range(N_EMOTIONS)},
        }


class EnvState:
    """
    Complete environment state for SocialConsensusEnv.

    Manages:
    - Trust matrix (3x3 float32): pairwise trust between agents
    - Emotion vectors (3x8 float32): emotional state per agent
    - Agreement scores (3 float32): how much each agent agrees
    - Conversation history: list of turn records
    """

    def __init__(self, topic: str = "climate change", max_rounds: int = 20):
        self.agent_names = AGENT_NAMES
        self.topic = topic
        self.max_rounds = max_rounds
        self.current_round = 0
        self.current_agent_idx = 0
        self._turn_count = 0  # total turns taken

        # Initialize agent states
        self.agent_states: List[AgentState] = [AgentState(name) for name in AGENT_NAMES]

        # Trust matrix: [i][j] = how much agent i trusts agent j
        self.trust_matrix = np.full((N_AGENTS, N_AGENTS), 0.5, dtype=np.float32)
        np.fill_diagonal(self.trust_matrix, 1.0)

        # Emotion vectors: [agent_idx][emotion_idx]
        self.emotion_vectors = np.array(
            [agent.emotions.copy() for agent in self.agent_states], dtype=np.float32
        )

        # Agreement scores: how much each agent is in agreement with the group
        self.agreement_scores = np.full(N_AGENTS, 0.5, dtype=np.float32)

        # Conversation history
        self.conversation_history: List[Dict[str, Any]] = []

    def reset(self, topic: str = "climate change", max_rounds: int = 20):
        """Reset state to initial conditions."""
        self.__init__(topic=topic, max_rounds=max_rounds)

    def update(
        self,
        agent_idx: int,
        action: int,
        response_text: str,
        action_name: str,
    ) -> None:
        """
        Apply one agent turn's update to the state.

        Steps:
        1. Apply emotion decay
        2. Apply action emotion influence
        3. Update trust matrix
        4. Update agreement scores
        5. If full round completed (every 3 turns), apply emotional contagion
        6. Append to conversation history
        7. Advance agent index and round counter
        """
        # --- 1. Emotion decay (exponential) ---
        decay = np.exp(-EMOTION_DECAY_RATE)
        self.emotion_vectors[agent_idx] *= decay

        # --- 2. Action emotion influence ---
        influence = ACTION_EMOTION_INFLUENCE.get(action_name, {})
        for emotion_idx, delta in influence.items():
            self.emotion_vectors[agent_idx][emotion_idx] += delta
        self.emotion_vectors = np.clip(self.emotion_vectors, 0.0, 1.0)

        # --- 3. Trust update ---
        trust_delta = ACTION_TRUST_DELTAS.get(action_name, 0.0)
        for j in range(N_AGENTS):
            if j != agent_idx:
                # All other agents' trust in the acting agent changes
                self.trust_matrix[j][agent_idx] += trust_delta
                # The acting agent's trust in others also changes slightly
                self.trust_matrix[agent_idx][j] += trust_delta * 0.5
        np.fill_diagonal(self.trust_matrix, 1.0)
        self.trust_matrix = np.clip(self.trust_matrix, 0.0, 1.0)

        # --- 4. Agreement score update ---
        agreement_delta = ACTION_AGREEMENT_DELTAS.get(action_name, 0.0)
        self.agreement_scores[agent_idx] += agreement_delta
        self.agreement_scores = np.clip(self.agreement_scores, 0.0, 1.0)

        # --- 5. Take snapshots for history BEFORE advancing turn ---
        trust_snapshot = self.trust_matrix.copy().tolist()
        emotion_snapshot = self.emotion_vectors.copy().tolist()

        # --- 6. Append to conversation history ---
        self.conversation_history.append(
            {
                "round": self.current_round,
                "agent": self.agent_names[agent_idx],
                "agent_idx": agent_idx,
                "action": action,
                "action_name": action_name,
                "text": response_text,
                "prompt": "",  # filled in by env
                "reward": 0.0,  # filled in by env after reward computation
                "trust_snapshot": trust_snapshot,
                "emotion_snapshot": emotion_snapshot,
                "agreement_scores": self.agreement_scores.tolist(),
            }
        )

        # --- 7. Advance turn counter and agent index ---
        self._turn_count += 1
        self.current_agent_idx = (agent_idx + 1) % N_AGENTS

        # Increment round every 3 agent turns (one full round = all 3 agents acted)
        if self._turn_count % N_AGENTS == 0:
            self.current_round += 1
            # --- 5b. Emotional contagion at end of every full round ---
            self._apply_emotional_contagion()

    def _apply_emotional_contagion(self) -> None:
        """
        After each full round, apply emotional contagion:
        For each agent i, for each emotion e:
            emotion[i][e] += sum_{j != i} (trust[i][j] * CONTAGION_WEIGHT * (emotion[j][e] - emotion[i][e]))
        Then clip to [0, 1].
        """
        new_emotions = self.emotion_vectors.copy()
        for i in range(N_AGENTS):
            for e in range(N_EMOTIONS):
                contagion_sum = 0.0
                for j in range(N_AGENTS):
                    if j != i:
                        contagion_sum += (
                            self.trust_matrix[i][j]
                            * EMOTIONAL_CONTAGION_WEIGHT
                            * (self.emotion_vectors[j][e] - self.emotion_vectors[i][e])
                        )
                new_emotions[i][e] += contagion_sum
        self.emotion_vectors = np.clip(new_emotions, 0.0, 1.0)

    def get_trust_matrix(self) -> np.ndarray:
        """Return a clipped copy of the trust matrix."""
        return np.clip(self.trust_matrix.copy(), 0.0, 1.0)

    def get_emotion_vectors(self) -> np.ndarray:
        """Return a clipped copy of the emotion vectors."""
        return np.clip(self.emotion_vectors.copy(), 0.0, 1.0)

    def get_agreement_scores(self) -> np.ndarray:
        """Return a clipped copy of the agreement scores."""
        return np.clip(self.agreement_scores.copy(), 0.0, 1.0)

    def is_consensus_reached(self, threshold: float = 0.75) -> bool:
        """
        Consensus is reached when:
        - All agreement scores > threshold
        - Mean off-diagonal trust > threshold * 0.8
        """
        agreement_ok = bool(np.all(self.agreement_scores > threshold))

        # Off-diagonal trust mean
        mask = ~np.eye(N_AGENTS, dtype=bool)
        off_diag_trust = self.trust_matrix[mask]
        trust_ok = bool(np.mean(off_diag_trust) > threshold * 0.8)

        return agreement_ok and trust_ok

    def get_conversation_summary(self, n: int = 5) -> str:
        """Return the last n messages as a formatted string."""
        recent = self.conversation_history[-n:] if len(self.conversation_history) >= n else self.conversation_history
        lines = []
        for entry in recent:
            lines.append(
                f"[Round {entry['round']}] {entry['agent']} ({entry['action_name']}): {entry['text']}"
            )
        return "\n".join(lines) if lines else "(no conversation yet)"

    def get_off_diagonal_trust_mean(self) -> float:
        """Return mean of off-diagonal trust values."""
        mask = ~np.eye(N_AGENTS, dtype=bool)
        return float(np.mean(self.trust_matrix[mask]))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to a JSON-compatible dict."""
        return {
            "topic": self.topic,
            "current_round": self.current_round,
            "current_agent_idx": self.current_agent_idx,
            "max_rounds": self.max_rounds,
            "turn_count": self._turn_count,
            "trust_matrix": self.trust_matrix.tolist(),
            "emotion_vectors": self.emotion_vectors.tolist(),
            "agreement_scores": self.agreement_scores.tolist(),
            "agent_names": self.agent_names,
            "emotion_names": EMOTION_NAMES,
            "conversation_length": len(self.conversation_history),
        }
