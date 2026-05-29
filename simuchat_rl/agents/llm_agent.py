"""
LLM-powered agent for SocialConsensusEnv.
Uses DeepInfra to select actions AND generate responses via structured prompting.
"""
import json
import re
import warnings
from typing import Dict, Any, Optional, Tuple

import numpy as np

from .base_agent import BaseAgent

# Action name to index mapping
ACTION_NAME_TO_IDX = {
    "AGREE": 0,
    "DISAGREE": 1,
    "PERSUADE": 2,
    "QUESTION": 3,
    "SUPPORT": 4,
    "CHALLENGE": 5,
    "PROVIDE_EVIDENCE": 6,
    "SEEK_CONSENSUS": 7,
}

ACTION_IDX_TO_NAME = {v: k for k, v in ACTION_NAME_TO_IDX.items()}

AGENT_NAMES = ["Alice", "Bob", "Charlie"]


class LLMAgent(BaseAgent):
    """
    An agent that uses an LLM to select actions via structured JSON prompting.

    The LLM is asked to return a JSON object:
        {"action": "AGREE", "reasoning": "..."}

    The action is parsed and mapped to an integer. If parsing fails,
    the agent falls back to a random action.

    Additionally stores last_response_text and last_prompt for rollout extraction.
    """

    def __init__(
        self,
        agent_idx: int,
        n_actions: int = 8,
        llm_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(agent_idx=agent_idx, n_actions=n_actions)
        self.agent_name = AGENT_NAMES[agent_idx]
        self.llm_config = llm_config or {}

        # Lazy-load LLM client
        self._llm_client = None
        self._init_llm_client()

        # Last turn state (for rollout extraction)
        self.last_response_text: str = ""
        self.last_prompt: str = ""
        self.last_action_reasoning: str = ""
        self._rollout_turns = []

    def _init_llm_client(self) -> None:
        """Initialize the DeepInfra client."""
        try:
            from ..llm.deepinfra_client import DeepInfraClient
            self._llm_client = DeepInfraClient(config=self.llm_config)
        except ImportError:
            warnings.warn(
                "DeepInfraClient not available. LLMAgent will use random actions.",
                UserWarning,
                stacklevel=2,
            )

    def select_action(self, observation: Dict[str, Any]) -> int:
        """
        Use LLM to select an action by generating a structured JSON response.

        The LLM receives the current observation and is asked to choose
        one of the 8 actions with reasoning.

        Returns:
            action: integer in [0, n_actions)
        """
        if self._llm_client is None:
            return self._random_action()

        prompt = self._build_action_selection_prompt(observation)
        self.last_prompt = prompt

        try:
            # Create a temporary observation for the LLM call
            # We use action=0 (AGREE) as placeholder since we're asking the LLM to decide
            response_text, full_prompt = self._llm_client.generate_agent_response(
                observation=observation,
                action=0,  # placeholder
                agent_name=self.agent_name,
                topic=observation.get("topic", "general discussion"),
                conversation_history=observation.get("conversation_history", []),
            )

            # Actually use the dedicated action-selection prompt
            action, reasoning = self._parse_action_response(response_text)
            self.last_response_text = response_text
            self.last_action_reasoning = reasoning
            return action

        except Exception as e:
            warnings.warn(f"LLMAgent.select_action failed: {e}. Using random action.")
            return self._random_action()

    def _build_action_selection_prompt(self, observation: Dict[str, Any]) -> str:
        """Build a structured prompt asking the LLM to select an action."""
        trust = observation.get("trust_matrix", [[0.5] * 3] * 3)
        agreements = observation.get("agreement_scores", [0.5, 0.5, 0.5])
        current_round = observation.get("current_round", 0)
        topic = observation.get("topic", "general discussion")

        agent_idx = self.agent_idx
        my_trust_row = trust[agent_idx] if hasattr(trust, '__getitem__') else [0.5, 0.5, 0.5]
        my_agreement = float(agreements[agent_idx]) if hasattr(agreements, '__getitem__') else 0.5

        actions_description = (
            "Available actions:\n"
            "  0: AGREE - Express agreement with the current discussion\n"
            "  1: DISAGREE - Express disagreement\n"
            "  2: PERSUADE - Try to persuade others to your position\n"
            "  3: QUESTION - Ask a probing question\n"
            "  4: SUPPORT - Support and reinforce a recent point\n"
            "  5: CHALLENGE - Challenge an assumption or claim\n"
            "  6: PROVIDE_EVIDENCE - Provide evidence or data\n"
            "  7: SEEK_CONSENSUS - Try to find common ground\n"
        )

        return (
            f"You are {self.agent_name} in a discussion about: {topic}\n"
            f"Current round: {current_round}\n"
            f"Your agreement level: {my_agreement:.2f}\n\n"
            f"{actions_description}\n"
            "Based on your personality and the current state, choose the best action.\n"
            'Respond ONLY with valid JSON: {"action": "ACTION_NAME", "reasoning": "brief explanation"}\n'
            "Example: {\"action\": \"SEEK_CONSENSUS\", \"reasoning\": \"Trust is high enough to bridge gaps\"}"
        )

    def _parse_action_response(self, response_text: str) -> Tuple[int, str]:
        """
        Parse LLM response to extract action and reasoning.

        Returns:
            (action_int, reasoning_string)
        """
        # Try to extract JSON from the response
        json_patterns = [
            r'\{[^{}]*"action"[^{}]*\}',
            r'\{.*?\}',
        ]

        for pattern in json_patterns:
            match = re.search(pattern, response_text, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group())
                    action_name = parsed.get("action", "").upper().strip()
                    reasoning = parsed.get("reasoning", "")

                    if action_name in ACTION_NAME_TO_IDX:
                        return ACTION_NAME_TO_IDX[action_name], reasoning
                except json.JSONDecodeError:
                    pass

        # Try to find action name directly in text
        response_upper = response_text.upper()
        for action_name, action_idx in ACTION_NAME_TO_IDX.items():
            if action_name in response_upper:
                return action_idx, f"Extracted from text: {action_name}"

        # Fallback to random
        return self._random_action(), "Fallback: could not parse LLM response"

    def _random_action(self) -> int:
        """Return a random action."""
        return int(np.random.randint(0, self.n_actions))

    def update(
        self,
        observation: Dict[str, Any],
        action: int,
        reward: float,
        next_observation: Dict[str, Any],
        done: bool,
    ) -> None:
        """Record experience and save rollout turn."""
        super().update(observation, action, reward, next_observation, done)

        # Record for rollout extraction
        self._rollout_turns.append(
            {
                "prompt": self.last_prompt,
                "completion": self.last_response_text,
                "reasoning": self.last_action_reasoning,
                "agent_idx": self.agent_idx,
                "action": action,
                "reward": reward,
                "done": done,
            }
        )

    def reset(self) -> None:
        """Reset episode state."""
        super().reset()
        self.last_response_text = ""
        self.last_prompt = ""
        self.last_action_reasoning = ""
        self._rollout_turns = []

    def get_last_rollout_turn(self) -> Dict[str, Any]:
        """
        Return the last turn's rollout data for Atropos compatibility.

        Returns:
            dict with keys: prompt, completion, agent_idx
        """
        if not self._rollout_turns:
            return {
                "prompt": self.last_prompt,
                "completion": self.last_response_text,
                "agent_idx": self.agent_idx,
            }
        return {
            "prompt": self._rollout_turns[-1]["prompt"],
            "completion": self._rollout_turns[-1]["completion"],
            "agent_idx": self.agent_idx,
        }

    def get_all_rollout_turns(self) -> list:
        """Return all rollout turns for the current episode."""
        return list(self._rollout_turns)

    def __repr__(self) -> str:
        llm_status = "connected" if self._llm_client is not None else "unavailable"
        return (
            f"LLMAgent(name={self.agent_name}, "
            f"llm={llm_status}, "
            f"total_reward={self.total_reward:.3f})"
        )
