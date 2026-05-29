"""
SocialConsensusEnv: Gymnasium-compatible multi-agent social consensus environment.

Agents: Alice (empathetic), Bob (analytical), Charlie (bold)
Task: Reach consensus on a given topic through structured conversation.
LLM backend: DeepInfra via OpenAI-compatible API.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json
import time
from typing import Dict, Any, Optional, Tuple, List

try:
    from colorama import Fore, Style, Back, init as colorama_init
    colorama_init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False

    class _NoColor:
        def __getattr__(self, name):
            return ""

    Fore = _NoColor()
    Style = _NoColor()
    Back = _NoColor()

from .state import EnvState, AGENT_NAMES, EMOTION_NAMES, N_AGENTS, N_EMOTIONS
from .reward_fn import RewardFunction

# Action definitions
ACTIONS = {
    0: "AGREE",
    1: "DISAGREE",
    2: "PERSUADE",
    3: "QUESTION",
    4: "SUPPORT",
    5: "CHALLENGE",
    6: "PROVIDE_EVIDENCE",
    7: "SEEK_CONSENSUS",
}
ACTION_NAMES = list(ACTIONS.values())

N_ACTIONS = 8
DEFAULT_MAX_ROUNDS = 20
DEFAULT_CONSENSUS_THRESHOLD = 0.75

DEFAULT_TOPICS = [
    "climate change",
    "artificial intelligence ethics",
    "universal basic income",
    "space exploration",
    "social media regulation",
]

# Color assignments for agents
AGENT_COLORS = [Fore.CYAN, Fore.YELLOW, Fore.MAGENTA]
ACTION_COLORS = {
    "AGREE": Fore.GREEN,
    "DISAGREE": Fore.RED,
    "PERSUADE": Fore.YELLOW,
    "QUESTION": Fore.BLUE,
    "SUPPORT": Fore.GREEN,
    "CHALLENGE": Fore.RED,
    "PROVIDE_EVIDENCE": Fore.CYAN,
    "SEEK_CONSENSUS": Fore.GREEN,
}


class SocialConsensusEnv(gym.Env):
    """
    A Gymnasium-compatible multi-agent RL environment for social consensus tasks.

    Observation Space (Dict):
        - trust_matrix: Box(3, 3) in [0, 1] — pairwise trust between agents
        - emotion_vectors: Box(3, 8) in [0, 1] — 8 emotions per agent
        - agreement_scores: Box(3,) in [0, 1] — per-agent agreement level
        - current_round: Discrete(101) — current conversation round
        - current_agent: Discrete(3) — index of agent whose turn it is

    Action Space:
        Discrete(8): AGREE, DISAGREE, PERSUADE, QUESTION, SUPPORT,
                     CHALLENGE, PROVIDE_EVIDENCE, SEEK_CONSENSUS

    Reward:
        Shaped reward based on trust dynamics, consensus achievement,
        evidence quality, and polarization avoidance.

    Termination:
        - Consensus reached (all agreements > threshold AND sufficient trust)
        - Max rounds exceeded
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 1}

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        config = config or {}

        self.max_rounds: int = config.get("max_rounds", DEFAULT_MAX_ROUNDS)
        self.consensus_threshold: float = config.get("consensus_threshold", DEFAULT_CONSENSUS_THRESHOLD)
        self.render_mode: str = config.get("render_mode", "human")
        self.use_llm: bool = config.get("use_llm", True)
        self.llm_config: Dict[str, Any] = config.get("llm", {})
        self.verbose: bool = config.get("verbose", False)

        # Define observation space
        self.observation_space = spaces.Dict(
            {
                "trust_matrix": spaces.Box(
                    low=0.0, high=1.0, shape=(N_AGENTS, N_AGENTS), dtype=np.float32
                ),
                "emotion_vectors": spaces.Box(
                    low=0.0, high=1.0, shape=(N_AGENTS, N_EMOTIONS), dtype=np.float32
                ),
                "agreement_scores": spaces.Box(
                    low=0.0, high=1.0, shape=(N_AGENTS,), dtype=np.float32
                ),
                "current_round": spaces.Discrete(self.max_rounds + 1),
                "current_agent": spaces.Discrete(N_AGENTS),
            }
        )

        # Define action space
        self.action_space = spaces.Discrete(N_ACTIONS)

        # Initialize reward function
        self.reward_fn = RewardFunction(
            trust_build_reward=config.get("reward_trust_build", 1.0),
            consensus_reward=config.get("reward_consensus", 5.0),
            evidence_reward=config.get("reward_evidence", 2.0),
            conflict_penalty=config.get("penalty_conflict", -1.0),
            polarization_penalty=config.get("penalty_polarization", -3.0),
        )

        # Lazy-load LLM client (supports both package and flat sys.path imports)
        self.llm_client = None
        if self.use_llm:
            try:
                try:
                    from ..llm.deepinfra_client import DeepInfraClient
                except ImportError:
                    from llm.deepinfra_client import DeepInfraClient
                self.llm_client = DeepInfraClient(config=self.llm_config)
            except ImportError:
                print("[SocialConsensusEnv] WARNING: DeepInfra client not available. Using template responses.")
            except Exception as e:
                print(f"[SocialConsensusEnv] WARNING: Could not initialize LLM client: {e}. Using template responses.")

        # Lazy-load metrics tracker
        try:
            try:
                from ..metrics.tracker import MetricsTracker
            except ImportError:
                from metrics.tracker import MetricsTracker
            self.metrics_tracker = MetricsTracker()
        except ImportError:
            self.metrics_tracker = None

        # Episode state (initialized in reset)
        self.state: Optional[EnvState] = None
        self._episode_reward: float = 0.0
        self._consensus_was_reached: bool = False
        self._episode_start_time: float = 0.0
        self._step_count: int = 0
        self._rollout_buffer: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset environment to initial state.

        Args:
            seed: random seed for reproducibility
            options: optional dict, may contain 'topic' key

        Returns:
            observation: initial observation dict
            info: info dict with topic and initial state summary
        """
        super().reset(seed=seed)

        options = options or {}
        topic = options.get("topic", DEFAULT_TOPICS[0])

        # Create fresh state
        self.state = EnvState(topic=topic, max_rounds=self.max_rounds)
        self._episode_reward = 0.0
        self._consensus_was_reached = False
        self._episode_start_time = time.time()
        self._step_count = 0
        self._rollout_buffer = []

        obs = self._get_observation()
        info = {
            "topic": topic,
            "current_agent": AGENT_NAMES[self.state.current_agent_idx],
            "current_round": self.state.current_round,
            "state_summary": self.state.to_dict(),
        }

        return obs, info

    def step(
        self, action: int
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        Execute one agent turn.

        Args:
            action: integer in [0, N_ACTIONS)

        Returns:
            observation: next observation dict
            reward: float reward for this step
            terminated: True if consensus reached or max rounds exceeded
            truncated: False (never truncated by timeout here)
            info: info dict with step details
        """
        assert self.state is not None, "Call reset() before step()"
        assert self.action_space.contains(action), f"Invalid action: {action}"

        action_name = ACTIONS[action]
        agent_idx = self.state.current_agent_idx
        agent_name = AGENT_NAMES[agent_idx]

        # Snapshot state before update
        prev_trust = self.state.get_trust_matrix()
        prev_agreements = self.state.get_agreement_scores()

        # Generate LLM response for the current agent
        response_text, prompt_text = self._generate_response(
            agent_idx=agent_idx,
            action=action,
            action_name=action_name,
            agent_name=agent_name,
        )

        # Update environment state
        self.state.update(
            agent_idx=agent_idx,
            action=action,
            response_text=response_text,
            action_name=action_name,
        )

        # Fill in the prompt in the last history entry
        if self.state.conversation_history:
            self.state.conversation_history[-1]["prompt"] = prompt_text

        # Compute reward
        new_trust = self.state.get_trust_matrix()
        new_agreements = self.state.get_agreement_scores()
        consensus_reached_now = self.state.is_consensus_reached(self.consensus_threshold)
        consensus_newly_reached = consensus_reached_now and not self._consensus_was_reached

        reward = self.reward_fn.compute(
            action=action,
            action_name=action_name,
            prev_trust=prev_trust,
            new_trust=new_trust,
            prev_agreements=prev_agreements,
            new_agreements=new_agreements,
            agent_idx=agent_idx,
            consensus_reached=consensus_newly_reached,
        )

        if consensus_newly_reached:
            self._consensus_was_reached = True

        self._episode_reward += reward
        self._step_count += 1

        # Update reward in history
        if self.state.conversation_history:
            self.state.conversation_history[-1]["reward"] = reward

        # Build rollout entry for Atropos
        self._rollout_buffer.append(
            {
                "prompt": prompt_text,
                "completion": response_text,
                "reward": reward,
                "agent": agent_name,
                "action": action_name,
                "round": self.state.current_round,
                "metadata": {
                    "agent_idx": agent_idx,
                    "action_id": action,
                    "trust_mean": float(np.mean(new_trust[~np.eye(N_AGENTS, dtype=bool)])),
                    "agreement_mean": float(np.mean(new_agreements)),
                    "consensus_reached": consensus_reached_now,
                    "step": self._step_count,
                },
            }
        )

        # Termination conditions
        terminated = bool(
            consensus_reached_now or self.state.current_round >= self.max_rounds
        )
        truncated = False

        if terminated:
            # Episode-level bonus
            episode_bonus = self.reward_fn.compute_episode_bonus(
                final_trust=new_trust,
                final_agreements=new_agreements,
                n_rounds_taken=self.state.current_round,
                max_rounds=self.max_rounds,
                consensus_reached=self._consensus_was_reached,
            )
            reward += episode_bonus
            self._episode_reward += episode_bonus
            self._record_episode_metrics(consensus_reached=self._consensus_was_reached)

        obs = self._get_observation()

        # Compute polarization
        polarization = self._compute_polarization()
        avg_trust = float(np.mean(new_trust[~np.eye(N_AGENTS, dtype=bool)]))

        info = {
            "response_text": response_text,
            "prompt": prompt_text,
            "action_name": action_name,
            "agent_name": agent_name,
            "agent_idx": agent_idx,
            "consensus_reached": consensus_reached_now,
            "consensus_newly_reached": consensus_newly_reached,
            "average_trust": avg_trust,
            "polarization_score": polarization,
            "episode_reward": self._episode_reward,
            "step": self._step_count,
            "round": self.state.current_round,
            "agreement_scores": new_agreements.tolist(),
            "trust_matrix": new_trust.tolist(),
        }

        if self.verbose:
            self._print_step_info(agent_name, action_name, response_text, reward, info)

        return obs, reward, terminated, truncated, info

    def render(self, mode: str = "human") -> Optional[str]:
        """Render current state to terminal using colorama."""
        assert self.state is not None, "Call reset() before render()"

        trust = self.state.get_trust_matrix()
        emotions = self.state.get_emotion_vectors()
        agreements = self.state.get_agreement_scores()

        lines = []
        sep = "=" * 65

        lines.append(f"\n{Fore.WHITE}{Style.BRIGHT}{sep}")
        lines.append(
            f"{Fore.WHITE}{Style.BRIGHT}  SocialConsensusEnv | "
            f"Round {self.state.current_round}/{self.state.max_rounds} | "
            f"Topic: {self.state.topic}"
        )
        lines.append(f"{Fore.WHITE}{Style.BRIGHT}{sep}")

        # Trust matrix
        lines.append(f"\n{Fore.CYAN}{Style.BRIGHT}  Trust Matrix:")
        header = "         " + "  ".join(f"{n:>7}" for n in AGENT_NAMES)
        lines.append(f"{Fore.CYAN}  {header}")
        for i, row_name in enumerate(AGENT_NAMES):
            row_vals = "  ".join(
                f"{Fore.GREEN if trust[i][j] > 0.6 else Fore.RED if trust[i][j] < 0.4 else Fore.YELLOW}{trust[i][j]:.4f}{Fore.CYAN}"
                for j in range(N_AGENTS)
            )
            lines.append(f"{Fore.CYAN}  {row_name:>7}: {row_vals}")

        # Agreement scores
        lines.append(f"\n{Fore.YELLOW}{Style.BRIGHT}  Agreement Scores:")
        for i, name in enumerate(AGENT_NAMES):
            bar_len = int(agreements[i] * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            color = Fore.GREEN if agreements[i] > 0.7 else Fore.YELLOW if agreements[i] > 0.4 else Fore.RED
            lines.append(f"  {name:>7}: {color}[{bar}]{Style.RESET_ALL} {agreements[i]:.3f}")

        # Key emotions
        lines.append(f"\n{Fore.MAGENTA}{Style.BRIGHT}  Emotions (top 3 per agent):")
        for i, name in enumerate(AGENT_NAMES):
            top_indices = np.argsort(emotions[i])[::-1][:3]
            emotion_strs = [
                f"{EMOTION_NAMES[idx]}={emotions[i][idx]:.2f}" for idx in top_indices
            ]
            lines.append(f"  {AGENT_COLORS[i]}{name:>7}: {', '.join(emotion_strs)}")

        # Last message
        if self.state.conversation_history:
            last = self.state.conversation_history[-1]
            agent_color = AGENT_COLORS[last["agent_idx"]]
            action_color = ACTION_COLORS.get(last["action_name"], Fore.WHITE)
            lines.append(f"\n{Fore.WHITE}{Style.BRIGHT}  Last Message:")
            lines.append(
                f"  {agent_color}{last['agent']}{Style.RESET_ALL} "
                f"[{action_color}{last['action_name']}{Style.RESET_ALL}]: "
                f"{last['text']}"
            )

        # Episode stats
        lines.append(f"\n{Fore.WHITE}  Episode Reward: {self._episode_reward:+.3f}")
        consensus_str = (
            f"{Fore.GREEN}YES" if self._consensus_was_reached else f"{Fore.RED}NO"
        )
        lines.append(f"  Consensus: {consensus_str}{Style.RESET_ALL}")
        lines.append(f"{Fore.WHITE}{Style.BRIGHT}{sep}\n")

        output = "\n".join(lines)
        if mode == "human":
            print(output)
        return output

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def _get_observation(self) -> Dict[str, Any]:
        """Build observation dict from current state."""
        return {
            "trust_matrix": self.state.get_trust_matrix(),
            "emotion_vectors": self.state.get_emotion_vectors(),
            "agreement_scores": self.state.get_agreement_scores(),
            "current_round": int(self.state.current_round),
            "current_agent": int(self.state.current_agent_idx),
        }

    def get_observation(self) -> Dict[str, Any]:
        """Return full structured JSON-serializable observation dict."""
        assert self.state is not None, "Call reset() before get_observation()"
        obs = self._get_observation()
        return {
            "trust_matrix": obs["trust_matrix"].tolist(),
            "emotion_vectors": obs["emotion_vectors"].tolist(),
            "agreement_scores": obs["agreement_scores"].tolist(),
            "current_round": obs["current_round"],
            "current_agent": obs["current_agent"],
            "agent_names": AGENT_NAMES,
            "emotion_names": EMOTION_NAMES,
            "topic": self.state.topic,
            "consensus_reached": self.state.is_consensus_reached(self.consensus_threshold),
            "conversation_summary": self.state.get_conversation_summary(5),
        }

    # ------------------------------------------------------------------
    # LLM response generation
    # ------------------------------------------------------------------

    def _generate_response(
        self,
        agent_idx: int,
        action: int,
        action_name: str,
        agent_name: str,
    ) -> Tuple[str, str]:
        """Generate agent response via LLM or fallback templates."""
        obs = self._get_observation()

        if self.llm_client is not None:
            try:
                response_text, prompt_text = self.llm_client.generate_agent_response(
                    observation=obs,
                    action=action,
                    agent_name=agent_name,
                    topic=self.state.topic,
                    conversation_history=self.state.conversation_history,
                )
                return response_text, prompt_text
            except Exception as e:
                if self.verbose:
                    print(f"[SocialConsensusEnv] LLM error for {agent_name}: {e}. Using fallback.")

        # Fallback template response
        return self._get_template_response(agent_name, action_name), self._get_template_prompt(
            agent_name, action_name, self.state.topic
        )

    def _get_template_response(self, agent_name: str, action_name: str) -> str:
        """Return personality-consistent template response."""
        templates = {
            "Alice": {
                "AGREE": "I completely understand and agree with that perspective — it resonates with me on a deep level.",
                "DISAGREE": "I hear you, but I must gently say I see things differently and feel there's another side to this.",
                "PERSUADE": "I truly believe we can find common ground if we approach this with empathy and openness.",
                "QUESTION": "I'm curious — how does that perspective account for the human impact on those affected most?",
                "SUPPORT": "I want to build on what was just said, because I think it's a really important point.",
                "CHALLENGE": "With respect, I'd like to challenge that assumption — the human element seems to be missing here.",
                "PROVIDE_EVIDENCE": "Research consistently shows that empathy-driven approaches lead to more sustainable outcomes.",
                "SEEK_CONSENSUS": "I think we're all actually closer than we realize — let's focus on what unites us here.",
            },
            "Bob": {
                "AGREE": "The data supports this position, and logically it makes the most sense given what we know.",
                "DISAGREE": "The evidence doesn't fully support that claim — we should look at this more analytically.",
                "PERSUADE": "Consider the systematic analysis: the numbers clearly indicate this is the optimal approach.",
                "QUESTION": "What empirical evidence supports that claim? We need to ground this in verifiable facts.",
                "SUPPORT": "That argument is logically sound and aligns with the available data.",
                "CHALLENGE": "I'd challenge that on methodological grounds — the underlying assumptions need scrutiny.",
                "PROVIDE_EVIDENCE": "Multiple peer-reviewed studies confirm this: the statistical evidence is quite compelling.",
                "SEEK_CONSENSUS": "If we focus on the areas of empirical agreement, we can build a rational common framework.",
            },
            "Charlie": {
                "AGREE": "Absolutely — that's exactly the bold truth that needs to be said here.",
                "DISAGREE": "No, I fundamentally disagree — we're dancing around the real issue and that's unacceptable.",
                "PERSUADE": "Look, the facts are clear and we need to act decisively rather than hedging.",
                "QUESTION": "But why are we still debating this? What's actually stopping us from taking action?",
                "SUPPORT": "Yes! That's the kind of decisive thinking we need — I fully back that position.",
                "CHALLENGE": "I challenge anyone to defend the status quo here — complacency is not an option.",
                "PROVIDE_EVIDENCE": "Here's the undeniable reality: the evidence demands a bold, immediate response.",
                "SEEK_CONSENSUS": "We can agree on the core truth here — let's cut through the noise and commit.",
            },
        }
        return templates.get(agent_name, {}).get(action_name, f"{agent_name} takes action: {action_name}.")

    def _get_template_prompt(self, agent_name: str, action_name: str, topic: str) -> str:
        return (
            f"[Template] Agent: {agent_name} | Action: {action_name} | Topic: {topic}"
        )

    # ------------------------------------------------------------------
    # Atropos compatibility
    # ------------------------------------------------------------------

    def get_rollout_for_atropos(self) -> List[Dict[str, Any]]:
        """
        Return rollout buffer in Atropos-compatible format.
        Each entry: {prompt, completion, reward, agent, action, round, metadata}
        """
        return list(self._rollout_buffer)

    # ------------------------------------------------------------------
    # Metrics helpers
    # ------------------------------------------------------------------

    def _compute_polarization(self) -> float:
        """Standard deviation of agreement scores — measures polarization."""
        return float(np.std(self.state.get_agreement_scores()))

    def _record_episode_metrics(self, consensus_reached: bool) -> None:
        """Record episode-level metrics to the tracker."""
        if self.metrics_tracker is None:
            return

        duration = time.time() - self._episode_start_time
        trust = self.state.get_trust_matrix()
        mask = ~np.eye(N_AGENTS, dtype=bool)
        avg_trust = float(np.mean(trust[mask]))
        polarization = self._compute_polarization()

        # Time to consensus: number of turns until consensus (if reached)
        time_to_consensus = None
        if consensus_reached:
            time_to_consensus = self._step_count

        self.metrics_tracker.record_episode(
            consensus_reached=consensus_reached,
            average_trust=avg_trust,
            polarization_score=polarization,
            episode_reward=self._episode_reward,
            time_to_consensus=time_to_consensus,
            n_rounds=self.state.current_round,
            duration=duration,
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _print_step_info(
        self,
        agent_name: str,
        action_name: str,
        response_text: str,
        reward: float,
        info: Dict[str, Any],
    ) -> None:
        """Print verbose step info."""
        agent_idx = AGENT_NAMES.index(agent_name)
        color = AGENT_COLORS[agent_idx]
        action_color = ACTION_COLORS.get(action_name, Fore.WHITE)
        reward_color = Fore.GREEN if reward >= 0 else Fore.RED
        print(
            f"{color}{agent_name}{Style.RESET_ALL} "
            f"[{action_color}{action_name}{Style.RESET_ALL}] "
            f"(R={reward_color}{reward:+.2f}{Style.RESET_ALL}): "
            f"{response_text[:80]}{'...' if len(response_text) > 80 else ''}"
        )

    def get_action_name(self, action: int) -> str:
        return ACTIONS.get(action, "UNKNOWN")

    def get_state_summary(self) -> Dict[str, Any]:
        """Return JSON-serializable state summary."""
        assert self.state is not None
        return self.state.to_dict()

    def __repr__(self) -> str:
        status = "not initialized" if self.state is None else (
            f"round={self.state.current_round}/{self.max_rounds}, "
            f"agent={AGENT_NAMES[self.state.current_agent_idx]}, "
            f"consensus={'YES' if self._consensus_was_reached else 'NO'}"
        )
        return f"SocialConsensusEnv({status})"
