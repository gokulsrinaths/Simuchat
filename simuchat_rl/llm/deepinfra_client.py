"""
DeepInfra LLM client for SocialConsensusEnv.
Provides OpenAI-compatible API access to DeepInfra-hosted models.
"""
import os
import time
import json
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

# Load .env from the project root (final_a/.env)
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parents[2] / ".env"
    load_dotenv(dotenv_path=_env_path, override=False)
except ImportError:
    pass

# Supported DeepInfra models
SUPPORTED_MODELS = {
    "llama-3.1-8b-turbo": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "llama-3.1-8b": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "llama-3.1-70b": "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "qwen-72b": "Qwen/Qwen2.5-72B-Instruct",
    "deepseek-v3": "deepseek-ai/DeepSeek-V3",
    "llama-3.3-70b": "meta-llama/Llama-3.3-70B-Instruct",
}

DEFAULT_MODEL = "llama-3.1-8b-turbo"
DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1/openai"

# Agent personality system prompts
AGENT_SYSTEM_PROMPTS = {
    "Alice": (
        "You are Alice, an empathetic and warm conversationalist. "
        "You approach every discussion with deep emotional intelligence, compassion, and a desire to understand others. "
        "You prioritize human connection and mutual understanding. "
        "Your communication style is warm, inclusive, and emotionally resonant. "
        "You seek harmony and are skilled at finding common ground. "
        "Personality traits: empathy=0.9, stability=0.7, boldness=0.3."
    ),
    "Bob": (
        "You are Bob, a methodical and analytical thinker. "
        "You approach discussions with careful logical reasoning, data, and evidence. "
        "You value precision, accuracy, and systematic analysis. "
        "Your communication style is measured, fact-based, and objective. "
        "You prefer structured arguments backed by evidence. "
        "Personality traits: empathy=0.4, stability=0.9, boldness=0.5."
    ),
    "Charlie": (
        "You are Charlie, a bold and decisive communicator. "
        "You approach discussions with confidence, directness, and a willingness to challenge assumptions. "
        "You are not afraid of controversy and believe in cutting through ambiguity. "
        "Your communication style is assertive, energetic, and action-oriented. "
        "You push conversations forward and resist status quo thinking. "
        "Personality traits: empathy=0.5, stability=0.5, boldness=0.9."
    ),
}

# Template responses per agent per action (fallback when API unavailable)
TEMPLATE_RESPONSES = {
    "Alice": {
        "AGREE": "I completely understand and agree with that perspective — it resonates with me deeply and I think it's the right direction.",
        "DISAGREE": "I hear you, but I must gently say I see things quite differently and feel there's an important human dimension we're missing.",
        "PERSUADE": "I truly believe we can find common ground here if we approach this with empathy — our shared values can guide us forward.",
        "QUESTION": "I'm genuinely curious — how does this perspective account for the human impact on those who will be most affected by our decisions?",
        "SUPPORT": "I want to wholeheartedly build on what was just said, because I think it captures something really important that we should explore further.",
        "CHALLENGE": "With deep respect, I'd like to gently challenge that assumption — the human element and emotional impact seem to be missing from this picture.",
        "PROVIDE_EVIDENCE": "Research on human dynamics consistently shows that empathy-driven, inclusive approaches lead to more sustainable and just outcomes for everyone.",
        "SEEK_CONSENSUS": "I believe we're actually much closer than we realize — let's focus on our shared values and what truly unites us on this issue.",
    },
    "Bob": {
        "AGREE": "The empirical evidence supports this position, and logically it represents the most defensible conclusion given the available data.",
        "DISAGREE": "The current evidence base doesn't fully support that claim — we need a more rigorous analytical framework before accepting this conclusion.",
        "PERSUADE": "Consider the systematic analysis: the quantitative indicators clearly support this as the most rational and effective approach available.",
        "QUESTION": "What empirical evidence supports that specific claim? We need to ground this discussion in verifiable, reproducible facts before proceeding.",
        "SUPPORT": "That argument is logically well-structured and aligns robustly with the preponderance of available data and established methodology.",
        "CHALLENGE": "I would challenge that on methodological grounds — the underlying assumptions require more rigorous scrutiny and validation.",
        "PROVIDE_EVIDENCE": "Multiple peer-reviewed studies and meta-analyses confirm this: the statistical evidence is compelling and consistent across different contexts.",
        "SEEK_CONSENSUS": "If we focus systematically on our areas of empirical agreement, we can construct a rational, evidence-based common framework for moving forward.",
    },
    "Charlie": {
        "AGREE": "Absolutely — that's exactly the bold, honest truth that needs to be said loudly and clearly here.",
        "DISAGREE": "No, I fundamentally and strongly disagree — we're dancing around the real issue and that kind of avoidance is completely unacceptable.",
        "PERSUADE": "The facts are unambiguous and we need to act decisively and immediately rather than continuing to hedge and delay.",
        "QUESTION": "Why are we still debating this? What's actually stopping us from committing to bold action right now?",
        "SUPPORT": "Yes! That's exactly the kind of decisive, forward-thinking approach we need — I fully and enthusiastically back that position.",
        "CHALLENGE": "I challenge anyone at this table to mount a coherent defense of the status quo — complacency and incrementalism are simply not options.",
        "PROVIDE_EVIDENCE": "Here's the undeniable, inconvenient reality: the evidence clearly demands a bold, immediate, and comprehensive response from all of us.",
        "SEEK_CONSENSUS": "We can absolutely agree on the core truth here — let's cut through all the noise and make a decisive, unified commitment right now.",
    },
}


class DeepInfraClient:
    """
    Client for DeepInfra's OpenAI-compatible LLM API.

    Supports multiple open-source models via DeepInfra hosting.
    Falls back to template responses when API is unavailable or key is missing.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}

        # Load API key from environment
        self.api_key = os.environ.get("DEEPINFRA_API_KEY", "")
        if not self.api_key:
            warnings.warn(
                "DEEPINFRA_API_KEY not set. DeepInfra client will use template responses. "
                "Set the DEEPINFRA_API_KEY environment variable to enable LLM generation.",
                UserWarning,
                stacklevel=2,
            )
            self.use_api = False
        else:
            self.use_api = True

        # Model selection — accepts short key ("llama-3.1-8b-turbo") or full model ID
        model_key = os.environ.get("SIMUCHAT_MODEL", config.get("model", DEFAULT_MODEL))

        # If user passed a full model ID (contains "/"), use it directly
        if "/" in model_key:
            self.model_key = model_key
            self.model_id = model_key
        elif model_key in SUPPORTED_MODELS:
            self.model_key = model_key
            self.model_id = SUPPORTED_MODELS[model_key]
        else:
            warnings.warn(
                f"Unknown model key '{model_key}'. Falling back to '{DEFAULT_MODEL}'. "
                f"Supported keys: {list(SUPPORTED_MODELS.keys())}",
                UserWarning,
                stacklevel=2,
            )
            self.model_key = DEFAULT_MODEL
            self.model_id = SUPPORTED_MODELS[DEFAULT_MODEL]

        # Generation parameters
        self.temperature = config.get("temperature", 0.8)
        self.max_tokens = config.get("max_tokens", 150)
        self.top_p = config.get("top_p", 0.9)
        self.presence_penalty = config.get("presence_penalty", 0.6)
        self.frequency_penalty = config.get("frequency_penalty", 0.4)
        self.max_retries = config.get("max_retries", 3)
        self.timeout = config.get("timeout", 30)
        self.base_url = config.get("base_url", DEEPINFRA_BASE_URL)

        # Lazy-load OpenAI client
        self._openai_client = None
        if self.use_api:
            self._init_openai_client()

    def _init_openai_client(self) -> None:
        """Initialize the OpenAI client pointed at DeepInfra."""
        try:
            from openai import OpenAI
            self._openai_client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
            )
        except ImportError:
            warnings.warn(
                "openai package not installed. Install with: pip install openai. "
                "Falling back to template responses.",
                UserWarning,
                stacklevel=2,
            )
            self.use_api = False
        except Exception as e:
            warnings.warn(
                f"Failed to initialize OpenAI client: {e}. Falling back to templates.",
                UserWarning,
                stacklevel=2,
            )
            self.use_api = False

    def generate_agent_response(
        self,
        observation: Dict[str, Any],
        action: int,
        agent_name: str,
        topic: str,
        conversation_history: List[Dict[str, Any]],
    ) -> Tuple[str, str]:
        """
        Generate a response for an agent given current observation and action.

        Args:
            observation: current environment observation dict
            action: integer action index
            agent_name: name of the acting agent ("Alice", "Bob", "Charlie")
            topic: current discussion topic
            conversation_history: list of previous turns

        Returns:
            (response_text, full_prompt_string)
        """
        # Map action int to name (inline to avoid circular/relative import)
        _ACTIONS = {0:"AGREE",1:"DISAGREE",2:"PERSUADE",3:"QUESTION",4:"SUPPORT",5:"CHALLENGE",6:"PROVIDE_EVIDENCE",7:"SEEK_CONSENSUS"}
        action_name = _ACTIONS.get(action, action) if isinstance(action, int) else action

        # Build prompt components
        system_prompt = self._build_system_prompt(agent_name, observation)
        user_prompt = self._build_user_prompt(
            agent_name, topic, conversation_history, action_name, observation
        )
        full_prompt = f"SYSTEM: {system_prompt}\n\nUSER: {user_prompt}"

        if not self.use_api or self._openai_client is None:
            return self._get_template_response(agent_name, action_name), full_prompt

        # Attempt API call with retry logic
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self._openai_client.chat.completions.create(
                    model=self.model_id,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                    presence_penalty=self.presence_penalty,
                    frequency_penalty=self.frequency_penalty,
                )
                response_text = response.choices[0].message.content.strip()
                if not response_text:
                    response_text = self._get_template_response(agent_name, action_name)
                return response_text, full_prompt

            except Exception as e:
                last_error = e
                wait_time = (2 ** attempt) * 0.5  # exponential backoff: 0.5s, 1s, 2s
                if attempt < self.max_retries - 1:
                    time.sleep(wait_time)

        # All retries exhausted
        warnings.warn(
            f"DeepInfra API failed after {self.max_retries} attempts: {last_error}. "
            "Using template response.",
            UserWarning,
            stacklevel=2,
        )
        return self._get_template_response(agent_name, action_name), full_prompt

    def _build_system_prompt(
        self,
        agent_name: str,
        observation: Dict[str, Any],
    ) -> str:
        """Build the system prompt including personality and emotional context."""
        base_prompt = AGENT_SYSTEM_PROMPTS.get(agent_name, AGENT_SYSTEM_PROMPTS["Bob"])

        # Add emotional state context
        emotion_context = self._format_emotional_state(agent_name, observation)

        # Add trust context
        trust_context = self._format_trust_context(agent_name, observation)

        return (
            f"{base_prompt}\n\n"
            f"Current Emotional State:\n{emotion_context}\n\n"
            f"Trust Context:\n{trust_context}\n\n"
            "Keep your response to 1-2 sentences. Stay in character. "
            "Be authentic to your personality and current emotional state."
        )

    def _build_user_prompt(
        self,
        agent_name: str,
        topic: str,
        conversation_history: List[Dict[str, Any]],
        action_name: str,
        observation: Dict[str, Any],
    ) -> str:
        """Build user prompt with topic, history, and action instruction."""
        # Format last 3 turns of conversation history
        recent_history = conversation_history[-3:] if conversation_history else []
        history_text = ""
        if recent_history:
            history_lines = []
            for entry in recent_history:
                history_lines.append(
                    f"  {entry['agent']} ({entry['action_name']}): {entry['text']}"
                )
            history_text = "Recent conversation:\n" + "\n".join(history_lines) + "\n\n"

        # Agreement context
        agreements = observation.get("agreement_scores", [0.5, 0.5, 0.5])
        agent_names_list = ["Alice", "Bob", "Charlie"]
        agent_idx = agent_names_list.index(agent_name)
        my_agreement = agreements[agent_idx] if hasattr(agreements, '__getitem__') else 0.5
        if hasattr(my_agreement, 'item'):
            my_agreement = my_agreement.item()

        action_instructions = {
            "AGREE": "Express agreement with the recent discussion point. Show you align with the direction of the conversation.",
            "DISAGREE": "Express a disagreement. Articulate why you see things differently.",
            "PERSUADE": "Try to persuade others toward your position. Make a compelling argument.",
            "QUESTION": "Ask a probing question to deepen understanding or reveal assumptions.",
            "SUPPORT": "Support and reinforce a point that was just made. Add weight to it.",
            "CHALLENGE": "Challenge an assumption or position. Be critical but constructive.",
            "PROVIDE_EVIDENCE": "Provide specific evidence, data, or a concrete example that supports your position.",
            "SEEK_CONSENSUS": "Try to find common ground. Identify shared values or areas of agreement.",
        }
        instruction = action_instructions.get(action_name, f"Take action: {action_name}")

        return (
            f"Topic under discussion: {topic}\n\n"
            f"{history_text}"
            f"Your current agreement level with the group: {my_agreement:.2f}/1.0\n\n"
            f"Your assigned action this turn: {action_name}\n"
            f"Instruction: {instruction}\n\n"
            f"Generate your response as {agent_name}. "
            "Respond naturally in 1-2 sentences. Do not explain your action — just speak."
        )

    def _format_emotional_state(
        self, agent_name: str, observation: Dict[str, Any]
    ) -> str:
        """Format the agent's current emotional state as a readable string."""
        AGENT_NAMES = ["Alice", "Bob", "Charlie"]
        EMOTION_NAMES = ["joy", "trust", "fear", "anger", "optimism", "calmness", "interest", "confidence"]
        emotion_vectors = observation.get("emotion_vectors")
        if emotion_vectors is None:
            return "Emotional state: neutral"

        agent_idx = AGENT_NAMES.index(agent_name) if agent_name in AGENT_NAMES else 0
        emotions = emotion_vectors[agent_idx]

        # Sort by intensity, show top 4
        if hasattr(emotions, '__len__'):
            import numpy as np
            emotions_arr = np.array(emotions)
            top_indices = emotions_arr.argsort()[::-1][:4]
            emotion_strs = [
                f"{EMOTION_NAMES[i]}={float(emotions_arr[i]):.2f}"
                for i in top_indices
            ]
            return "Dominant emotions: " + ", ".join(emotion_strs)
        return "Emotional state: neutral"

    def _format_trust_context(
        self, agent_name: str, observation: Dict[str, Any]
    ) -> str:
        """Format trust relationships as readable string."""
        AGENT_NAMES = ["Alice", "Bob", "Charlie"]
        trust_matrix = observation.get("trust_matrix")
        if trust_matrix is None:
            return "Trust: moderate with all agents"

        agent_idx = AGENT_NAMES.index(agent_name) if agent_name in AGENT_NAMES else 0
        trust_levels = []
        for j, other_name in enumerate(AGENT_NAMES):
            if j != agent_idx:
                trust_val = float(trust_matrix[agent_idx][j]) if hasattr(trust_matrix[agent_idx], '__getitem__') else 0.5
                level = "high" if trust_val > 0.65 else "low" if trust_val < 0.35 else "moderate"
                trust_levels.append(f"{other_name}: {level} ({trust_val:.2f})")

        return "Trust in others — " + ", ".join(trust_levels)

    def _get_template_response(self, agent_name: str, action_name: str) -> str:
        """Return personality-consistent template response."""
        agent_templates = TEMPLATE_RESPONSES.get(agent_name, TEMPLATE_RESPONSES["Bob"])
        return agent_templates.get(action_name, f"{agent_name} responds with action: {action_name}.")

    def build_observation_prompt(
        self, observation: Dict[str, Any], agent_name: str
    ) -> str:
        """
        Build a formatted string describing the current state for the LLM.
        Useful for debugging or external prompt inspection.
        """
        AGENT_NAMES = ["Alice", "Bob", "Charlie"]
        EMOTION_NAMES = ["joy", "trust", "fear", "anger", "optimism", "calmness", "interest", "confidence"]
        import numpy as np

        trust = observation.get("trust_matrix", [[0.5] * 3] * 3)
        emotions = observation.get("emotion_vectors", [[0.5] * 8] * 3)
        agreements = observation.get("agreement_scores", [0.5, 0.5, 0.5])
        current_round = observation.get("current_round", 0)
        topic = observation.get("topic", "unknown topic")

        agent_idx = AGENT_NAMES.index(agent_name) if agent_name in AGENT_NAMES else 0

        lines = [
            f"=== Observation for {agent_name} ===",
            f"Topic: {topic}",
            f"Round: {current_round}",
            "",
            "Trust Matrix:",
        ]
        for i, name in enumerate(AGENT_NAMES):
            trust_row = [f"{float(trust[i][j]):.3f}" for j in range(len(AGENT_NAMES))]
            lines.append(f"  {name}: [{', '.join(trust_row)}]")

        lines.append("")
        lines.append("Agreement Scores:")
        for i, name in enumerate(AGENT_NAMES):
            lines.append(f"  {name}: {float(agreements[i]):.3f}")

        lines.append("")
        lines.append(f"Your Emotions ({agent_name}):")
        agent_emotions = emotions[agent_idx]
        for e_idx, e_name in enumerate(EMOTION_NAMES):
            lines.append(f"  {e_name}: {float(agent_emotions[e_idx]):.3f}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        status = "API connected" if self.use_api else "template mode"
        return f"DeepInfraClient(model={self.model_key}, status={status})"


def generate_agent_response(
    observation: Dict[str, Any],
    action: int,
    agent_name: Optional[str] = None,
    model: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str]:
    """
    Module-level convenience wrapper for generating agent responses.

    Args:
        observation: current environment observation dict
        action: integer action index
        agent_name: name of the acting agent (default: "Bob")
        model: model key override (e.g., "llama-3.1-70b")
        config: optional config dict override

    Returns:
        (response_text, prompt_text)
    """
    config = config or {}
    if model is not None:
        config = {**config, "model": model}

    agent_name = agent_name or "Bob"

    client = DeepInfraClient(config=config)
    return client.generate_agent_response(
        observation=observation,
        action=action,
        agent_name=agent_name,
        topic=observation.get("topic", "general discussion"),
        conversation_history=observation.get("conversation_history", []),
    )
