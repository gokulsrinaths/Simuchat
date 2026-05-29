"""
Comprehensive pytest test suite for SocialConsensusEnv.

Tests cover:
- Environment creation and reset
- Step mechanics and reward bounds
- Observation shapes and types
- Consensus detection
- Atropos rollout format
- MetricsTracker functionality
- Full episode termination
"""
import os
import sys
import tempfile
import numpy as np
import pytest

# Allow running from any working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.social_consensus_env import SocialConsensusEnv, N_ACTIONS, N_AGENTS, N_EMOTIONS, ACTIONS
from envs.state import EnvState, AgentState, AGENT_NAMES, EMOTION_NAMES
from envs.reward_fn import RewardFunction
from metrics.tracker import MetricsTracker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def env():
    """Create a fresh SocialConsensusEnv with LLM disabled for speed."""
    config = {
        "max_rounds": 10,
        "consensus_threshold": 0.75,
        "use_llm": False,
        "verbose": False,
    }
    return SocialConsensusEnv(config=config)


@pytest.fixture
def reset_env(env):
    """Create a fresh env and call reset."""
    obs, info = env.reset(options={"topic": "climate change"})
    return env, obs, info


# ---------------------------------------------------------------------------
# Test 1: Environment creation
# ---------------------------------------------------------------------------

def test_env_creation():
    """SocialConsensusEnv() creates successfully with default config."""
    env = SocialConsensusEnv()
    assert env is not None
    assert env.action_space is not None
    assert env.observation_space is not None
    assert env.max_rounds > 0
    assert env.consensus_threshold > 0


def test_env_creation_with_config():
    """SocialConsensusEnv accepts custom config."""
    config = {
        "max_rounds": 15,
        "consensus_threshold": 0.8,
        "use_llm": False,
    }
    env = SocialConsensusEnv(config=config)
    assert env.max_rounds == 15
    assert env.consensus_threshold == 0.8


# ---------------------------------------------------------------------------
# Test 2: Reset
# ---------------------------------------------------------------------------

def test_reset(env):
    """reset() returns valid obs dict and info dict."""
    obs, info = env.reset(options={"topic": "AI ethics"})

    # Check obs is a dict with required keys
    assert isinstance(obs, dict)
    assert "trust_matrix" in obs
    assert "emotion_vectors" in obs
    assert "agreement_scores" in obs
    assert "current_round" in obs
    assert "current_agent" in obs

    # Check info
    assert isinstance(info, dict)
    assert "topic" in info
    assert info["topic"] == "AI ethics"
    assert "current_round" in info

    # Initial state checks
    assert obs["current_round"] == 0
    assert obs["current_agent"] in range(N_AGENTS)


def test_reset_with_seed(env):
    """reset(seed=...) is accepted without errors."""
    obs, info = env.reset(seed=42, options={"topic": "climate change"})
    assert isinstance(obs, dict)


def test_reset_initializes_state(env):
    """reset() properly initializes the environment state."""
    obs, info = env.reset()
    assert env.state is not None
    assert env.state.current_round == 0
    assert env._episode_reward == 0.0
    assert not env._consensus_was_reached


# ---------------------------------------------------------------------------
# Test 3: Step with all 8 actions
# ---------------------------------------------------------------------------

def test_step_all_actions(env):
    """step() with each of the 8 actions returns valid (obs, reward, term, trunc, info)."""
    for action in range(N_ACTIONS):
        obs, info = env.reset(options={"topic": "cryptocurrency"})
        next_obs, reward, terminated, truncated, step_info = env.step(action)

        # Type checks
        assert isinstance(next_obs, dict)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(step_info, dict)

        # Info must have required keys
        assert "response_text" in step_info
        assert "action_name" in step_info
        assert "consensus_reached" in step_info
        assert "average_trust" in step_info
        assert "episode_reward" in step_info

        # Action name should match
        assert step_info["action_name"] == ACTIONS[action]


def test_step_reward_finite(env):
    """Reward from any step is a finite float."""
    obs, info = env.reset()
    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert np.isfinite(reward), f"Non-finite reward: {reward}"
        if terminated or truncated:
            break


# ---------------------------------------------------------------------------
# Test 4: Full episode terminates
# ---------------------------------------------------------------------------

def test_full_episode_terminates(env):
    """A full episode with random actions terminates within max_rounds."""
    obs, info = env.reset(options={"topic": "immigration policy"})
    max_steps = env.max_rounds * N_AGENTS + 10  # generous upper bound
    steps = 0
    done = False

    while not done and steps < max_steps:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1

    assert done, f"Episode did not terminate after {steps} steps"
    assert steps <= max_steps


# ---------------------------------------------------------------------------
# Test 5: Reward bounds
# ---------------------------------------------------------------------------

def test_reward_bounds(env):
    """Single step reward is finite and within a reasonable range."""
    obs, info = env.reset()
    action = env.action_space.sample()
    _, reward, _, _, _ = env.step(action)

    assert np.isfinite(reward)
    # Reward should not be astronomically large in magnitude
    assert abs(reward) < 100.0, f"Reward {reward} seems unreasonably large"


# ---------------------------------------------------------------------------
# Test 6: Observation shapes
# ---------------------------------------------------------------------------

def test_observation_shapes(env):
    """Observation arrays have correct shapes."""
    obs, info = env.reset()

    trust = obs["trust_matrix"]
    emotions = obs["emotion_vectors"]
    agreement = obs["agreement_scores"]

    assert np.array(trust).shape == (N_AGENTS, N_AGENTS), (
        f"Expected trust_matrix shape ({N_AGENTS}, {N_AGENTS}), got {np.array(trust).shape}"
    )
    assert np.array(emotions).shape == (N_AGENTS, N_EMOTIONS), (
        f"Expected emotion_vectors shape ({N_AGENTS}, {N_EMOTIONS}), got {np.array(emotions).shape}"
    )
    assert np.array(agreement).shape == (N_AGENTS,), (
        f"Expected agreement_scores shape ({N_AGENTS},), got {np.array(agreement).shape}"
    )


def test_observation_value_ranges(env):
    """All observation values are in valid ranges."""
    obs, info = env.reset()

    trust = np.array(obs["trust_matrix"])
    emotions = np.array(obs["emotion_vectors"])
    agreement = np.array(obs["agreement_scores"])

    assert np.all(trust >= 0.0) and np.all(trust <= 1.0), "Trust values out of [0, 1]"
    assert np.all(emotions >= 0.0) and np.all(emotions <= 1.0), "Emotion values out of [0, 1]"
    assert np.all(agreement >= 0.0) and np.all(agreement <= 1.0), "Agreement values out of [0, 1]"


def test_observation_after_step(env):
    """Observation shapes remain valid after steps."""
    obs, info = env.reset()
    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        assert np.array(obs["trust_matrix"]).shape == (N_AGENTS, N_AGENTS)
        assert np.array(obs["emotion_vectors"]).shape == (N_AGENTS, N_EMOTIONS)
        assert np.array(obs["agreement_scores"]).shape == (N_AGENTS,)

        if terminated or truncated:
            break


# ---------------------------------------------------------------------------
# Test 7: Consensus detection
# ---------------------------------------------------------------------------

def test_consensus_detection():
    """Manually set state, assert is_consensus_reached() = True."""
    state = EnvState(topic="test", max_rounds=20)

    # Set all agreement scores above threshold
    state.agreement_scores = np.array([0.95, 0.95, 0.95], dtype=np.float32)

    # Set trust matrix to high values (off-diagonal > 0.6)
    state.trust_matrix = np.array(
        [[1.0, 0.85, 0.85],
         [0.85, 1.0, 0.85],
         [0.85, 0.85, 1.0]],
        dtype=np.float32
    )

    assert state.is_consensus_reached(threshold=0.75), (
        "Expected consensus to be detected with high agreements and trust"
    )


def test_no_consensus_low_agreement():
    """Consensus not reached when agreements are low."""
    state = EnvState(topic="test", max_rounds=20)
    state.agreement_scores = np.array([0.3, 0.3, 0.3], dtype=np.float32)
    assert not state.is_consensus_reached(threshold=0.75)


def test_no_consensus_low_trust():
    """Consensus not reached when trust is low even if agreements are high."""
    state = EnvState(topic="test", max_rounds=20)
    state.agreement_scores = np.array([0.9, 0.9, 0.9], dtype=np.float32)
    state.trust_matrix = np.array(
        [[1.0, 0.1, 0.1],
         [0.1, 1.0, 0.1],
         [0.1, 0.1, 1.0]],
        dtype=np.float32
    )
    assert not state.is_consensus_reached(threshold=0.75)


# ---------------------------------------------------------------------------
# Test 8: Action space
# ---------------------------------------------------------------------------

def test_action_space(env):
    """All 8 action IDs are valid and action_space contains all of them."""
    assert env.action_space.n == N_ACTIONS

    for action_id in range(N_ACTIONS):
        assert env.action_space.contains(action_id), f"Action {action_id} not in action space"

    assert len(ACTIONS) == N_ACTIONS

    expected_actions = {
        "AGREE", "DISAGREE", "PERSUADE", "QUESTION",
        "SUPPORT", "CHALLENGE", "PROVIDE_EVIDENCE", "SEEK_CONSENSUS"
    }
    actual_actions = set(ACTIONS.values())
    assert actual_actions == expected_actions


# ---------------------------------------------------------------------------
# Test 9: Atropos rollout format
# ---------------------------------------------------------------------------

def test_atropos_rollout_format(env):
    """run 3+ steps, call get_rollout_for_atropos(), check structure."""
    obs, info = env.reset(options={"topic": "space exploration"})

    # Take at least 3 steps
    for i in range(3):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, step_info = env.step(action)
        if terminated or truncated:
            break

    rollout = env.get_rollout_for_atropos()

    # Should be a list
    assert isinstance(rollout, list)
    assert len(rollout) >= 1

    # Check structure of each entry
    required_keys = {"prompt", "completion", "reward", "agent", "action", "round", "metadata"}
    for entry in rollout:
        assert isinstance(entry, dict)
        for key in required_keys:
            assert key in entry, f"Missing key '{key}' in rollout entry"

        # Type checks
        assert isinstance(entry["prompt"], str)
        assert isinstance(entry["completion"], str)
        assert isinstance(entry["reward"], (int, float))
        assert isinstance(entry["agent"], str)
        assert isinstance(entry["action"], str)
        assert isinstance(entry["round"], int)
        assert isinstance(entry["metadata"], dict)

        # Agent name should be valid
        assert entry["agent"] in AGENT_NAMES

        # Action should be a valid action name
        assert entry["action"] in ACTIONS.values()


def test_atropos_adapter_conversion(env):
    """AtroposAdapter correctly converts env rollout to AtroposRollout objects."""
    from atropos.adapter import AtroposAdapter, AtroposRollout

    obs, info = env.reset()
    for _ in range(3):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    env_rollout = env.get_rollout_for_atropos()
    adapter = AtroposAdapter(reward_scale=1.0, normalize_rewards=False)
    atropos_rollout = adapter.convert_episode(env_rollout)

    assert isinstance(atropos_rollout, list)
    assert len(atropos_rollout) == len(env_rollout)

    for r in atropos_rollout:
        assert isinstance(r, AtroposRollout)
        assert isinstance(r.prompt, str)
        assert isinstance(r.completion, str)
        assert isinstance(r.reward, float)


# ---------------------------------------------------------------------------
# Test 10: MetricsTracker
# ---------------------------------------------------------------------------

def test_metrics_tracker_records_episodes():
    """MetricsTracker records episodes and get_summary returns correct keys."""
    tracker = MetricsTracker()

    # Record several episodes
    for i in range(10):
        tracker.record_episode(
            consensus_reached=(i % 3 == 0),
            average_trust=0.5 + i * 0.02,
            polarization_score=0.2 - i * 0.01,
            episode_reward=float(i),
            time_to_consensus=20 if (i % 3 == 0) else None,
            n_rounds=10 + i,
            duration=1.0,
        )

    assert len(tracker) == 10

    summary = tracker.get_summary()

    # Required summary keys
    required_keys = [
        "total_episodes",
        "consensus_rate",
        "mean_reward",
        "std_reward",
        "mean_trust",
        "mean_polarization",
        "mean_rounds",
        "mean_time_to_consensus",
    ]
    for key in required_keys:
        assert key in summary, f"Missing key '{key}' in summary"

    assert summary["total_episodes"] == 10
    assert 0.0 <= summary["consensus_rate"] <= 100.0


def test_metrics_tracker_to_csv():
    """MetricsTracker.to_csv() writes a valid CSV file."""
    tracker = MetricsTracker()
    for i in range(5):
        tracker.record_episode(
            consensus_reached=False,
            average_trust=0.5,
            polarization_score=0.2,
            episode_reward=float(i),
            time_to_consensus=None,
            n_rounds=10,
            duration=0.5,
        )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        tmp_path = f.name

    try:
        tracker.to_csv(tmp_path)
        assert os.path.exists(tmp_path)

        # Check file has content
        with open(tmp_path, "r") as f:
            content = f.read()
        assert len(content) > 0
        assert "episode" in content or "consensus_reached" in content
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_metrics_tracker_to_json():
    """MetricsTracker.to_json() writes valid JSON."""
    import json as json_module
    tracker = MetricsTracker()
    tracker.record_episode(
        consensus_reached=True,
        average_trust=0.8,
        polarization_score=0.1,
        episode_reward=10.0,
        time_to_consensus=15,
        n_rounds=8,
        duration=2.0,
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        tmp_path = f.name

    try:
        tracker.to_json(tmp_path)
        assert os.path.exists(tmp_path)

        with open(tmp_path, "r") as f:
            data = json_module.load(f)

        assert "summary" in data
        assert "episodes" in data
        assert len(data["episodes"]) == 1
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_metrics_tracker_empty_summary():
    """MetricsTracker returns default summary when no episodes recorded."""
    tracker = MetricsTracker()
    summary = tracker.get_summary()
    assert summary["total_episodes"] == 0
    assert summary["consensus_rate"] == 0.0


# ---------------------------------------------------------------------------
# Test 11: Reward function
# ---------------------------------------------------------------------------

def test_reward_fn_trust_build():
    """RewardFunction gives positive reward when trust increases."""
    rf = RewardFunction()
    prev_trust = np.array([[1.0, 0.5, 0.5],
                           [0.5, 1.0, 0.5],
                           [0.5, 0.5, 1.0]], dtype=np.float32)
    new_trust = np.array([[1.0, 0.6, 0.6],
                          [0.6, 1.0, 0.6],
                          [0.6, 0.6, 1.0]], dtype=np.float32)
    agreements = np.array([0.5, 0.5, 0.5], dtype=np.float32)

    reward = rf.compute(
        action=0,
        action_name="AGREE",
        prev_trust=prev_trust,
        new_trust=new_trust,
        prev_agreements=agreements,
        new_agreements=agreements,
        agent_idx=0,
        consensus_reached=False,
    )
    assert reward > 0, f"Expected positive reward for trust build, got {reward}"


def test_reward_fn_consensus_bonus():
    """RewardFunction gives +5 bonus when consensus is newly reached."""
    rf = RewardFunction()
    trust = np.full((3, 3), 0.8, dtype=np.float32)
    np.fill_diagonal(trust, 1.0)
    agreements = np.array([0.9, 0.9, 0.9], dtype=np.float32)

    reward = rf.compute(
        action=7,
        action_name="SEEK_CONSENSUS",
        prev_trust=trust,
        new_trust=trust,
        prev_agreements=agreements,
        new_agreements=agreements,
        agent_idx=0,
        consensus_reached=True,  # Newly reached
    )
    assert reward >= rf.consensus_reward, (
        f"Expected consensus reward >= {rf.consensus_reward}, got {reward}"
    )


# ---------------------------------------------------------------------------
# Test 12: State update
# ---------------------------------------------------------------------------

def test_state_update_advances_agent():
    """EnvState.update() correctly advances current_agent_idx."""
    state = EnvState(topic="test", max_rounds=20)

    assert state.current_agent_idx == 0
    state.update(agent_idx=0, action=0, response_text="Hello", action_name="AGREE")
    assert state.current_agent_idx == 1

    state.update(agent_idx=1, action=4, response_text="I support this", action_name="SUPPORT")
    assert state.current_agent_idx == 2


def test_state_update_increments_round():
    """EnvState.update() increments round after every 3 turns."""
    state = EnvState(topic="test", max_rounds=20)

    assert state.current_round == 0

    # 3 turns = 1 full round
    state.update(0, 0, "A", "AGREE")
    state.update(1, 0, "B", "AGREE")
    state.update(2, 0, "C", "AGREE")

    assert state.current_round == 1


def test_state_conversation_history():
    """EnvState correctly records conversation history."""
    state = EnvState(topic="test topic", max_rounds=20)

    state.update(0, 0, "I agree with this", "AGREE")
    state.update(1, 1, "I disagree", "DISAGREE")

    assert len(state.conversation_history) == 2
    assert state.conversation_history[0]["agent"] == "Alice"
    assert state.conversation_history[1]["agent"] == "Bob"
    assert state.conversation_history[0]["action_name"] == "AGREE"
    assert state.conversation_history[1]["action_name"] == "DISAGREE"


# ---------------------------------------------------------------------------
# Test 13: EnvState initialization
# ---------------------------------------------------------------------------

def test_envstate_initial_trust():
    """EnvState initializes trust matrix correctly."""
    state = EnvState()
    trust = state.get_trust_matrix()

    # Off-diagonal should be 0.5
    for i in range(N_AGENTS):
        for j in range(N_AGENTS):
            if i == j:
                assert trust[i][j] == 1.0, f"Diagonal trust[{i}][{j}] should be 1.0"
            else:
                assert abs(trust[i][j] - 0.5) < 1e-6, (
                    f"Off-diagonal trust[{i}][{j}] should be 0.5, got {trust[i][j]}"
                )


def test_envstate_initial_agreements():
    """EnvState initializes agreement scores to 0.5."""
    state = EnvState()
    agreements = state.get_agreement_scores()
    assert np.allclose(agreements, 0.5), f"Expected all agreements=0.5, got {agreements}"


def test_agent_state_personality():
    """AgentState correctly loads personality traits."""
    alice = AgentState("Alice")
    assert alice.empathy == 0.9
    assert alice.stability == 0.7
    assert alice.boldness == 0.3

    bob = AgentState("Bob")
    assert bob.stability == 0.9

    charlie = AgentState("Charlie")
    assert charlie.boldness == 0.9


# ---------------------------------------------------------------------------
# Test 14: get_observation
# ---------------------------------------------------------------------------

def test_get_observation_json_serializable(env):
    """get_observation() returns a JSON-serializable dict."""
    import json as json_module
    obs, info = env.reset()
    full_obs = env.get_observation()

    # Should be serializable
    try:
        json_str = json_module.dumps(full_obs)
        assert len(json_str) > 0
    except TypeError as e:
        pytest.fail(f"get_observation() returned non-serializable data: {e}")


# ---------------------------------------------------------------------------
# Run standalone
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
