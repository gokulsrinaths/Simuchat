# SocialConsensusEnv: A Gymnasium-Compatible RL Environment for Multi-Agent Social Dynamics

## Abstract

SocialConsensusEnv is a Gymnasium-compatible reinforcement learning environment that simulates structured social discourse between three LLM-powered agents (Alice, Bob, and Charlie) with distinct personalities, emotional states, and trust relationships. The environment is designed to advance research at the intersection of multi-agent RL and large language models — specifically, training policies that guide agents toward social consensus through constructive dialogue. It provides rich, shaped rewards based on trust dynamics, agreement evolution, and polarization avoidance, making it an ideal testbed for GRPO/PPO-style LLM fine-tuning via the Atropos framework.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    SocialConsensusEnv Loop                       │
│                                                                  │
│   ┌────────────┐      action (int)     ┌────────────────────┐   │
│   │  RL Policy │ ──────────────────►  │  SocialConsensus   │   │
│   │ (DQN/PPO/  │                       │      Env           │   │
│   │  GRPO/LLM) │ ◄─────────────────── │                    │   │
│   └────────────┘   observation dict   │  ┌──────────────┐  │   │
│                    + reward           │  │  EnvState    │  │   │
│                                       │  │  - Trust 3x3 │  │   │
│   ┌────────────┐                      │  │  - Emotions  │  │   │
│   │ DeepInfra  │ ◄─────────────────── │  │    3x8       │  │   │
│   │    LLM     │   prompt (agent,     │  │  - Agreement │  │   │
│   │  Backend   │    action, topic,    │  │    3-vector  │  │   │
│   └─────┬──────┘    history)          │  └──────────────┘  │   │
│         │                             │                    │   │
│         │  response_text              │  ┌──────────────┐  │   │
│         └────────────────────────►    │  │ RewardFn     │  │   │
│                                       │  │ trust_build  │  │   │
│                                       │  │ consensus    │  │   │
│   ┌──────────────┐                    │  │ evidence     │  │   │
│   │   Atropos    │ ◄─────────────────  │  │ polarization │  │   │
│   │   Adapter    │  rollout buffer    │  └──────────────┘  │   │
│   │ (GRPO/PPO)   │  (prompt,          └────────────────────┘   │
│   └──────────────┘   completion,                               │
│                       reward)                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## Installation

```bash
# Clone or navigate to the project directory
cd path/to/final_a/simuchat_rl

# Install dependencies
pip install -r requirements.txt

# For GPU training (optional)
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## Quick Start

```python
from simuchat_rl import SocialConsensusEnv

# Create environment
env = SocialConsensusEnv(config={
    "max_rounds": 20,
    "consensus_threshold": 0.75,
    "use_llm": False,  # Set True to use DeepInfra LLM
})

# Run an episode
obs, info = env.reset(options={"topic": "climate change"})
print(f"Starting topic: {info['topic']}")

total_reward = 0.0
done = False

while not done:
    action = env.action_space.sample()  # or use your trained agent
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    done = terminated or truncated
    
    print(f"  {info['agent_name']} [{info['action_name']}]: {info['response_text']}")
    print(f"  Reward: {reward:+.2f} | Avg Trust: {info['average_trust']:.3f}")

print(f"\nEpisode complete! Total reward: {total_reward:.2f}")
print(f"Consensus reached: {info['consensus_reached']}")

# Render final state
env.render()

# Get rollout for Atropos training
rollout = env.get_rollout_for_atropos()
print(f"Rollout turns: {len(rollout)}")
```

---

## Environment Specification

| Property | Value |
|----------|-------|
| Observation Space | Dict (see below) |
| Action Space | Discrete(8) |
| Reward Range | [-10, +12] approx per step |
| Max Steps | max_rounds × 3 (default: 60) |
| Agents | Alice (empathetic), Bob (analytical), Charlie (bold) |
| Turn Order | Round-robin: Alice → Bob → Charlie → Alice → ... |
| Termination | Consensus reached OR max_rounds exceeded |

### Observation Space

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `trust_matrix` | (3, 3) | float32 ∈ [0,1] | Pairwise trust between agents |
| `emotion_vectors` | (3, 8) | float32 ∈ [0,1] | 8 emotions per agent |
| `agreement_scores` | (3,) | float32 ∈ [0,1] | Per-agent agreement level |
| `current_round` | scalar | int ∈ [0, max_rounds] | Current round number |
| `current_agent` | scalar | int ∈ {0,1,2} | Index of agent whose turn it is |

**Emotions (8):** joy, trust, fear, anger, optimism, calmness, interest, confidence

---

## Action Descriptions

| ID | Name | Description |
|----|------|-------------|
| 0 | `AGREE` | Express agreement with the current discussion direction |
| 1 | `DISAGREE` | Express principled disagreement |
| 2 | `PERSUADE` | Attempt to persuade others toward your position |
| 3 | `QUESTION` | Ask a probing question to deepen understanding |
| 4 | `SUPPORT` | Support and reinforce a point recently made |
| 5 | `CHALLENGE` | Challenge an assumption or claim critically |
| 6 | `PROVIDE_EVIDENCE` | Offer evidence, data, or concrete examples |
| 7 | `SEEK_CONSENSUS` | Actively seek common ground among agents |

---

## Reward Function

| Component | Value | Trigger |
|-----------|-------|---------|
| Trust Build | +1.0 per agent | Each agent whose trust increases > 0.01 |
| Consensus Bonus | +5.0 (one-time) | When consensus is newly achieved |
| Evidence Reward | +2.0 | `PROVIDE_EVIDENCE` that increases mean agreement > 0.02 |
| Conflict Penalty | -1.0 per agent | Each agent whose trust decreases > 0.01 |
| Polarization Penalty | -3.0 | std(agreement_scores) increases > 0.05 |
| Episode Efficiency | +0 to +2.0 | Bonus for reaching consensus faster |
| Trust Level Bonus | +0 to +0.5 | Final mean trust × 0.5 |

**Consensus condition:** All agreement_scores > threshold AND mean off-diagonal trust > threshold × 0.8

---

## Agent Personalities

| Agent | Empathy | Stability | Boldness | Dominant Trait |
|-------|---------|-----------|----------|----------------|
| Alice | 0.9 | 0.7 | 0.3 | Empathetic communicator, seeks harmony |
| Bob | 0.4 | 0.9 | 0.5 | Analytical, evidence-driven, methodical |
| Charlie | 0.5 | 0.5 | 0.9 | Bold, decisive, challenges status quo |

---

## Training Scripts

### Q-Learning (Tabular Baseline)

```bash
python training/train_qlearning.py \
    --episodes 500 \
    --max_rounds 20 \
    --save_dir checkpoints/qlearning \
    --eval_every 50
```

### Deep Q-Network (DQN)

```bash
python training/train_dqn.py \
    --episodes 1000 \
    --max_rounds 20 \
    --save_dir checkpoints/dqn \
    --eval_every 50 \
    --warmup 500
```

### Proximal Policy Optimization (PPO)

```bash
python training/train_ppo.py \
    --episodes 1000 \
    --max_rounds 20 \
    --rollout_steps 256 \
    --save_dir checkpoints/ppo \
    --eval_every 50
```

### Evaluation Suite

```bash
python evaluation/evaluate.py
```

---

## DeepInfra LLM Setup

Set environment variables to enable LLM-generated responses:

```bash
# Required: your DeepInfra API key
export DEEPINFRA_API_KEY="your_api_key_here"

# Optional: model selection (default: llama-3.1-8b)
export SIMUCHAT_MODEL="llama-3.1-8b"
# Other options: llama-3.1-70b, qwen-72b, deepseek-v3, llama-3.3-70b
```

Without `DEEPINFRA_API_KEY`, the environment falls back to personality-consistent template responses — fully functional for RL training, just without real LLM generation.

```python
# With LLM enabled
env = SocialConsensusEnv(config={"use_llm": True})

# Without LLM (template responses, faster)
env = SocialConsensusEnv(config={"use_llm": False})
```

---

## Atropos Integration

SocialConsensusEnv is designed for compatibility with the [Atropos](https://github.com/NovaSky-AI/atropos) GRPO/PPO trainer for LLM fine-tuning.

```python
from simuchat_rl import SocialConsensusEnv
from simuchat_rl.atropos import AtroposAdapter

env = SocialConsensusEnv(config={"use_llm": True, "max_rounds": 20})
adapter = AtroposAdapter(reward_scale=1.0, normalize_rewards=True)

# Collect episodes
all_episodes = []
for _ in range(32):  # batch of 32 episodes
    obs, info = env.reset()
    done = False
    while not done:
        action = policy.select_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    
    # Convert to Atropos format
    env_rollout = env.get_rollout_for_atropos()
    atropos_rollout = adapter.convert_episode(env_rollout)
    all_episodes.append(atropos_rollout)

# Format for trainer
trainer_batch = adapter.format_for_trainer(all_episodes)
# trainer_batch has keys: prompts, completions, rewards, metadata

# Compatible with Atropos trainer.add_batch()
# trainer.add_batch(**trainer_batch)

# Save rollouts for offline training
adapter.save_rollouts(all_episodes, "rollouts/episode_batch.json")
```

---

## Metrics

The `MetricsTracker` class records per-episode metrics and supports CSV/JSON export and matplotlib visualization.

```python
from simuchat_rl.metrics import MetricsTracker

tracker = MetricsTracker()
# ... training loop records episodes via env._record_episode_metrics()

# Get summary statistics
print(tracker.get_summary())

# Export
tracker.to_csv("results/metrics.csv")
tracker.to_json("results/metrics.json")
tracker.plot_learning_curve("results/learning_curve.png")
```

### Summary Keys

| Key | Description |
|-----|-------------|
| `total_episodes` | Total episodes recorded |
| `consensus_rate` | % of episodes that reached consensus |
| `mean_reward` | Mean episode reward |
| `std_reward` | Std of episode rewards |
| `mean_trust` | Mean final off-diagonal trust |
| `mean_polarization` | Mean final std(agreement_scores) |
| `mean_rounds` | Mean rounds per episode |
| `mean_time_to_consensus` | Mean steps to consensus (consensus episodes only) |

---

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=simuchat_rl --cov-report=html

# Run specific test
pytest tests/test_env.py::test_consensus_detection -v
```

---

## Project Structure

```
simuchat_rl/
├── __init__.py
├── envs/
│   ├── __init__.py
│   ├── social_consensus_env.py   # Main Gymnasium environment
│   ├── state.py                  # EnvState, AgentState
│   └── reward_fn.py              # RewardFunction
├── llm/
│   ├── __init__.py
│   └── deepinfra_client.py       # DeepInfra API client
├── agents/
│   ├── __init__.py
│   ├── base_agent.py             # Abstract BaseAgent
│   ├── llm_agent.py              # LLM-powered agent
│   └── baseline_agents.py        # Random, GreedyTrust, ConsensusSeeking, Adversarial
├── metrics/
│   ├── __init__.py
│   └── tracker.py                # MetricsTracker
├── atropos/
│   ├── __init__.py
│   └── adapter.py                # Atropos compatibility layer
├── training/
│   ├── train_qlearning.py        # Tabular Q-learning
│   ├── train_dqn.py              # Deep Q-Network
│   └── train_ppo.py              # Proximal Policy Optimization
├── evaluation/
│   ├── __init__.py
│   └── evaluate.py               # Evaluation suite
├── configs/
│   └── default.yaml              # Default configuration
├── tests/
│   ├── __init__.py
│   └── test_env.py               # Pytest test suite (14 test groups)
├── requirements.txt
└── README.md
```

---

## Citation

If you use SocialConsensusEnv in your research, please cite:

```bibtex
@software{simuchat_rl2025,
  title  = {SocialConsensusEnv: A Gymnasium-Compatible RL Environment for Multi-Agent Social Dynamics},
  year   = {2025},
  note   = {DeepInfra LLM integration and Atropos GRPO compatibility},
  url    = {https://github.com/your-repo/simuchat_rl}
}
```
