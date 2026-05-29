# SimuChat — Social Consensus RL Environment

<div align="center">

**Built for the Nous Research × Cerebral Valley RL Environments Hackathon**
📍 San Francisco · May 18, 2025

[![Nous Research](https://img.shields.io/badge/Nous_Research-RL_Hackathon-blue)](https://github.com/NousResearch/atropos)
[![Cerebral Valley](https://img.shields.io/badge/Hosted_By-Cerebral_Valley-purple)](https://cerebralvalley.ai)
[![Atropos](https://img.shields.io/badge/Framework-Atropos-green)](https://github.com/NousResearch/atropos)
[![Python](https://img.shields.io/badge/Python-3.10%2B-orange)](https://python.org)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-Compatible-red)](https://gymnasium.farama.org)

**Author:** Gokul Srinath Seetha Ram
**Team:** Gokul Srinath Seetha Ram · Khamalesh Kumar · Rashmi Elavazhagan

</div>

---

> *"Good environments thrive on domain expertise and thinking outside the box."*
> — Nous Research Hackathon Brief

---

## What Is This?

SimuChat is a **Gymnasium-compatible Reinforcement Learning environment** that trains LLM agents to navigate complex social dynamics — trust-building, consensus formation, persuasion, and disagreement — inside a simulated multi-agent group chat.

Most existing RL environments for LLMs reward **task correctness** (math, code, tool-calling). SimuChat explores an orthogonal axis: **social intelligence** — the ability to steer a group from disagreement toward consensus using only conversational actions as levers.

This was built for the **Nous Research × Cerebral Valley RL Environments Hackathon** (May 18, 2025, San Francisco), hacking on [Atropos](https://github.com/NousResearch/atropos) — Nous Research's async RL environment framework for LLMs.

---

## The Core Idea

Three LLM agents with distinct personalities are placed on a controversial topic (climate change, AI ethics, UBI, etc.). They start with neutral trust and neutral opinions. A policy learns to assign **social actions** to each agent turn — AGREE, DISAGREE, PERSUADE, PROVIDE_EVIDENCE, SEEK_CONSENSUS — to steer the group from disagreement to consensus as efficiently as possible.

The reward is **emergent, not labeled**. No ground truth. Reward flows from trust dynamics, agreement convergence, and a penalty for increasing polarization.

---

## Architecture

```
+------------------------------------------------------------------+
|                     SocialConsensusEnv                           |
|                  (Gymnasium-compatible)                          |
|                                                                  |
|  +----------+  action (0-7)  +-----------------------------+    |
|  |  Policy  | -------------> |         EnvState            |    |
|  | (PPO/DQN |                |  - Trust matrix  (3x3)      |    |
|  | /QLearning <------------- |  - Emotion vecs  (3x8)      |    |
|  | /LLMAgent)  observation   |  - Agreement     (3,)       |    |
|  +----------+                |  - Conversation history     |    |
|                              +---------------+-------------+    |
|                                              |                   |
|                              +---------------v-------------+    |
|                              |      DeepInfra LLM          |    |
|                              |  Llama-3.1-8B-Turbo         |    |
|                              |  (generates agent response) |    |
|                              +---------------+-------------+    |
|                                              |                   |
|                              +---------------v-------------+    |
|                              |      RewardFunction         |    |
|                              |  +1  trust build            |    |
|                              |  +5  consensus reached      |    |
|                              |  +2  evidence shifts group  |    |
|                              |  -1  trust destroyed        |    |
|                              |  -3  polarization rises     |    |
|                              +-----------------------------+    |
+------------------------------------------------------------------+
                          |
         +----------------v----------------+
         |         AtroposAdapter          |
         |  {prompt, completion, reward}   |
         |  -> trainer.add_batch()         |
         |  (Atropos ScoredDataGroup fmt)  |
         +---------------------------------+
```

---

## Agents

| Agent | Personality | Emotional Bias |
|-------|-------------|----------------|
| **Alice** | Empathetic, warm, seeks harmony | Joy, Trust, Calmness |
| **Bob** | Analytical, methodical, evidence-driven | Confidence, Interest, Stability |
| **Charlie** | Bold, assertive, competitive | Optimism, Surprise, Confidence |

---

## State Space

```python
observation = {
    "trust_matrix":     np.float32,  # (3, 3) — pairwise trust between all agents
    "emotion_vectors":  np.float32,  # (3, 8) — 8-dim emotion per agent
    "agreement_scores": np.float32,  # (3,)   — how aligned each agent is
    "current_round":    int,         # episode progress
    "current_agent":    int,         # 0=Alice, 1=Bob, 2=Charlie
}
```

---

## Action Space — 8 Discrete Social Actions

| ID | Action | Effect |
|----|--------|--------|
| 0 | `AGREE` | Validates group direction, builds trust |
| 1 | `DISAGREE` | Introduces tension, reduces trust |
| 2 | `PERSUADE` | Shifts others toward own view |
| 3 | `QUESTION` | Requests evidence/clarification, neutral |
| 4 | `SUPPORT` | Reinforces another agent's position |
| 5 | `CHALLENGE` | Directly disputes a claim |
| 6 | `PROVIDE_EVIDENCE` | Grounds discussion in facts, strong trust builder |
| 7 | `SEEK_CONSENSUS` | Explicitly moves toward agreement |

---

## Reward Function

```
R = sum(trust_build)   +1.0 per agent pair where trust increased > 0.01
  + consensus          +5.0 one-time when all agreements exceed threshold
  + evidence           +2.0 when PROVIDE_EVIDENCE raises group agreement > 0.02
  + sum(conflict)      -1.0 per agent pair where trust dropped > 0.01
  + polarization       -3.0 when std(agreement_scores) increases > 0.05
```

---

## Emotional Contagion Model

After each full round (3 agent turns), emotions propagate via trust-weighted contagion:

```python
# For each agent i, each emotion e:
emotion[i][e] += sum(trust[i][j] * 0.3 * (emotion[j][e] - emotion[i][e]) for j != i)
emotion        *= exp(-0.05)   # exponential decay per turn
```

Agents emotionally synchronize with agents they trust — creating non-stationary, realistic social dynamics that make credit assignment genuinely hard.

---

## Training Results — 50 Episodes Each

| Metric | Q-Learning | DQN | PPO |
|--------|-----------|-----|-----|
| Consensus Rate | 2.0% | 2.0% | **66.0%** |
| Mean Reward | +19.80 | +15.99 | **+36.33** |
| Max Reward | +36.36 | +35.94 | **+51.37** |
| Mean Trust | 0.649 | 0.632 | **0.687** |
| Mean Polarization | 0.071 | 0.077 | **0.050** |
| Avg Rounds/Episode | 9.92 | 9.94 | **7.76** |

PPO learns to exploit the +5 consensus bonus by episode 20, reaching 66% consensus rate by episode 50. Consensus rate climbed 10% -> 30% -> 47% -> 58% -> 66% across training blocks.

---

## Atropos Integration

```python
from simuchat_rl.envs.social_consensus_env import SocialConsensusEnv
from simuchat_rl.atropos.adapter import AtroposAdapter

env = SocialConsensusEnv()
obs, info = env.reset(options={"topic": "climate change"})

# ... run episode ...

# Get Atropos-compatible rollouts
rollouts = env.get_rollout_for_atropos()
# Each turn: {prompt, completion, reward, agent, action, round, metadata}

# Format for trainer
adapter = AtroposAdapter(normalize_rewards=True)
batch = adapter.format_for_trainer([adapter.convert_episode(rollouts)])
trainer.add_batch(batch)  # compatible with Atropos trainer.add_batch()
```

The ScoredDataGroup payload flows:
`environment -> /scored_data -> API queue -> /batch -> trainer`
exactly as the Atropos framework expects.

---

## Quick Start

```bash
git clone https://github.com/gokulsrinaths/Simuchat.git
cd Simuchat

pip install -r simuchat_rl/requirements.txt

# Add your DeepInfra key
echo "DEEPINFRA_API_KEY=your_key_here" > .env
export SIMUCHAT_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo
export PYTHONIOENCODING=utf-8

# Run a single episode
python run_episode.py

# Train
python simuchat_rl/training/train_ppo.py       --episodes 1000
python simuchat_rl/training/train_dqn.py       --episodes 1000
python simuchat_rl/training/train_qlearning.py --episodes 500
```

---

## Supported Models (via DeepInfra)

| Key | Model |
|-----|-------|
| `llama-3.1-8b-turbo` *(default)* | `meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo` |
| `llama-3.1-8b` | `meta-llama/Meta-Llama-3.1-8B-Instruct` |
| `llama-3.1-70b` | `meta-llama/Meta-Llama-3.1-70B-Instruct` |
| `llama-3.3-70b` | `meta-llama/Llama-3.3-70B-Instruct` |
| `qwen-72b` | `Qwen/Qwen2.5-72B-Instruct` |
| `deepseek-v3` | `deepseek-ai/DeepSeek-V3` |

Set via: `export SIMUCHAT_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo`

---

## Project Structure

```
Simuchat/
├── simuchat_rl/                       <- RL environment (hackathon)
│   ├── envs/
│   │   ├── social_consensus_env.py    <- Gymnasium env (reset/step/render)
│   │   ├── state.py                   <- Trust, emotion, agreement dynamics
│   │   └── reward_fn.py               <- 5-component reward function
│   ├── llm/
│   │   └── deepinfra_client.py        <- DeepInfra API + fallback templates
│   ├── agents/
│   │   ├── llm_agent.py               <- LLM-driven JSON action selection
│   │   └── baseline_agents.py         <- Random/Greedy/Consensus/Adversarial
│   ├── training/
│   │   ├── train_qlearning.py         <- Tabular Q-learning
│   │   ├── train_dqn.py               <- DQN + replay buffer + target net
│   │   └── train_ppo.py               <- Clipped PPO + GAE + ActorCritic
│   ├── atropos/
│   │   └── adapter.py                 <- Atropos rollout format adapter
│   ├── metrics/
│   │   └── tracker.py                 <- CSV/JSON/learning curve plots
│   └── evaluation/
│       └── evaluate.py                <- Multi-agent comparison suite
├── environments/hack0/simuchat/       <- Original Atropos env structure
├── checkpoints/                       <- Saved weights from training runs
│   ├── qlearning/                     <- Q-table JSON checkpoints
│   ├── dqn/                           <- DQN .pt checkpoints
│   └── ppo/                           <- PPO .pt checkpoints
└── run_episode.py                     <- Quick demo script
```

---

## Why This Matters for Atropos

Most Atropos environments optimize for **correctness on well-defined tasks** (GSM8K, MBPP, tool-calling). SimuChat opens a new category: **social reasoning under uncertainty** — a capability orthogonal to math and code that's critical for agents operating in multi-agent or human-facing settings.

Key research properties:

- **Non-stationary rewards** — trust and emotion evolve continuously; optimal policy changes mid-episode
- **Emergent coordination** — the +5 consensus bonus creates a coordination game with no explicit signaling
- **Personality-conditioned generation** — three distinct LLM personas prevent rollout mode collapse
- **Sparse high-value signals** — a single PROVIDE_EVIDENCE can shift the whole group; hard credit assignment
- **Adversarial stress-testing** — AdversarialAgent baseline deliberately disrupts trust for robustness eval

---

## Hackathon Context

Submitted to the **Nous Research × Cerebral Valley RL Environments Hackathon**
📍 San Francisco · May 18, 2025 · 10:00 AM – 11:00 PM PDT

Sponsors: xAI · NVIDIA · Nebius · Lambda · Akash Networks · TensorStax · RunPod

Judges from: xAI · MIT · P-1 · Together AI · Nebius · Lambda Labs · Axolotl · Haize Labs ·
Sophont · Mistral · Cursor · Edge AGI · RunPod · NVIDIA · Google · Latent Space Podcast ·
TensorStax · SemiAnalysis · Nous Research

Built on [Atropos](https://github.com/NousResearch/atropos) — Nous Research's async-first RL
environment framework for LLMs.

---

## Citation

```bibtex
@misc{simuchat2025,
  title   = {SimuChat: A Social Consensus RL Environment for Multi-Agent LLMs},
  author  = {Seetha Ram, Gokul Srinath and Kumar, Khamalesh and Elavazhagan, Rashmi},
  year    = {2025},
  note    = {Nous Research x Cerebral Valley RL Environments Hackathon, San Francisco},
  url     = {https://github.com/gokulsrinaths/Simuchat}
}
```

---

<div align="center">

*Built for the Nous Research x Cerebral Valley RL Environments Hackathon · May 18, 2025*

</div>
