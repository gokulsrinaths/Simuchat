"""
Deep Q-Network (DQN) training for SocialConsensusEnv.

Implements:
- DQN with experience replay buffer
- Target network with periodic hard updates
- Epsilon-greedy exploration with decay
- LayerNorm-stabilized neural network
- Gradient clipping

Usage:
    python train_dqn.py --episodes 1000 --max_rounds 20 --save_dir checkpoints/dqn
"""
import numpy as np
import argparse
import os
import sys
import json
from collections import deque
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.social_consensus_env import SocialConsensusEnv, N_ACTIONS, N_AGENTS, N_EMOTIONS
from metrics.tracker import MetricsTracker

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[train_dqn] ERROR: PyTorch not installed. Install with: pip install torch")
    sys.exit(1)

TOPICS = [
    "climate change",
    "artificial intelligence ethics",
    "universal basic income",
    "space exploration",
    "social media regulation",
    "nuclear energy",
    "vaccine mandates",
    "cryptocurrency",
    "remote work",
    "immigration policy",
]

# Observation dimensionality
# trust_matrix (3x3=9) + emotion_vectors (3x8=24) + agreement_scores (3) + round_norm (1) + agent_onehot (3) = 40
OBS_DIM = 9 + 24 + 3 + 1 + 3  # = 40


def flatten_observation(obs: dict) -> np.ndarray:
    """
    Flatten structured observation dict to a 1D float32 numpy array.

    Layout:
    - trust_matrix (9): flattened 3x3 matrix
    - emotion_vectors (24): flattened 3x8 matrix
    - agreement_scores (3)
    - round_norm (1): current_round / max_rounds (normalized to [0,1])
    - agent_onehot (3): one-hot encoding of current agent index

    Total: 40 dimensions
    """
    trust_flat = np.array(obs["trust_matrix"], dtype=np.float32).flatten()        # 9
    emotion_flat = np.array(obs["emotion_vectors"], dtype=np.float32).flatten()   # 24
    agreement = np.array(obs["agreement_scores"], dtype=np.float32)               # 3
    round_norm = np.array([obs["current_round"] / 20.0], dtype=np.float32)        # 1
    agent_onehot = np.zeros(N_AGENTS, dtype=np.float32)
    agent_onehot[int(obs["current_agent"])] = 1.0                                  # 3

    return np.concatenate([trust_flat, emotion_flat, agreement, round_norm, agent_onehot])


class DQNNetwork(nn.Module):
    """
    Deep Q-Network with LayerNorm stabilization.

    Architecture: Linear(40→256) → LN → ReLU → Linear(256→256) → LN → ReLU
                  → Linear(256→128) → ReLU → Linear(128→8)
    """

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        n_actions: int = N_ACTIONS,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_actions),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        # Output layer: smaller gain for more conservative initial Q-values
        last_linear = [m for m in self.modules() if isinstance(m, nn.Linear)][-1]
        nn.init.orthogonal_(last_linear.weight, gain=0.01)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.net(x)


class ReplayBuffer:
    """
    Experience replay buffer for DQN.
    Stores (obs, action, reward, next_obs, done) tuples.
    """

    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: float,
    ) -> None:
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int):
        """Sample a random mini-batch."""
        batch = random.sample(self.buffer, batch_size)
        obs_b, actions_b, rewards_b, next_obs_b, dones_b = zip(*batch)
        return (
            torch.FloatTensor(np.array(obs_b)),
            torch.LongTensor(np.array(actions_b)),
            torch.FloatTensor(np.array(rewards_b)),
            torch.FloatTensor(np.array(next_obs_b)),
            torch.FloatTensor(np.array(dones_b)),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Network agent with:
    - Experience replay
    - Target network (hard update every target_update_freq steps)
    - Epsilon-greedy exploration
    - Gradient clipping (max norm = 10.0)
    """

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        n_actions: int = N_ACTIONS,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.997,
        epsilon_min: float = 0.05,
        batch_size: int = 64,
        target_update_freq: int = 100,
        buffer_size: int = 10000,
        hidden_dim: int = 256,
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Networks
        self.policy_net = DQNNetwork(obs_dim, n_actions, hidden_dim).to(self.device)
        self.target_net = DQNNetwork(obs_dim, n_actions, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_size)

        self.update_count: int = 0
        self.losses: list = []

    def select_action(self, obs_flat: np.ndarray) -> int:
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return int(np.random.randint(0, self.n_actions))
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs_flat).unsqueeze(0).to(self.device)
            q_values = self.policy_net(obs_tensor)
            return int(q_values.argmax().item())

    def push_experience(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: float,
    ) -> None:
        """Push transition to replay buffer."""
        self.replay_buffer.push(obs, action, reward, next_obs, done)

    def update(self) -> float | None:
        """
        Sample a mini-batch from replay buffer and perform one gradient update.

        Returns:
            loss value if update was performed, else None
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        obs_b, actions_b, rewards_b, next_obs_b, dones_b = self.replay_buffer.sample(
            self.batch_size
        )

        obs_b = obs_b.to(self.device)
        actions_b = actions_b.to(self.device)
        rewards_b = rewards_b.to(self.device)
        next_obs_b = next_obs_b.to(self.device)
        dones_b = dones_b.to(self.device)

        # Current Q-values
        current_q = (
            self.policy_net(obs_b).gather(1, actions_b.unsqueeze(1)).squeeze(1)
        )

        # Target Q-values (using target network, no gradient)
        with torch.no_grad():
            next_q = self.target_net(next_obs_b).max(1)[0]
            target_q = rewards_b + self.gamma * next_q * (1.0 - dones_b)

        # Huber loss (more robust than MSE for RL)
        loss = nn.SmoothL1Loss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        self.update_count += 1

        # Periodic hard target network update
        if self.update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        loss_val = float(loss.item())
        self.losses.append(loss_val)
        return loss_val

    def decay_epsilon(self) -> None:
        """Decay epsilon after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath: str) -> None:
        """Save model checkpoint."""
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "update_count": self.update_count,
                "n_losses": len(self.losses),
            },
            filepath,
        )

    def load(self, filepath: str) -> None:
        """Load model checkpoint."""
        ckpt = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(ckpt["policy_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = ckpt["epsilon"]
        self.update_count = ckpt["update_count"]


def train(
    n_episodes: int = 1000,
    max_rounds: int = 20,
    save_dir: str = "checkpoints/dqn",
    eval_every: int = 50,
    render_every: int = 200,
    warmup_steps: int = 500,
) -> tuple:
    """
    Train DQN agent on SocialConsensusEnv.

    Args:
        n_episodes: number of training episodes
        max_rounds: maximum rounds per episode
        save_dir: checkpoint directory
        eval_every: print stats every N episodes
        render_every: render env every N episodes
        warmup_steps: steps before starting gradient updates (fills replay buffer)

    Returns:
        (agent, tracker)
    """
    os.makedirs(save_dir, exist_ok=True)

    config = {
        "max_rounds": max_rounds,
        "consensus_threshold": 0.7,
        "use_llm": False,
    }
    env = SocialConsensusEnv(config=config)
    agent = DQNAgent()
    tracker = MetricsTracker()

    print(f"{'=' * 65}")
    print(f"  DQN Training — SocialConsensusEnv")
    print(f"{'=' * 65}")
    print(f"  Device:        {agent.device}")
    print(f"  Policy params: {sum(p.numel() for p in agent.policy_net.parameters()):,}")
    print(f"  Obs dim:       {OBS_DIM}")
    print(f"  Episodes:      {n_episodes}")
    print(f"  Warmup steps:  {warmup_steps}")
    print(f"{'=' * 65}\n")

    episode_rewards = []
    total_steps = 0

    for episode in range(n_episodes):
        topic = TOPICS[episode % len(TOPICS)]
        obs, info = env.reset(options={"topic": topic})
        obs_flat = flatten_observation(obs)
        episode_reward = 0.0
        episode_losses = []
        done = False

        while not done:
            action = agent.select_action(obs_flat)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_obs_flat = flatten_observation(next_obs)

            agent.push_experience(obs_flat, action, reward, next_obs_flat, float(done))

            # Only start learning after warmup
            if total_steps >= warmup_steps:
                loss = agent.update()
                if loss is not None:
                    episode_losses.append(loss)

            episode_reward += reward
            obs_flat = next_obs_flat
            total_steps += 1

        agent.decay_epsilon()
        episode_rewards.append(episode_reward)

        # Record metrics
        trust_matrix = np.array(info.get("trust_matrix", [[0.5] * 3] * 3))
        mask = ~np.eye(trust_matrix.shape[0], dtype=bool)
        avg_trust = float(np.mean(trust_matrix[mask]))

        tracker.record_episode(
            consensus_reached=bool(info.get("consensus_reached", False)),
            average_trust=avg_trust,
            polarization_score=float(info.get("polarization_score", 0.0)),
            episode_reward=episode_reward,
            time_to_consensus=info.get("step") if info.get("consensus_reached") else None,
            n_rounds=int(info.get("round", max_rounds)),
            duration=0.0,
        )

        if (episode + 1) % eval_every == 0:
            recent_rewards = episode_rewards[-eval_every:]
            mean_loss = float(np.mean(episode_losses)) if episode_losses else 0.0
            summary = tracker.get_summary()
            print(
                f"Ep {episode + 1:4d}/{n_episodes} | "
                f"Reward: {np.mean(recent_rewards):+.3f}±{np.std(recent_rewards):.3f} | "
                f"Loss: {mean_loss:.4f} | "
                f"Eps: {agent.epsilon:.3f} | "
                f"Steps: {total_steps:,} | "
                f"Buffer: {len(agent.replay_buffer):,} | "
                f"Consensus: {summary.get('consensus_rate', 0):.1f}%"
            )
            agent.save(os.path.join(save_dir, f"dqn_ep{episode + 1}.pt"))

        if (episode + 1) % render_every == 0:
            env.render()

    # Final save
    agent.save(os.path.join(save_dir, "dqn_final.pt"))
    tracker.to_csv(os.path.join(save_dir, "metrics.csv"))
    tracker.to_json(os.path.join(save_dir, "metrics.json"))
    try:
        tracker.plot_learning_curve(os.path.join(save_dir, "learning_curve.png"))
    except Exception as e:
        print(f"[Warning] Could not plot: {e}")

    summary = tracker.get_summary()
    print(f"\n{'=' * 65}")
    print(f"  DQN Training complete — Final Summary")
    print(f"{'=' * 65}")
    print(json.dumps(summary, indent=2))

    return agent, tracker


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN on SocialConsensusEnv")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--max_rounds", type=int, default=20)
    parser.add_argument("--save_dir", type=str, default="checkpoints/dqn")
    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument("--render_every", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=500)
    args = parser.parse_args()

    train(
        n_episodes=args.episodes,
        max_rounds=args.max_rounds,
        save_dir=args.save_dir,
        eval_every=args.eval_every,
        render_every=args.render_every,
        warmup_steps=args.warmup,
    )
