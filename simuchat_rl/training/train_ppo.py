"""
Proximal Policy Optimization (PPO) for SocialConsensusEnv.

Implements:
- Clipped PPO objective (Schulman et al. 2017)
- Generalized Advantage Estimation (GAE, lambda=0.95)
- Shared Actor-Critic backbone with orthogonal initialization
- Multiple epochs per rollout batch
- Gradient clipping

Usage:
    python train_ppo.py --episodes 1000 --max_rounds 20 --save_dir checkpoints/ppo
"""
import numpy as np
import argparse
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.social_consensus_env import SocialConsensusEnv, N_ACTIONS, N_AGENTS
from metrics.tracker import MetricsTracker
from training.train_dqn import flatten_observation, OBS_DIM, TOPICS

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Categorical
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[train_ppo] ERROR: PyTorch not installed. Install with: pip install torch")
    sys.exit(1)


class ActorCritic(nn.Module):
    """
    Shared Actor-Critic network for PPO.

    Architecture:
    - Shared backbone: Linear(40→256) → LN → Tanh → Linear(256→256) → LN → Tanh
    - Actor head: Linear(256→8) [policy logits]
    - Critic head: Linear(256→1) [value estimate]

    Orthogonal initialization with different gains for each part.
    """

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        n_actions: int = N_ACTIONS,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_dim, n_actions)
        self.critic = nn.Linear(hidden_dim, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        """Orthogonal initialization for stable PPO training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        # Actor: small gain for near-uniform initial policy
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)
        # Critic: standard gain
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def forward(self, x: "torch.Tensor"):
        """Forward pass returning (logits, value)."""
        shared = self.shared(x)
        logits = self.actor(shared)
        value = self.critic(shared).squeeze(-1)
        return logits, value

    def get_action_and_value(self, x: "torch.Tensor", action=None):
        """
        Sample action and compute log prob, entropy, and value.

        Args:
            x: observation tensor [batch, obs_dim]
            action: optional pre-specified action tensor (for PPO update)

        Returns:
            (action, log_prob, entropy, value)
        """
        logits, value = self(x)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value


class PPORolloutBuffer:
    """
    Buffer for storing on-policy rollout data for PPO updates.
    Stores one rollout batch before each PPO update.
    """

    def __init__(self):
        self.clear()

    def clear(self) -> None:
        """Reset the buffer."""
        self.observations: list = []
        self.actions: list = []
        self.log_probs: list = []
        self.rewards: list = []
        self.values: list = []
        self.dones: list = []

    def add(
        self,
        obs: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        value: float,
        done: float,
    ) -> None:
        """Add one transition to the buffer."""
        self.observations.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def compute_advantages(
        self,
        last_value: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        """
        Compute GAE advantages and returns.

        GAE: A_t = sum_{l=0}^{inf} (gamma * lambda)^l * delta_{t+l}
        where delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)

        Returns:
            (advantages, returns) as numpy arrays
        """
        rewards = np.array(self.rewards, dtype=np.float64)
        values_extended = np.array(self.values + [last_value], dtype=np.float64)
        dones = np.array(self.dones, dtype=np.float64)

        advantages = np.zeros_like(rewards)
        last_gae = 0.0

        for t in reversed(range(len(rewards))):
            delta = (
                rewards[t]
                + gamma * values_extended[t + 1] * (1.0 - dones[t])
                - values_extended[t]
            )
            advantages[t] = last_gae = (
                delta + gamma * gae_lambda * (1.0 - dones[t]) * last_gae
            )

        returns = advantages + np.array(self.values, dtype=np.float64)
        return advantages.astype(np.float32), returns.astype(np.float32)

    def get_tensors(self, device):
        """Convert stored data to tensors for the PPO update."""
        return (
            torch.FloatTensor(np.array(self.observations)).to(device),
            torch.LongTensor(np.array(self.actions)).to(device),
            torch.FloatTensor(np.array(self.log_probs)).to(device),
        )

    def __len__(self) -> int:
        return len(self.rewards)


class PPOAgent:
    """
    PPO agent with clipped objective and GAE.

    Key hyperparameters:
    - clip_eps (0.2): clipping range for policy ratio
    - entropy_coef (0.01): entropy bonus for exploration
    - value_coef (0.5): weight for critic loss
    - n_epochs (4): gradient updates per rollout batch
    - gae_lambda (0.95): GAE smoothing parameter
    """

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        n_actions: int = N_ACTIONS,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        n_epochs: int = 4,
        batch_size: int = 64,
        max_grad_norm: float = 0.5,
        hidden_dim: int = 256,
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = ActorCritic(obs_dim, n_actions, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr, eps=1e-5)

        self.rollout_buffer = PPORolloutBuffer()
        self.update_count: int = 0
        self._all_losses: list = []

    def select_action(self, obs_flat: np.ndarray):
        """
        Sample action from current policy.

        Returns:
            (action_int, log_prob_float, value_float)
        """
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs_flat).unsqueeze(0).to(self.device)
            action, log_prob, _, value = self.net.get_action_and_value(obs_t)
        return int(action.item()), float(log_prob.item()), float(value.item())

    def update(self, last_obs_flat: np.ndarray) -> float:
        """
        Perform PPO update using collected rollout buffer.

        Steps:
        1. Bootstrap last value for GAE
        2. Compute advantages and returns
        3. Run n_epochs of mini-batch gradient updates
        4. Clear buffer

        Returns:
            mean loss across all mini-batch updates
        """
        with torch.no_grad():
            last_obs_t = torch.FloatTensor(last_obs_flat).unsqueeze(0).to(self.device)
            _, _, _, last_value = self.net.get_action_and_value(last_obs_t)
            last_value_scalar = float(last_value.item())

        advantages, returns = self.rollout_buffer.compute_advantages(
            last_value_scalar, self.gamma, self.gae_lambda
        )

        # Normalize advantages
        advantages_t = torch.FloatTensor(advantages).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        obs_t, actions_t, old_log_probs_t = self.rollout_buffer.get_tensors(self.device)

        total_losses = []
        n_samples = len(self.rollout_buffer)

        for epoch in range(self.n_epochs):
            indices = np.random.permutation(n_samples)

            for start in range(0, n_samples, self.batch_size):
                batch_idx = indices[start: start + self.batch_size]
                if len(batch_idx) < 2:
                    continue

                batch_obs = obs_t[batch_idx]
                batch_actions = actions_t[batch_idx]
                batch_old_log_probs = old_log_probs_t[batch_idx]
                batch_advantages = advantages_t[batch_idx]
                batch_returns = returns_t[batch_idx]

                _, new_log_probs, entropy, new_values = self.net.get_action_and_value(
                    batch_obs, batch_actions
                )

                # Policy loss (clipped surrogate objective)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
                    * batch_advantages
                )
                actor_loss = -torch.min(surr1, surr2).mean()

                # Value loss (MSE between value predictions and returns)
                critic_loss = nn.MSELoss()(new_values, batch_returns)

                # Entropy bonus (encourages exploration)
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    actor_loss
                    + self.value_coef * critic_loss
                    + self.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_losses.append(float(loss.item()))

        self.rollout_buffer.clear()
        self.update_count += 1

        mean_loss = float(np.mean(total_losses)) if total_losses else 0.0
        self._all_losses.append(mean_loss)
        return mean_loss

    def save(self, filepath: str) -> None:
        """Save model checkpoint."""
        torch.save(
            {
                "net": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "update_count": self.update_count,
            },
            filepath,
        )

    def load(self, filepath: str) -> None:
        """Load model checkpoint."""
        ckpt = torch.load(filepath, map_location=self.device)
        self.net.load_state_dict(ckpt["net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.update_count = ckpt["update_count"]


def train(
    n_episodes: int = 1000,
    rollout_steps: int = 256,
    max_rounds: int = 20,
    save_dir: str = "checkpoints/ppo",
    eval_every: int = 50,
    render_every: int = 200,
) -> tuple:
    """
    Train PPO agent on SocialConsensusEnv.

    Args:
        n_episodes: number of training episodes
        rollout_steps: steps to collect before each PPO update
        max_rounds: maximum rounds per episode
        save_dir: checkpoint directory
        eval_every: print stats every N episodes
        render_every: render env every N episodes

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
    agent = PPOAgent()
    tracker = MetricsTracker()

    print(f"{'=' * 65}")
    print(f"  PPO Training — SocialConsensusEnv")
    print(f"{'=' * 65}")
    print(f"  Device:         {agent.device}")
    print(f"  Network params: {sum(p.numel() for p in agent.net.parameters()):,}")
    print(f"  Obs dim:        {OBS_DIM}")
    print(f"  Episodes:       {n_episodes}")
    print(f"  Rollout steps:  {rollout_steps}")
    print(f"  PPO epochs:     {agent.n_epochs}")
    print(f"  Clip eps:       {agent.clip_eps}")
    print(f"{'=' * 65}\n")

    episode_rewards = []
    steps_since_update = 0

    for episode in range(n_episodes):
        topic = TOPICS[episode % len(TOPICS)]
        obs, info = env.reset(options={"topic": topic})
        obs_flat = flatten_observation(obs)
        episode_reward = 0.0
        done = False
        last_loss = 0.0

        while not done:
            action, log_prob, value = agent.select_action(obs_flat)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_obs_flat = flatten_observation(next_obs)

            agent.rollout_buffer.add(
                obs_flat, action, log_prob, reward, value, float(done)
            )
            steps_since_update += 1
            episode_reward += reward
            obs_flat = next_obs_flat

            # PPO update when rollout buffer is full
            if steps_since_update >= rollout_steps:
                last_loss = agent.update(obs_flat)
                steps_since_update = 0

        # Update at episode end if buffer has data
        if len(agent.rollout_buffer) > 0:
            last_loss = agent.update(obs_flat)
            steps_since_update = 0

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
            summary = tracker.get_summary()
            print(
                f"Ep {episode + 1:4d}/{n_episodes} | "
                f"Reward: {np.mean(recent_rewards):+.3f}±{np.std(recent_rewards):.3f} | "
                f"Loss: {last_loss:.4f} | "
                f"PPO updates: {agent.update_count} | "
                f"Consensus: {summary.get('consensus_rate', 0):.1f}%"
            )
            agent.save(os.path.join(save_dir, f"ppo_ep{episode + 1}.pt"))

        if (episode + 1) % render_every == 0:
            env.render()

    # Final save
    agent.save(os.path.join(save_dir, "ppo_final.pt"))
    tracker.to_csv(os.path.join(save_dir, "metrics.csv"))
    tracker.to_json(os.path.join(save_dir, "metrics.json"))
    try:
        tracker.plot_learning_curve(os.path.join(save_dir, "learning_curve.png"))
    except Exception as e:
        print(f"[Warning] Could not plot: {e}")

    summary = tracker.get_summary()
    print(f"\n{'=' * 65}")
    print(f"  PPO Training complete — Final Summary")
    print(f"{'=' * 65}")
    print(json.dumps(summary, indent=2))

    return agent, tracker


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO on SocialConsensusEnv")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--max_rounds", type=int, default=20)
    parser.add_argument("--rollout_steps", type=int, default=256)
    parser.add_argument("--save_dir", type=str, default="checkpoints/ppo")
    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument("--render_every", type=int, default=200)
    args = parser.parse_args()

    train(
        n_episodes=args.episodes,
        rollout_steps=args.rollout_steps,
        max_rounds=args.max_rounds,
        save_dir=args.save_dir,
        eval_every=args.eval_every,
        render_every=args.render_every,
    )
