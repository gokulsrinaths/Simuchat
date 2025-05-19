"""
Reward system for SimuChat agents.
Handles tracking rewards, insights, and trust improvements.
"""

from typing import Dict, List, Any, Optional
import env

class RewardSystem:
    """Class to manage agent rewards."""
    
    def __init__(self):
        """Initialize the reward system."""
        self.agent_rewards = {}
        self.reward_history = {}
        self.reset_rewards()
    
    def reset_rewards(self):
        """Reset all agent rewards to zero."""
        for agent_name in env.get_all_agent_names():
            self.agent_rewards[agent_name] = 0
            self.reward_history[agent_name] = []
    
    def add_reward(self, agent_name: str, points: int, reason: str) -> None:
        """
        Add reward points to an agent.
        
        Args:
            agent_name: The name of the agent
            points: Number of points to add
            reason: Reason for giving the reward
        """
        if agent_name in self.agent_rewards:
            self.agent_rewards[agent_name] += points
            
            # Add to history
            self.reward_history[agent_name].append({
                "points": points,
                "reason": reason,
                "total": self.agent_rewards[agent_name]
            })
    
    def process_message_rewards(
        self, 
        agent_name: str, 
        has_insight: bool,
        trust_changes: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Process rewards for a message from an agent.
        
        Args:
            agent_name: The name of the agent who sent the message
            has_insight: Whether this message contains an insight
            trust_changes: Dict of trust changes from other agents
            
        Returns:
            Dict with rewards information
        """
        rewards_info = {
            "agent": agent_name,
            "rewards_earned": 0,
            "reasons": []
        }
        
        # Check for insight reward
        if has_insight:
            self.add_reward(agent_name, 2, "Triggered an insight moment")
            rewards_info["rewards_earned"] += 2
            rewards_info["reasons"].append("Insight moment (+2)")
        
        # Check for trust increases toward this agent
        for other_agent, changes in trust_changes.items():
            if other_agent != agent_name:  # Skip self
                agent_changes = changes.get(agent_name, {})
                
                if "change" in agent_changes and agent_changes["change"] > 0:
                    self.add_reward(agent_name, 1, f"Increased trust from {other_agent}")
                    rewards_info["rewards_earned"] += 1
                    rewards_info["reasons"].append(f"Trust increase from {other_agent} (+1)")
        
        return rewards_info
    
    def get_agent_rewards(self, agent_name: str) -> int:
        """
        Get the current rewards for an agent.
        
        Args:
            agent_name: The name of the agent
            
        Returns:
            The agent's current reward points
        """
        return self.agent_rewards.get(agent_name, 0)
    
    def get_agent_rewards_history(self, agent_name: str) -> List[Dict[str, Any]]:
        """
        Get the reward history for an agent.
        
        Args:
            agent_name: The name of the agent
            
        Returns:
            List of reward events for the agent
        """
        return self.reward_history.get(agent_name, [])
    
    def get_all_rewards(self) -> Dict[str, int]:
        """
        Get rewards for all agents.
        
        Returns:
            Dict mapping agent names to reward points
        """
        return self.agent_rewards
    
    def get_reward_summary(self) -> Dict[str, Any]:
        """
        Get a summary of rewards for all agents.
        
        Returns:
            Dict with reward summary information
        """
        total_points = sum(self.agent_rewards.values())
        leading_agent = max(self.agent_rewards.items(), key=lambda x: x[1])[0] if self.agent_rewards else None
        
        return {
            "agent_rewards": self.agent_rewards.copy(),
            "total_points": total_points,
            "leading_agent": leading_agent
        }
    
    def get_reward_context(self, agent_name: str) -> str:
        """
        Get reward context text for an agent to include in prompts.
        
        Args:
            agent_name: The name of the agent
            
        Returns:
            Text describing the agent's reward status
        """
        total_reward = self.get_agent_rewards(agent_name)
        all_rewards = self.get_all_rewards()
        rank_list = sorted(all_rewards.items(), key=lambda x: x[1], reverse=True)
        
        # Find agent's rank
        rank = 1
        for name, points in rank_list:
            if name == agent_name:
                break
            rank += 1
        
        recent_rewards = self.get_agent_rewards_history(agent_name)[-3:] if self.get_agent_rewards_history(agent_name) else []
        
        # Generate context text
        context = f"You have earned {total_reward} points so far (rank #{rank})."
        
        if recent_rewards:
            context += " Recent rewards: "
            context += ", ".join([f"{r['points']} points for {r['reason']}" for r in recent_rewards])
        
        return context 