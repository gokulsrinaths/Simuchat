"""
Trust engine for SimuChat.
Handles trust relationships between agents and trust score updates.
"""

import random
import numpy as np
from typing import Dict, List, Tuple, Any
import env


class TrustEngine:
    """Engine to manage trust relationships between agents."""
    
    def __init__(self):
        """Initialize the trust engine."""
        self.agent_names = env.get_all_agent_names()
        self.trust_matrix = {}
        self.initialize_trust()
    
    def initialize_trust(self):
        """Initialize all agent trust relationships."""
        for agent1 in self.agent_names:
            self.trust_matrix[agent1] = {}
            agent1_config = env.get_agent_config(agent1)
            initial_trust = agent1_config.get("initial_trust", 0.5)
            
            for agent2 in self.agent_names:
                if agent1 != agent2:
                    self.trust_matrix[agent1][agent2] = initial_trust
    
    def get_trust(self, agent1: str, agent2: str) -> float:
        """Get the trust score from agent1 to agent2.
        
        Args:
            agent1: The agent that trusts
            agent2: The agent that is trusted
            
        Returns:
            Trust score from 0.0 to 1.0
        """
        if agent1 == agent2:
            return 1.0  # Agents fully trust themselves
        
        return self.trust_matrix.get(agent1, {}).get(agent2, 0.5)
    
    def update_trust(self, agent1: str, agent2: str, messages: List[Dict[str, Any]]) -> Tuple[float, str]:
        """Update trust between two agents based on their recent messages.
        
        Args:
            agent1: The agent that trusts
            agent2: The agent that is trusted
            messages: Recent messages from both agents
            
        Returns:
            Tuple of (trust_change, reason)
        """
        if agent1 == agent2:
            return 0.0, "self"
            
        # Find the most recent messages from both agents
        agent1_msg = None
        agent2_msg = None
        
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and "metadata" in msg:
                agent_name = msg["metadata"].get("agent_name")
                if agent_name == agent1 and agent1_msg is None:
                    agent1_msg = msg
                elif agent_name == agent2 and agent2_msg is None:
                    agent2_msg = msg
                    
            if agent1_msg and agent2_msg:
                break
        
        # If we don't have messages from both agents, no trust update
        if not agent1_msg or not agent2_msg:
            return 0.0, "no_messages"
        
        # Calculate text similarity to determine if agents are aligned or in conflict
        content1 = agent1_msg["content"].lower()
        content2 = agent2_msg["content"].lower()
        
        # Very simple heuristic for detecting agreement/disagreement
        # Could be improved with more sophisticated NLP techniques
        agreement_words = ["agree", "yes", "right", "correct", "exactly", "true", "good point"]
        disagreement_words = ["disagree", "no", "wrong", "incorrect", "false", "not true", "flawed"]
        
        agreement_score = sum(1 for word in agreement_words if word in content1)
        disagreement_score = sum(1 for word in disagreement_words if word in content1)
        
        # If agent1 mentions agent2, check for agreement/disagreement
        if agent2.lower() in content1.lower():
            if agreement_score > disagreement_score:
                trust_change = random.uniform(0.03, 0.08)
                reason = "agreement"
            elif disagreement_score > agreement_score:
                trust_change = random.uniform(-0.08, -0.03)
                reason = "disagreement"
            else:
                trust_change = random.uniform(-0.02, 0.02)
                reason = "neutral_mention"
        else:
            # Check content similarity
            words1 = set(content1.split())
            words2 = set(content2.split())
            intersection = words1.intersection(words2)
            
            if len(intersection) > 3:  # Arbitrary threshold for content similarity
                trust_change = random.uniform(0.01, 0.05)
                reason = "similar_content"
            else:
                trust_change = random.uniform(-0.02, 0.02)
                reason = "different_content"
        
        # Apply the trust change
        current_trust = self.get_trust(agent1, agent2)
        new_trust = max(0.0, min(1.0, current_trust + trust_change))
        self.trust_matrix[agent1][agent2] = new_trust
        
        return trust_change, reason
    
    def update_all_trust(self, messages: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Update trust between all agent pairs.
        
        Args:
            messages: Recent messages from all agents
            
        Returns:
            Dictionary mapping agent pairs to trust changes
        """
        trust_changes = {}
        
        for agent1 in self.agent_names:
            trust_changes[agent1] = {}
            for agent2 in self.agent_names:
                if agent1 != agent2:
                    change, reason = self.update_trust(agent1, agent2, messages)
                    trust_changes[agent1][agent2] = {
                        "change": change,
                        "reason": reason,
                        "new_value": self.get_trust(agent1, agent2)
                    }
        
        return trust_changes
    
    def get_agent_trust_summary(self, agent_name: str) -> Dict[str, float]:
        """Get a summary of trust scores for an agent.
        
        Args:
            agent_name: The agent to get trust summary for
            
        Returns:
            Dictionary mapping other agents to trust scores
        """
        trust_summary = {}
        for other_agent in self.agent_names:
            if agent_name != other_agent:
                trust_summary[other_agent] = self.get_trust(agent_name, other_agent)
        return trust_summary
    
    def get_trust_matrix(self) -> Dict[str, Dict[str, float]]:
        """Get the full trust matrix.
        
        Returns:
            The complete trust matrix
        """
        return self.trust_matrix
    
    def get_mood_from_trust(self, agent_name: str) -> str:
        """Determine agent mood based on trust trends.
        
        Args:
            agent_name: The agent to determine mood for
            
        Returns:
            Mood state as a string
        """
        trust_values = [self.get_trust(agent_name, other) 
                       for other in self.agent_names if other != agent_name]
        
        if not trust_values:
            return "neutral"
            
        avg_trust = sum(trust_values) / len(trust_values)
        
        if avg_trust > 0.7:
            return "collaborative"
        elif avg_trust > 0.5:
            return "supportive"
        elif avg_trust > 0.3:
            return "contemplative"
        else:
            return "defensive" 