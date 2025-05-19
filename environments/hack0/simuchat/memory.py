"""
Memory management for SimuChat agents.
Handles agent-specific memory and context.
"""

from typing import Dict, List, Any
from collections import deque
import env


class AgentMemory:
    """Class to manage an agent's memory of conversation."""
    
    def __init__(self, agent_name: str):
        """Initialize memory for an agent.
        
        Args:
            agent_name: The name of the agent
        """
        self.agent_name = agent_name
        agent_config = env.get_agent_config(agent_name)
        self.memory_limit = agent_config.get("memory_limit", 
                                          env.get_api_setting("default_memory_limit", 3))
        self.memory = deque(maxlen=self.memory_limit)
        self.insights = []
    
    def add_message(self, message: Dict[str, Any]):
        """Add a message to the agent's memory.
        
        Args:
            message: The message to add (must have 'role', 'agent_name', and 'content')
        """
        # Only track messages from other agents or the user
        if message.get("role") == "assistant" and message.get("metadata", {}).get("agent_name") == self.agent_name:
            return
            
        # Create a simplified memory entry
        if message.get("role") == "user":
            memory_entry = {
                "type": "user",
                "content": message["content"]
            }
        elif message.get("role") == "assistant":
            agent_name = message.get("metadata", {}).get("agent_name")
            if agent_name:
                memory_entry = {
                    "type": "agent",
                    "agent": agent_name,
                    "content": message["content"],
                    "emotion": message.get("metadata", {}).get("emotion", "neutral")
                }
            else:
                return  # Skip if no agent name (shouldn't happen)
        else:
            return  # Skip system messages
            
        self.memory.append(memory_entry)
    
    def add_insight(self, insight: Dict[str, Any]):
        """Add an insight to the agent's memory.
        
        Args:
            insight: The insight to add
        """
        self.insights.append(insight)
    
    def get_memory_context(self) -> str:
        """Generate a memory context string from the agent's memory.
        
        Returns:
            A formatted string representing the agent's memory context
        """
        if not self.memory:
            return ""
            
        context_parts = ["Recent conversation memories:"]
        
        for i, mem in enumerate(self.memory):
            if mem["type"] == "user":
                context_parts.append(f"{i+1}. User said: \"{mem['content']}\"")
            else:
                context_parts.append(
                    f"{i+1}. {mem['agent']} ({mem['emotion']}) said: \"{mem['content']}\""
                )
        
        # Add insights if any
        if self.insights:
            context_parts.append("\nYour recent insights:")
            for i, insight in enumerate(self.insights[-3:]):  # Only include last 3 insights
                context_parts.append(f"- {insight['content']}")
        
        return "\n".join(context_parts)
    
    def clear(self):
        """Clear the agent's memory."""
        self.memory.clear()
        self.insights.clear()


class MemoryManager:
    """Class to manage memories for all agents."""
    
    def __init__(self):
        """Initialize the memory manager."""
        self.memories = {}
        
    def get_agent_memory(self, agent_name: str) -> AgentMemory:
        """Get or create memory for an agent.
        
        Args:
            agent_name: The name of the agent
            
        Returns:
            The agent's memory object
        """
        if agent_name not in self.memories:
            self.memories[agent_name] = AgentMemory(agent_name)
        return self.memories[agent_name]
    
    def add_message_to_all_memories(self, message: Dict[str, Any]):
        """Add a message to all agent memories.
        
        Args:
            message: The message to add
        """
        for agent_name in env.get_all_agent_names():
            self.get_agent_memory(agent_name).add_message(message)
    
    def get_memory_context(self, agent_name: str) -> str:
        """Get the memory context for an agent.
        
        Args:
            agent_name: The name of the agent
            
        Returns:
            The agent's memory context
        """
        return self.get_agent_memory(agent_name).get_memory_context() 