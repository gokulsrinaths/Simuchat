"""
Environment settings for SimuChat application.
Contains API details and model configuration.
"""

import os
import json
from pathlib import Path
import time
import random
from typing import Dict, List, Any, Optional

# Define paths
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
CONFIG_PATH = BASE_DIR / "agents_config.json"

# Ensure output directory exists
OUTPUT_DIR.mkdir(exist_ok=True)

# Meta LLaMA API configuration
API_KEY = "LLM|1050039463644017|ZZzxjun1klZ76kW0xu5Zg4BW5-o"  # API key
API_BASE_URL = "https://api.llama.com/v1/chat/completions"  # API endpoint
MODEL_NAME = "Llama-4-Maverick-17B-128E-Instruct-FP8"  # Using Maverick model

# Load configuration from JSON
def load_config():
    """Load the configuration from agents_config.json."""
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading configuration: {e}")
        raise

# Cached config
_config = None

def get_config():
    """Get the configuration, loading it if necessary."""
    global _config
    if _config is None:
        _config = load_config()
    return _config

def get_agent_config(agent_name):
    """Get configuration for a specific agent."""
    config = get_config()
    for agent in config["agents"]:
        if agent["name"] == agent_name:
            return agent
    return None

def get_emotion_emoji(emotion):
    """Get the emoji for a given emotion."""
    config = get_config()
    return config["emotion_map"].get(emotion, "😐")

def get_mood_emoji(mood):
    """Get the emoji for a given mood."""
    config = get_config()
    return config["mood_map"].get(mood, "🔄")

def get_all_agent_names():
    """Get a list of all agent names."""
    config = get_config()
    return [agent["name"] for agent in config["agents"]]

def get_api_setting(key, default=None):
    """Get a specific API setting."""
    config = get_config()
    return config["api_settings"].get(key, default)

# File paths for logging
def get_jsonl_log_path():
    """Get the path for the JSON Lines log file."""
    return OUTPUT_DIR / "chatlog.jsonl"

def get_html_log_path():
    """Get the path for the HTML log file."""
    return OUTPUT_DIR / "chatlog.html"

def get_metrics_path():
    """Get the path for the metrics JSON Lines file."""
    return OUTPUT_DIR / "metrics.jsonl"

def run(prompt: str, num_turns: int = 3, headless: bool = True) -> Dict[str, Any]:
    """
    Run a SimuChat simulation in headless mode.
    
    Args:
        prompt: The starting prompt/topic for conversation
        num_turns: Number of conversation turns to run
        headless: Whether to run in headless mode (no UI)
        
    Returns:
        Dictionary with metrics and information about the run
    """
    # Import here to avoid circular imports
    from memory import MemoryManager
    from trust import TrustEngine
    from utils import get_random_emotion, detect_insight, get_insight_message
    from logger import Logger
    from llama_api import get_agent_response
    
    # Initialize components
    memory_manager = MemoryManager()
    trust_engine = TrustEngine()
    logger = Logger()
    
    # Start with the user's prompt
    user_message = {"role": "user", "content": prompt}
    message_history = [user_message]
    
    # Add the user message to agent memories
    memory_manager.add_message_to_all_memories(user_message)
    
    # Log the user message
    logger.log_message(user_message)
    
    # Track metrics
    metrics = {
        "num_messages": 1,  # Start with the user prompt
        "agent_messages": {},
        "insights": {},
        "emotions": {},
        "moods": {},
        "trust_scores": {},
        "start_time": time.time(),
        "prompt": prompt
    }
    
    for agent_name in get_all_agent_names():
        metrics["agent_messages"][agent_name] = 0
        metrics["insights"][agent_name] = 0
        metrics["emotions"][agent_name] = {}
        metrics["moods"][agent_name] = {}
    
    # Run the simulation for the specified number of turns
    try:
        turn = 0
        agent_names = get_all_agent_names()
        
        while turn < num_turns:
            # Cycle through all agents
            for agent_idx, agent_name in enumerate(agent_names):
                # Get agent configuration
                agent_config = get_agent_config(agent_name)
                
                # Get memory context
                memory_context = memory_manager.get_memory_context(agent_name)
                
                # Get temperature for this agent
                temperature = agent_config.get("temperature", 0.6)
                temperature_multiplier = get_api_setting("temperature_multiplier", 1.0)
                adjusted_temperature = temperature * temperature_multiplier
                
                # Get response from the agent
                response = get_agent_response(
                    agent_name=agent_name,
                    agent_system_prompt=agent_config["system_prompt"],
                    message_history=message_history,
                    memory_context=memory_context,
                    temperature=adjusted_temperature
                )
                
                # Determine emotion and mood
                emotion = get_random_emotion()
                mood = trust_engine.get_mood_from_trust(agent_name)
                
                # Check for insights
                has_insight = False
                if len(message_history) > 1:
                    has_insight = detect_insight(agent_name, response, message_history)
                
                # Add the response to the message history with metadata
                message = {
                    "role": "assistant",
                    "content": response,
                    "metadata": {
                        "agent_name": agent_name,
                        "emotion": emotion,
                        "mood": mood,
                        "turn": turn,
                        "is_insight": has_insight
                    }
                }
                
                # Add to message history
                message_history.append(message)
                
                # Update all agent memories with this new message
                memory_manager.add_message_to_all_memories(message)
                
                # Update trust between agents
                trust_changes = trust_engine.update_all_trust(message_history)
                
                # Log the message and trust changes
                logger.log_message(message, trust_changes)
                
                # If it was an insight, also log it as an insight
                if has_insight:
                    insight_message = get_insight_message(agent_name, response)
                    memory_manager.get_agent_memory(agent_name).add_insight({
                        "content": insight_message
                    })
                    logger.log_insight(agent_name, insight_message, trust_changes)
                
                # Update metrics
                metrics["num_messages"] += 1
                metrics["agent_messages"][agent_name] += 1
                
                if has_insight:
                    metrics["insights"][agent_name] += 1
                
                if emotion in metrics["emotions"][agent_name]:
                    metrics["emotions"][agent_name][emotion] += 1
                else:
                    metrics["emotions"][agent_name][emotion] = 1
                
                if mood in metrics["moods"][agent_name]:
                    metrics["moods"][agent_name][mood] += 1
                else:
                    metrics["moods"][agent_name][mood] = 1
            
            # Update trust metrics after each complete round
            for agent1 in agent_names:
                if agent1 not in metrics["trust_scores"]:
                    metrics["trust_scores"][agent1] = {}
                
                for agent2 in agent_names:
                    if agent1 != agent2:
                        metrics["trust_scores"][agent1][agent2] = trust_engine.get_trust(agent1, agent2)
            
            # Move to the next turn
            turn += 1
            
            # Brief pause between turns (only if not headless)
            if not headless:
                time.sleep(1)
        
        # Add final metrics
        metrics["end_time"] = time.time()
        metrics["duration"] = metrics["end_time"] - metrics["start_time"]
        
        # Calculate average trust scores
        avg_trust = {}
        for agent1 in agent_names:
            trust_sum = sum(metrics["trust_scores"][agent1].values())
            avg_trust[agent1] = trust_sum / (len(agent_names) - 1) if len(agent_names) > 1 else 0
        
        metrics["avg_trust"] = avg_trust
        
        # Write metrics to file
        metrics_file = get_metrics_path()
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f)
            f.write("\n")  # Add newline for JSONL format
        
        return metrics
        
    except Exception as e:
        print(f"Error in simulation: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)} 