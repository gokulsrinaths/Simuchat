"""
Utility functions for SimuChat application.
Includes formatting, display, and helper functions.
"""

import random
import time
import os
import re
from typing import Dict, List, Any, Optional
import env


def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def get_random_emotion() -> str:
    """Get a random emotion from the predefined list.
    
    Returns:
        Random emotion as a string
    """
    config = env.get_config()
    emotions = list(config["emotion_map"].keys())
    return random.choice(emotions)


def format_agent_message(agent_name: str, message: str, emotion: Optional[str] = None, 
                        mood: Optional[str] = None) -> str:
    """Format an agent's message for display.
    
    Args:
        agent_name: The agent's name
        message: The content of the message
        emotion: The agent's emotion (optional)
        mood: The agent's mood (optional)
        
    Returns:
        Formatted message string for display
    """
    agent_config = env.get_agent_config(agent_name)
    
    if not emotion:
        emotion = get_random_emotion()
    
    emoji = agent_config.get("emoji", "🤖")
    emotion_emoji = env.get_emotion_emoji(emotion)
    
    # Format with both emotion and mood if available
    if mood:
        mood_emoji = env.get_mood_emoji(mood)
        return f"{emotion_emoji} {emoji} {agent_name} ({emotion}, {mood_emoji} {mood}): {message}"
    else:
        return f"{emotion_emoji} {emoji} {agent_name} ({emotion}): {message}"


def display_message(message: str, delay: float = 0.5):
    """Display a message with a delay for better readability.
    
    Args:
        message: The message to display
        delay: Seconds to delay after displaying
    """
    print(message)
    time.sleep(delay)


def display_chat_history(messages: List[Dict[str, Any]], trust_engine=None):
    """Display the full chat history with formatted messages.
    
    Args:
        messages: List of messages to display
        trust_engine: Optional trust engine for displaying trust info
    """
    clear_screen()
    print("\n===== SimuChat =====")
    print("A WhatsApp-style AI Group Chat with Memory & Trust\n")
    
    for msg in messages:
        if msg["role"] == "user":
            print(f"👤 You: {msg['content']}")
        elif msg["role"] == "assistant" and "metadata" in msg:
            metadata = msg["metadata"]
            
            # Handle insights
            if metadata.get("is_insight", False):
                formatted = format_agent_message(
                    metadata["agent_name"], 
                    f"💡 {msg['content']}", 
                    metadata.get("emotion"),
                    metadata.get("mood")
                )
            else:
                formatted = format_agent_message(
                    metadata["agent_name"], 
                    msg["content"], 
                    metadata.get("emotion"),
                    metadata.get("mood")
                )
            print(formatted)
    
    # Display trust information if available
    if trust_engine:
        print("\n----- Trust Network -----")
        trust_matrix = trust_engine.get_trust_matrix()
        agent_names = env.get_all_agent_names()
        
        for agent1 in agent_names:
            trust_info = []
            for agent2 in agent_names:
                if agent1 != agent2:
                    trust_value = trust_engine.get_trust(agent1, agent2)
                    # Create a simple visual bar for trust level
                    bar_length = int(trust_value * 10)
                    bar = "█" * bar_length + "▒" * (10 - bar_length)
                    trust_info.append(f"{agent2}: {bar} ({trust_value:.2f})")
            
            print(f"{agent1} trusts: {', '.join(trust_info)}")
    
    print("\n" + "=" * 20 + "\n")


def detect_insight(agent_name: str, current_message: str, previous_messages: List[Dict[str, Any]]) -> bool:
    """Detect if an agent has had an insight moment.
    
    Args:
        agent_name: The agent's name
        current_message: The agent's current message
        previous_messages: List of previous messages
        
    Returns:
        True if an insight is detected, False otherwise
    """
    # Find the most recent message from this agent
    previous_msg = None
    for msg in reversed(previous_messages):
        if (msg.get("role") == "assistant" and 
            msg.get("metadata", {}).get("agent_name") == agent_name):
            previous_msg = msg
            break
    
    if not previous_msg:
        return False
    
    # Look for insight markers in the text
    insight_markers = [
        r"I see now",
        r"I understand",
        r"I realize",
        r"changed my mind",
        r"good point",
        r"I hadn't considered",
        r"you're right",
        r"that makes sense",
        r"I agree with",
        r"I've changed my perspective"
    ]
    
    # Check if the agent was previously disagreeing but now agrees
    previous_content = previous_msg["content"].lower()
    current_content = current_message.lower()
    
    # Check if previous message contained disagreement markers
    disagreement_markers = ["disagree", "no", "wrong", "incorrect", "not true", "false"]
    was_disagreeing = any(marker in previous_content for marker in disagreement_markers)
    
    # Check if current message contains agreement markers
    agreement_markers = ["agree", "yes", "right", "correct", "true", "good point"]
    is_agreeing = any(marker in current_content for marker in agreement_markers)
    
    # Explicit insight check
    has_insight_marker = any(re.search(marker, current_content, re.IGNORECASE) 
                            for marker in insight_markers)
    
    # Consider it an insight if:
    # 1. There's an explicit insight marker, or
    # 2. The agent was disagreeing but is now agreeing
    return has_insight_marker or (was_disagreeing and is_agreeing)


def get_insight_message(agent_name: str, message: str) -> str:
    """Format a message to highlight the insight.
    
    Args:
        agent_name: The agent's name
        message: The original message
        
    Returns:
        A message highlighting the insight
    """
    # This could use more sophistication to extract just the insight portion
    return message 