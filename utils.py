"""
Utility functions for SimuChat application.
Includes formatting, display, and other helper functions.
"""

import random
import time
import os
from agents import EMOTIONS, AGENTS


def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def get_random_emotion():
    """Return a random emotion from the defined list."""
    return random.choice(EMOTIONS)


def format_agent_message(agent_name, message, emotion=None):
    """Format an agent's message for display with name, emotion, and message."""
    if emotion is None:
        emotion = get_random_emotion()
    
    agent = AGENTS[agent_name]
    formatted_message = f"{agent['emoji']} {agent_name} ({emotion}): {message}"
    return formatted_message


def display_message(message, delay=0.5):
    """Display a message with a delay for better readability."""
    print(message)
    time.sleep(delay)


def display_chat_history(messages):
    """Display the full chat history with formatted messages."""
    clear_screen()
    print("\n===== SimuChat =====")
    print("A WhatsApp-style AI Group Chat\n")
    
    for msg in messages:
        if msg["role"] == "user":
            print(f"👤 You: {msg['content']}")
        elif msg["role"] == "assistant" and "metadata" in msg:
            metadata = msg["metadata"]
            formatted = format_agent_message(
                metadata["agent_name"], 
                msg["content"], 
                metadata["emotion"]
            )
            print(formatted)
    print("\n" + "=" * 20 + "\n")


def prepare_messages_for_api(system_prompt, message_history):
    """Prepare messages in the format expected by LLaMA API."""
    formatted_messages = [
        {"role": "system", "content": system_prompt}
    ]
    
    # Add the conversation history
    for msg in message_history:
        # Skip metadata when sending to API
        api_msg = {"role": msg["role"], "content": msg["content"]}
        formatted_messages.append(api_msg)
    
    return formatted_messages 