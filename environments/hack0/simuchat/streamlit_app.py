"""
Streamlit UI for SimuChat application.
"""

import streamlit as st
import time
import random
import os
import json
from typing import Dict, List, Any

import env
from llama_api import get_agent_response
from memory import MemoryManager
from trust import TrustEngine
from utils import (
    get_random_emotion,
    detect_insight,
    get_insight_message
)
from logger import Logger


# Initialize session state for persistent data
def init_session_state():
    """Initialize Streamlit session state variables."""
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        st.session_state.message_history = []
        st.session_state.memory_manager = MemoryManager()
        st.session_state.trust_engine = TrustEngine()
        st.session_state.logger = Logger()
        st.session_state.turn = 0
        st.session_state.agent_index = 0
        st.session_state.is_generating = False
        st.session_state.agent_names = env.get_all_agent_names()


def get_agent_response_for_ui(agent_name):
    """Get a response from an agent for the UI.
    
    Args:
        agent_name: The name of the agent
        
    Returns:
        Agent response with metadata
    """
    agent_config = env.get_agent_config(agent_name)
    
    # Get the agent's memory context
    memory_context = st.session_state.memory_manager.get_memory_context(agent_name)
    
    # Get temperature for this agent
    temperature = agent_config.get("temperature", 0.6)
    temperature_multiplier = env.get_api_setting("temperature_multiplier", 1.0)
    adjusted_temperature = temperature * temperature_multiplier
    
    with st.spinner(f"Waiting for {agent_name}'s response..."):
        # Get response from the agent
        response = get_agent_response(
            agent_name=agent_name,
            agent_system_prompt=agent_config["system_prompt"],
            message_history=st.session_state.message_history,
            memory_context=memory_context,
            temperature=adjusted_temperature
        )
    
    # Determine emotion and mood
    emotion = get_random_emotion()
    mood = st.session_state.trust_engine.get_mood_from_trust(agent_name)
    
    # Check for insights
    has_insight = False
    if len(st.session_state.message_history) > 1:
        has_insight = detect_insight(agent_name, response, st.session_state.message_history)
    
    # Create message with metadata
    message = {
        "role": "assistant",
        "content": response,
        "metadata": {
            "agent_name": agent_name,
            "emotion": emotion,
            "mood": mood,
            "turn": st.session_state.turn,
            "is_insight": has_insight
        }
    }
    
    # Add to message history
    st.session_state.message_history.append(message)
    
    # Update all agent memories with this new message
    st.session_state.memory_manager.add_message_to_all_memories(message)
    
    # Update trust between agents
    trust_changes = st.session_state.trust_engine.update_all_trust(st.session_state.message_history)
    
    # Log the message and trust changes
    st.session_state.logger.log_message(message, trust_changes)
    
    # If it was an insight, also log it as an insight
    if has_insight:
        insight_message = get_insight_message(agent_name, response)
        st.session_state.memory_manager.get_agent_memory(agent_name).add_insight({
            "content": insight_message
        })
        st.session_state.logger.log_insight(agent_name, insight_message, trust_changes)
    
    # Update turn and agent index
    st.session_state.turn += 1
    st.session_state.agent_index = (st.session_state.agent_index + 1) % len(st.session_state.agent_names)
    
    return message


def display_message_ui(message):
    """Display a message in the Streamlit UI.
    
    Args:
        message: The message to display
    """
    if message["role"] == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(message["content"])
    elif message["role"] == "assistant" and "metadata" in message:
        metadata = message["metadata"]
        agent_name = metadata["agent_name"]
        emotion = metadata.get("emotion", "neutral")
        mood = metadata.get("mood", "")
        
        # Get agent config for emoji
        agent_config = env.get_agent_config(agent_name)
        emoji = agent_config.get("emoji", "🤖")
        
        # Get emotion emoji
        emotion_emoji = env.get_emotion_emoji(emotion)
        
        # Format avatar with emotion
        avatar = f"{emoji}"
        
        with st.chat_message(agent_name.lower(), avatar=avatar):
            # Format header with emotion and mood
            header = f"**{agent_name}** ({emotion_emoji} {emotion}"
            if mood:
                mood_emoji = env.get_mood_emoji(mood)
                header += f", {mood_emoji} {mood}"
            header += ")"
            
            st.markdown(header)
            
            # Add insight indicator if applicable
            if metadata.get("is_insight", False):
                st.markdown("💡 **Insight!**")
            
            # Display the message content
            st.markdown(message["content"])


def display_trust_ui():
    """Display trust network in the Streamlit UI."""
    st.subheader("Trust Network")
    
    trust_matrix = st.session_state.trust_engine.get_trust_matrix()
    agent_names = st.session_state.agent_names
    
    # Create a table for the trust network
    for agent1 in agent_names:
        cols = st.columns(len(agent_names))
        
        for i, agent2 in enumerate(agent_names):
            if agent1 != agent2:
                trust_value = st.session_state.trust_engine.get_trust(agent1, agent2)
                cols[i].metric(
                    label=f"{agent1} → {agent2}",
                    value=f"{trust_value:.2f}"
                )
                # Show a progress bar for the trust level
                cols[i].progress(trust_value)


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="SimuChat",
        page_icon="💬",
        layout="wide"
    )
    
    st.title("SimuChat - AI Group Chat")
    st.markdown("A WhatsApp-style AI Group Chat with Memory & Trust")
    
    # Initialize session state
    init_session_state()
    
    # Sidebar with info and controls
    with st.sidebar:
        st.header("About SimuChat")
        st.markdown("""
        SimuChat simulates a group chat between AI agents with different personalities.
        Each agent has memory of past interactions and develops trust relationships.
        
        Watch for the 💡 icon when agents have insights!
        """)
        
        st.header("Agents")
        for agent_name in st.session_state.agent_names:
            agent_config = env.get_agent_config(agent_name)
            emoji = agent_config.get("emoji", "🤖")
            st.markdown(f"- **{emoji} {agent_name}**: {agent_config.get('mood', 'neutral')}")
        
        st.header("Controls")
        if st.button("Reset Chat"):
            st.session_state.message_history = []
            st.session_state.memory_manager = MemoryManager()
            st.session_state.trust_engine = TrustEngine()
            st.session_state.logger = Logger()
            st.session_state.turn = 0
            st.session_state.agent_index = 0
            st.rerun()
        
        if st.session_state.message_history:
            st.download_button(
                label="Download Chat Log (HTML)",
                data=open(env.get_html_log_path(), "rb").read(),
                file_name="simuchat_log.html",
                mime="text/html"
            )
    
    # Main chat interface
    chat_container = st.container()
    
    with chat_container:
        # Display all messages in the history
        for message in st.session_state.message_history:
            display_message_ui(message)
    
    # Display trust network
    if st.session_state.message_history:
        display_trust_ui()
    
    # User input
    user_input = st.chat_input("Type your message here")
    
    if user_input:
        # Add user message to history
        user_message = {"role": "user", "content": user_input}
        st.session_state.message_history.append(user_message)
        
        # Add user message to agent memories
        st.session_state.memory_manager.add_message_to_all_memories(user_message)
        
        # Log the user message
        st.session_state.logger.log_message(user_message)
        
        # Display the user message
        display_message_ui(user_message)
        
        # Set flag to trigger agent responses
        st.session_state.is_generating = True
        st.rerun()
    
    # Process agent responses
    if st.session_state.is_generating and st.session_state.message_history:
        # Get the next agent to respond
        agent_name = st.session_state.agent_names[st.session_state.agent_index]
        
        # Get and display the agent's response
        agent_message = get_agent_response_for_ui(agent_name)
        display_message_ui(agent_message)
        
        # Continue generating responses from all agents
        if st.session_state.agent_index > 0:
            time.sleep(0.5)  # Brief pause between responses
            st.rerun()
        else:
            # All agents have responded, wait for user input
            st.session_state.is_generating = False


if __name__ == "__main__":
    main() 