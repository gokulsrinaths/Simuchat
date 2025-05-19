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
from rewards import RewardSystem
from utils import (
    get_random_emotion,
    detect_insight,
    get_insight_message,
    detect_rudeness
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
        st.session_state.reward_system = RewardSystem()
        st.session_state.logger = Logger()
        st.session_state.turn = 0
        st.session_state.agent_index = 0
        st.session_state.is_generating = False
        st.session_state.auto_conversation = False
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
    
    # Get reward context
    reward_context = st.session_state.reward_system.get_reward_context(agent_name)
    
    # Combine contexts
    full_context = memory_context
    if reward_context:
        full_context += "\n\n" + reward_context
    
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
            memory_context=full_context,
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
    
    # Process rewards for this message
    rewards_info = st.session_state.reward_system.process_message_rewards(
        agent_name, 
        has_insight,
        trust_changes
    )
    message["metadata"]["rewards"] = rewards_info
    
    # Log the message, trust changes, and rewards
    st.session_state.logger.log_message(message, trust_changes, rewards_info)
    
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
            
            # Check for rudeness
            is_rude, rudeness_severity = detect_rudeness(agent_name, message["content"])
            if is_rude:
                header += f" 💢 **{rudeness_severity.capitalize()} Rudeness**"
            
            st.markdown(header)
            
            # Add insight indicator if applicable
            if metadata.get("is_insight", False):
                st.markdown("💡 **Insight!**")
            
            # Display the message content
            st.markdown(message["content"])
            
            # Display reward information if available
            if "rewards" in metadata and metadata["rewards"]["rewards_earned"] > 0:
                reward_info = metadata["rewards"]
                st.markdown(f"🏆 **+{reward_info['rewards_earned']} points**: {', '.join(reward_info['reasons'])}")


def display_rewards_ui():
    """Display rewards information in the Streamlit UI."""
    st.subheader("🏆 Agent Rewards")
    
    # Get current rewards
    rewards = st.session_state.reward_system.get_all_rewards()
    
    # Sort by rewards (highest first)
    sorted_agents = sorted(rewards.items(), key=lambda x: x[1], reverse=True)
    
    # Display in columns
    cols = st.columns(len(sorted_agents))
    
    for i, (agent_name, points) in enumerate(sorted_agents):
        agent_config = env.get_agent_config(agent_name)
        emoji = agent_config.get("emoji", "🤖")
        
        cols[i].metric(
            label=f"{emoji} {agent_name}",
            value=f"{points} pts"
        )
        
        # Add a small trophy for the leader
        if i == 0 and points > 0:
            cols[i].markdown("👑 Leader")


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="SimuChat",
        page_icon="💬",
        layout="wide"
    )
    
    st.title("SimuChat - AI Group Chat")
    st.markdown("A WhatsApp-style AI Group Chat with Memory, Trust & Rewards")
    
    # Initialize session state
    init_session_state()
    
    # Create a 3-column layout: left sidebar, center chat, right trust network
    # Sidebar with info and controls
    with st.sidebar:
        st.header("About SimuChat")
        st.markdown("""
        SimuChat simulates a group chat between AI agents with different personalities.
        Each agent has memory of past interactions and develops trust relationships.
        
        Agents earn points for increasing trust and having insights!
        Watch for the 💡 icon when agents have insights!
        
        When agents are rude to each other (💢), their trust decreases:
        - Mild rudeness: -5% to -10% trust
        - Moderate rudeness: -10% to -20% trust
        - Severe rudeness: -20% to -30% trust
        
        Direct rudeness (naming another agent) causes 50% more trust damage!
        """)
        
        st.header("Agents")
        for agent_name in st.session_state.agent_names:
            agent_config = env.get_agent_config(agent_name)
            emoji = agent_config.get("emoji", "🤖")
            st.markdown(f"- **{emoji} {agent_name}**: {agent_config.get('mood', 'neutral')}")
        
        st.header("Controls")
        col1, col2 = st.columns(2)
        
        if col1.button("Reset Chat"):
            st.session_state.message_history = []
            st.session_state.memory_manager = MemoryManager()
            st.session_state.trust_engine = TrustEngine()
            st.session_state.reward_system = RewardSystem()
            st.session_state.logger = Logger()
            st.session_state.turn = 0
            st.session_state.agent_index = 0
            st.session_state.auto_conversation = False
            st.rerun()
        
        # Auto conversation toggle
        auto_mode = col2.toggle("Auto Mode", value=st.session_state.auto_conversation)
        if auto_mode != st.session_state.auto_conversation:
            st.session_state.auto_conversation = auto_mode
            if auto_mode:
                st.info("Auto conversation mode enabled. Agents will chat continuously.")
            else:
                st.info("Auto conversation mode disabled.")
        
        # Number of automatic rounds
        if st.session_state.auto_conversation:
            max_auto_rounds = st.slider("Max Rounds", min_value=1, max_value=50, value=10)
        
        if st.session_state.message_history:
            st.download_button(
                label="Download Chat Log (HTML)",
                data=open(env.get_html_log_path(), "rb").read(),
                file_name="simuchat_log.html",
                mime="text/html"
            )
    
    # Main content area with chat and trust network
    chat_col, trust_col = st.columns([3, 1])  # 3:1 ratio to make chat wider
    
    # Main chat interface in the center column
    with chat_col:
        st.subheader("Chat")
        chat_container = st.container()
        
        with chat_container:
            # Display all messages in the history
            for message in st.session_state.message_history:
                display_message_ui(message)
        
        # User input at the bottom of the chat column
        user_input = st.chat_input("Type your message here")
    
    # Trust and rewards in the right column
    with trust_col:
        # Display rewards at the top of the right column
        if st.session_state.message_history:
            display_rewards_ui()
            
            st.markdown("---")  # Separator
            
            st.subheader("Trust Network")
            trust_matrix = st.session_state.trust_engine.get_trust_matrix()
            agent_names = st.session_state.agent_names
            
            # Display trust network vertically (one agent per row)
            for agent1 in agent_names:
                st.write(f"**{agent1}'s Trust:**")
                
                for agent2 in agent_names:
                    if agent1 != agent2:
                        trust_value = st.session_state.trust_engine.get_trust(agent1, agent2)
                        
                        # Show a progress bar for the trust level
                        st.progress(trust_value, text=f"{agent2}: {trust_value:.2f}")
                
                st.markdown("---")  # Add separator between agents
    
    # Process user input
    if user_input:
        # Add user message to history
        user_message = {"role": "user", "content": user_input}
        st.session_state.message_history.append(user_message)
        
        # Add user message to agent memories
        st.session_state.memory_manager.add_message_to_all_memories(user_message)
        
        # Log the user message
        st.session_state.logger.log_message(user_message)
        
        # Set flag to trigger agent responses
        st.session_state.is_generating = True
        st.rerun()
    
    # Process agent responses (normal mode)
    if st.session_state.is_generating and not st.session_state.auto_conversation and st.session_state.message_history:
        # Get the next agent to respond
        agent_name = st.session_state.agent_names[st.session_state.agent_index]
        
        # Get the agent's response
        agent_message = get_agent_response_for_ui(agent_name)
        
        # Continue generating responses from all agents
        if st.session_state.agent_index > 0:
            time.sleep(0.5)  # Brief pause between responses
            st.rerun()
        else:
            # All agents have responded, wait for user input
            st.session_state.is_generating = False
    
    # Process agent responses (auto conversation mode)
    elif st.session_state.auto_conversation and st.session_state.message_history:
        # Check if we need to start generating or continue
        if not st.session_state.is_generating:
            st.session_state.is_generating = True
            # Reset agent index if needed
            if st.session_state.agent_index == 0:
                st.info("Starting automatic conversation...")
        
        # Get the next agent to respond
        agent_name = st.session_state.agent_names[st.session_state.agent_index]
        
        # Get the agent's response
        agent_message = get_agent_response_for_ui(agent_name)
        
        # Add a pause to slow down the conversation
        time.sleep(1.5)  # Longer pause for auto mode
        
        # Always rerun in auto mode to continue the conversation
        st.rerun()


if __name__ == "__main__":
    main() 