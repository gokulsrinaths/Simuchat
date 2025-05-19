"""
SimuChat: A WhatsApp-style fake group chat between AI agents
using Meta's LLaMA API.

This is the main application file that initializes and runs the chat simulation.
"""

import time
import random
from typing import Dict, List, Any, Optional

import env
from llama_api import get_agent_response
from memory import MemoryManager
from trust import TrustEngine
from rewards import RewardSystem
from utils import (
    clear_screen,
    get_random_emotion,
    format_agent_message,
    display_message,
    display_chat_history,
    detect_insight,
    get_insight_message
)
from logger import Logger


class SimuChat:
    """Main SimuChat application class."""
    
    def __init__(self):
        """Initialize the SimuChat application."""
        self.config = env.get_config()
        self.agent_names = env.get_all_agent_names()
        self.message_history = []
        self.memory_manager = MemoryManager()
        self.trust_engine = TrustEngine()
        self.logger = Logger()
        self.reward_system = RewardSystem()
        self.auto_mode = False
        self.max_auto_rounds = 50
        self.stop_loop = False
    
    def print_welcome(self):
        """Display welcome message and instructions."""
        clear_screen()
        print("\n===== Welcome to SimuChat =====")
        print("A WhatsApp-style AI Group Chat Simulation with Memory, Trust & Rewards\n")
        print("This application simulates a group chat between AI agents:")
        
        for agent_name in self.agent_names:
            agent_config = env.get_agent_config(agent_name)
            emoji = agent_config.get("emoji", "🤖")
            print(f"- {emoji} {agent_name}: {agent_config.get('mood', 'neutral')}")
        
        print("\nYou provide a starting topic, and they'll discuss it.")
        print("Each agent has memory and will develop trust relationships.")
        print("Agents earn points for increasing trust and having insights.")
        print("Look for the 💡 icon when agents have an insight!\n")
        print("When agents are rude to each other (💢), their trust decreases significantly.")
        print("Different levels of rudeness cause different trust penalties:")
        print("- Mild rudeness: -5% to -10% trust")
        print("- Moderate rudeness: -10% to -20% trust")
        print("- Severe rudeness: -20% to -30% trust")
        print("Direct rudeness (naming another agent) causes 50% more trust damage.\n")
        print("Enter 'quit' at any time to exit.")
        print("Enter 'auto' to start automatic conversation mode.")
        print("Enter 'stop' to stop automatic conversation mode.")
        print("=" * 50 + "\n")
    
    def get_user_topic(self):
        """Get the initial topic from the user."""
        while True:
            topic = input("Enter a topic to discuss (or 'quit' to exit): ").strip()
            if topic.lower() == 'quit':
                return None
            if topic:
                return topic
            print("Please enter a valid topic.")
    
    def handle_agent_response(self, agent_name: str, turn: int):
        """Get and process a response from an agent.
        
        Args:
            agent_name: The name of the agent
            turn: The current turn number
            
        Returns:
            True if the agent responded successfully, False otherwise
        """
        agent_config = env.get_agent_config(agent_name)
        if not agent_config:
            print(f"Error: Agent configuration not found for {agent_name}")
            return False
        
        print(f"Waiting for {agent_name}'s response...")
        
        # Get the agent's memory context
        memory_context = self.memory_manager.get_memory_context(agent_name)
        
        # Get reward context
        reward_context = self.reward_system.get_reward_context(agent_name)
        
        # Combine contexts
        full_context = memory_context
        if reward_context:
            full_context += "\n\n" + reward_context
        
        # Get temperature for this agent
        temperature = agent_config.get("temperature", 0.6)
        temperature_multiplier = env.get_api_setting("temperature_multiplier", 1.0)
        adjusted_temperature = temperature * temperature_multiplier
        
        # Get response from the agent
        response = get_agent_response(
            agent_name=agent_name,
            agent_system_prompt=agent_config["system_prompt"],
            message_history=self.message_history,
            memory_context=full_context,
            temperature=adjusted_temperature
        )
        
        # Determine emotion and mood
        emotion = get_random_emotion()
        mood = self.trust_engine.get_mood_from_trust(agent_name)
        
        # Check for insights
        has_insight = False
        if len(self.message_history) > 1:
            has_insight = detect_insight(agent_name, response, self.message_history)
        
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
        self.message_history.append(message)
        
        # Update all agent memories with this new message
        self.memory_manager.add_message_to_all_memories(message)
        
        # Update trust between agents
        trust_changes = self.trust_engine.update_all_trust(self.message_history)
        
        # Process rewards for this message
        rewards_info = self.reward_system.process_message_rewards(
            agent_name, 
            has_insight,
            trust_changes
        )
        message["metadata"]["rewards"] = rewards_info
        
        # Log the message, trust changes, and rewards
        self.logger.log_message(message, trust_changes, rewards_info)
        
        # If it was an insight, also log it as an insight
        if has_insight:
            insight_message = get_insight_message(agent_name, response)
            self.memory_manager.get_agent_memory(agent_name).add_insight({
                "content": insight_message
            })
            self.logger.log_insight(agent_name, insight_message, trust_changes)
        
        # Display any earned rewards
        if rewards_info["rewards_earned"] > 0:
            reward_text = f"🏆 {agent_name} earned {rewards_info['rewards_earned']} point(s): {', '.join(rewards_info['reasons'])}"
            print(reward_text)
        
        return True
    
    def display_rewards_summary(self):
        """Display a summary of agent rewards."""
        print("\n===== Agent Rewards =====")
        rewards = self.reward_system.get_all_rewards()
        
        # Sort by rewards (highest first)
        sorted_agents = sorted(rewards.items(), key=lambda x: x[1], reverse=True)
        
        for agent_name, points in sorted_agents:
            agent_config = env.get_agent_config(agent_name)
            emoji = agent_config.get("emoji", "🤖")
            print(f"{emoji} {agent_name}: {points} points")
        
        print("=" * 25)
    
    def run_auto_conversation(self, max_rounds=50):
        """Run an automatic conversation without user intervention.
        
        Args:
            max_rounds: Maximum number of rounds to run
        """
        self.auto_mode = True
        self.stop_loop = False
        round_count = 0
        
        try:
            print("\nStarting automatic conversation mode...")
            print("Type 'stop' and press Enter to end auto mode.\n")
            
            while round_count < max_rounds and not self.stop_loop:
                print(f"\n----- Round {round_count + 1} -----")
                
                # Run one full round (each agent responds once)
                for agent_idx, agent_name in enumerate(self.agent_names):
                    if self.stop_loop:
                        break
                    
                    success = self.handle_agent_response(agent_name, round_count)
                    if not success:
                        continue
                    
                    # Display updated chat and rewards
                    display_chat_history(self.message_history, self.trust_engine)
                    self.display_rewards_summary()
                    
                    # Brief pause between agent responses
                    time.sleep(1)
                
                # Check if user wants to stop auto mode after each round
                round_count += 1
                
                # Non-blocking user input check
                import msvcrt
                if msvcrt.kbhit():
                    user_input = input("\nEnter 'stop' to end auto mode (or press Enter to continue): ").strip().lower()
                    if user_input == 'stop':
                        self.stop_loop = True
                        print("Stopping automatic conversation mode...")
                    
                # Brief pause between rounds
                time.sleep(2)
            
            if round_count >= max_rounds:
                print(f"\nReached maximum of {max_rounds} rounds in auto mode.")
            
            self.auto_mode = False
            print("\nReturning to manual mode. Press Enter to continue or type a message.")
            
        except KeyboardInterrupt:
            print("\nAuto conversation mode interrupted.")
            self.auto_mode = False
        except Exception as e:
            print(f"\nError in auto conversation: {str(e)}")
            self.auto_mode = False
    
    def run_chat_simulation(self, topic):
        """Run the main chat simulation with the given topic.
        
        Args:
            topic: The initial topic to discuss
        """
        # Initialize message history with the user's topic
        user_message = {"role": "user", "content": topic}
        self.message_history = [user_message]
        
        # Add the user message to agent memories
        self.memory_manager.add_message_to_all_memories(user_message)
        
        # Log the user message
        self.logger.log_message(user_message)
        
        # Reset rewards
        self.reward_system.reset_rewards()
        
        # Display the initial chat state
        display_chat_history(self.message_history, self.trust_engine)
        
        try:
            turn = 0
            
            # Keep the conversation going until user interrupts
            while True:
                # Cycle through all agents
                for agent_idx, agent_name in enumerate(self.agent_names):
                    success = self.handle_agent_response(agent_name, turn)
                    if not success:
                        continue
                    
                    # Display updated chat with trust information
                    display_chat_history(self.message_history, self.trust_engine)
                    self.display_rewards_summary()
                    
                    # Move to the next agent
                    turn += 1
                    
                    # Brief pause between agent responses
                    time.sleep(1)
                
                # After all agents have responded, ask if user wants to continue
                choice = input("\nPress Enter to continue, type a message, 'auto' for auto mode, or 'quit' to exit: ").strip()
                if choice.lower() == 'quit':
                    break
                elif choice.lower() == 'auto':
                    self.run_auto_conversation(self.max_auto_rounds)
                elif choice.lower() == 'stop':
                    print("Not in auto mode. Type 'auto' to start auto mode.")
                elif choice.strip():
                    # Add user's new message to the history
                    user_message = {"role": "user", "content": choice}
                    self.message_history.append(user_message)
                    
                    # Add the user message to agent memories
                    self.memory_manager.add_message_to_all_memories(user_message)
                    
                    # Log the user message
                    self.logger.log_message(user_message)
                    
                    display_chat_history(self.message_history, self.trust_engine)
        
        except KeyboardInterrupt:
            print("\nChat simulation interrupted.")
        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")
        
        print("\nChat simulation ended.")
        print(f"Conversation log saved to: {env.get_html_log_path()}")
    
    def run(self):
        """Run the SimuChat application."""
        try:
            self.print_welcome()
            
            # Get the initial topic from the user
            topic = self.get_user_topic()
            if topic:
                self.run_chat_simulation(topic)
            
        except KeyboardInterrupt:
            print("\nApplication terminated by user.")
        except Exception as e:
            import traceback
            print(f"\nAn unexpected error occurred: {str(e)}")
            traceback.print_exc()
        
        print("\nThank you for using SimuChat!")


if __name__ == "__main__":
    app = SimuChat()
    app.run() 