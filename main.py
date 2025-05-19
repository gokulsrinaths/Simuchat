"""
SimuChat: A WhatsApp-style fake group chat between 3 AI agents
using Meta's LLaMA API.

This is the main application file that initializes and runs the chat simulation.
"""

import time
from agents import AGENTS, AGENT_NAMES
from llama_api import get_agent_response
from utils import (
    clear_screen, 
    get_random_emotion, 
    format_agent_message, 
    display_message,
    display_chat_history
)


def print_welcome():
    """Display welcome message and instructions."""
    clear_screen()
    print("\n===== Welcome to SimuChat =====")
    print("A WhatsApp-style AI Group Chat Simulation\n")
    print("This application simulates a group chat between three AI agents:")
    print("- Alice: Kind and empathetic")
    print("- Bob: Logical and analytical")
    print("- Charlie: Bold and competitive\n")
    print("You provide a starting topic, and they'll discuss it.")
    print("Enter 'quit' at any time to exit.")
    print("=" * 35 + "\n")


def get_user_topic():
    """Get the initial topic from the user."""
    while True:
        topic = input("Enter a topic to discuss (or 'quit' to exit): ").strip()
        if topic.lower() == 'quit':
            return None
        if topic:
            return topic
        print("Please enter a valid topic.")


def run_chat_simulation(topic):
    """Run the main chat simulation with the given topic."""
    # Initialize message history with the user's topic
    message_history = [
        {"role": "user", "content": topic}
    ]
    
    # Display the initial chat state
    display_chat_history(message_history)
    
    try:
        turn = 0
        
        # Keep the conversation going until user interrupts
        while True:
            # Get the next agent to respond
            agent_name = AGENT_NAMES[turn % len(AGENT_NAMES)]
            agent = AGENTS[agent_name]
            
            print(f"Waiting for {agent_name}'s response...")
            
            # Get response from the agent
            response = get_agent_response(agent["system_prompt"], message_history)
            
            # Generate a random emotion for the agent
            emotion = get_random_emotion()
            
            # Add the response to the message history with metadata
            message_history.append({
                "role": "assistant",
                "content": response,
                "metadata": {
                    "agent_name": agent_name,
                    "emotion": emotion
                }
            })
            
            # Display updated chat
            display_chat_history(message_history)
            
            # Move to the next agent
            turn += 1
            
            # Brief pause between agent responses
            time.sleep(1)
            
            # Check if user wants to continue or add a new message
            if turn % len(AGENT_NAMES) == 0:
                choice = input("\nPress Enter to continue the conversation, type a new message, or 'quit' to exit: ")
                if choice.lower() == 'quit':
                    break
                elif choice.strip():
                    # Add user's new message to the history
                    message_history.append({"role": "user", "content": choice})
                    display_chat_history(message_history)
    
    except KeyboardInterrupt:
        print("\nChat simulation interrupted.")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
    
    print("\nChat simulation ended.")


def main():
    """Main application function."""
    try:
        print_welcome()
        
        # Get the initial topic from the user
        topic = get_user_topic()
        if topic:
            run_chat_simulation(topic)
        
    except KeyboardInterrupt:
        print("\nApplication terminated by user.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {str(e)}")
    
    print("\nThank you for using SimuChat!")


if __name__ == "__main__":
    main() 