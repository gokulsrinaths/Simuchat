"""
Environment settings for SimuChat application.
Contains API details and model configuration.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import time
import random
from typing import Dict, List, Any, Optional, Tuple
import wandb
from dataclasses import dataclass, field

# Add Atropos library to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from atroposlib.config import BaseEnvConfig, APIServerConfig

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

@dataclass
class SimuChatEnvConfig(BaseEnvConfig):
    """Environment configuration for SimuChat."""
    num_turns: int = 3  # Number of conversation turns to run
    data_path_to_save_groups: str = field(default_factory=lambda: str(OUTPUT_DIR / "simuchat_rollouts.jsonl"))
    prompt: str = "Let's discuss artificial intelligence and its impact on society."

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

def process(env_config: SimuChatEnvConfig, api_config: APIServerConfig) -> Dict[str, Any]:
    """
    Process command for Atropos framework - runs a simulation and generates HTML/JSONL output.
    
    Args:
        env_config: Environment configuration
        api_config: API server configuration
        
    Returns:
        Dictionary with metrics and results
    """
    # Override API configuration
    global API_KEY, API_BASE_URL, MODEL_NAME
    if api_config.api_key:
        API_KEY = api_config.api_key
    if api_config.base_url:
        API_BASE_URL = api_config.base_url
    if api_config.model_name:
        MODEL_NAME = api_config.model_name
    
    # Initialize WandB if enabled
    if env_config.use_wandb:
        wandb.init(
            project="simuchat",
            name=env_config.wandb_name or "simuchat-run",
            config={
                "num_turns": env_config.num_turns,
                "model": MODEL_NAME,
                "prompt": env_config.prompt
            }
        )
    
    # Run the simulation
    metrics = run(
        prompt=env_config.prompt,
        num_turns=env_config.num_turns,
        headless=True,
        save_path=env_config.data_path_to_save_groups
    )
    
    # Log metrics to WandB
    if env_config.use_wandb:
        # Log overall metrics
        wandb.log({
            "total_messages": metrics["num_messages"],
            "simulation_duration": metrics["duration"],
        })
        
        # Log agent-specific metrics
        for agent_name in get_all_agent_names():
            wandb.log({
                f"{agent_name}/messages": metrics["agent_messages"][agent_name],
                f"{agent_name}/insights": metrics["insights"][agent_name],
                f"{agent_name}/rewards": metrics["agent_rewards"][agent_name],
                f"{agent_name}/avg_trust": metrics["avg_trust"][agent_name],
            })
            
            # Log emotions
            for emotion, count in metrics["emotions"][agent_name].items():
                wandb.log({f"{agent_name}/emotion_{emotion}": count})
            
            # Log moods
            for mood, count in metrics["moods"][agent_name].items():
                wandb.log({f"{agent_name}/mood_{mood}": count})
        
        # Log trust relationships
        for agent1 in get_all_agent_names():
            for agent2 in get_all_agent_names():
                if agent1 != agent2:
                    trust_value = metrics["trust_scores"][agent1][agent2]
                    wandb.log({f"trust/{agent1}_to_{agent2}": trust_value})
        
        # Log the conversation as an HTML artifact
        html_path = str(get_html_log_path())
        if os.path.exists(html_path):
            wandb.log({"conversation": wandb.Html(open(html_path).read())})
    
    return metrics

def run(prompt: str, num_turns: int = 3, headless: bool = True, save_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Run a SimuChat simulation in headless mode.
    
    Args:
        prompt: The starting prompt/topic for conversation
        num_turns: Number of conversation turns to run
        headless: Whether to run in headless mode (no UI)
        save_path: Path to save the rollout data (for Atropos compatibility)
        
    Returns:
        Dictionary with metrics and information about the run
    """
    # Import here to avoid circular imports
    from memory import MemoryManager
    from trust import TrustEngine
    from utils import get_random_emotion, detect_insight, get_insight_message, detect_rudeness
    from logger import Logger
    from llama_api import get_agent_response
    from rewards import RewardSystem
    
    # Initialize components
    memory_manager = MemoryManager()
    trust_engine = TrustEngine()
    logger = Logger()
    reward_system = RewardSystem()
    
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
        "rudeness_detected": {},
        "start_time": time.time(),
        "prompt": prompt
    }
    
    for agent_name in get_all_agent_names():
        metrics["agent_messages"][agent_name] = 0
        metrics["insights"][agent_name] = 0
        metrics["emotions"][agent_name] = {}
        metrics["moods"][agent_name] = {}
        metrics["rudeness_detected"][agent_name] = {"mild": 0, "moderate": 0, "severe": 0}
    
    # Run the simulation for the specified number of turns
    try:
        turn = 0
        agent_names = get_all_agent_names()
        
        # Data for Atropos rollout format
        rollout_data = {
            "prompt": prompt,
            "turns": [],
            "metrics": {}
        }
        
        while turn < num_turns:
            turn_data = {
                "turn_number": turn,
                "messages": []
            }
            
            # Cycle through all agents
            for agent_idx, agent_name in enumerate(agent_names):
                # Get agent configuration
                agent_config = get_agent_config(agent_name)
                
                # Get memory context
                memory_context = memory_manager.get_memory_context(agent_name)
                
                # Get reward context
                reward_context = reward_system.get_reward_context(agent_name)
                
                # Combine contexts
                full_context = memory_context
                if reward_context:
                    full_context += "\n\n" + reward_context
                
                # Get temperature for this agent
                temperature = agent_config.get("temperature", 0.6)
                temperature_multiplier = get_api_setting("temperature_multiplier", 1.0)
                adjusted_temperature = temperature * temperature_multiplier
                
                # Get response from the agent
                response = get_agent_response(
                    agent_name=agent_name,
                    agent_system_prompt=agent_config["system_prompt"],
                    message_history=message_history,
                    memory_context=full_context,
                    temperature=adjusted_temperature
                )
                
                # Determine emotion and mood
                emotion = get_random_emotion()
                mood = trust_engine.get_mood_from_trust(agent_name)
                
                # Check for insights
                has_insight = False
                if len(message_history) > 1:
                    has_insight = detect_insight(agent_name, response, message_history)
                
                # Check for rudeness
                is_rude, rudeness_severity = detect_rudeness(agent_name, response)
                if is_rude and rudeness_severity in ["mild", "moderate", "severe"]:
                    metrics["rudeness_detected"][agent_name][rudeness_severity] += 1
                
                # Add the response to the message history with metadata
                message = {
                    "role": "assistant",
                    "content": response,
                    "metadata": {
                        "agent_name": agent_name,
                        "emotion": emotion,
                        "mood": mood,
                        "turn": turn,
                        "is_insight": has_insight,
                        "is_rude": is_rude,
                        "rudeness_severity": rudeness_severity if is_rude else "none"
                    }
                }
                
                # Add to message history
                message_history.append(message)
                
                # Add to turn data for rollout
                turn_data["messages"].append({
                    "agent": agent_name,
                    "content": response,
                    "emotion": emotion,
                    "mood": mood,
                    "is_insight": has_insight,
                    "is_rude": is_rude,
                    "rudeness_severity": rudeness_severity if is_rude else "none"
                })
                
                # Update all agent memories with this new message
                memory_manager.add_message_to_all_memories(message)
                
                # Update trust between agents
                trust_changes = trust_engine.update_all_trust(message_history)
                
                # Process rewards for this message
                rewards_info = reward_system.process_message_rewards(
                    agent_name, 
                    has_insight,
                    trust_changes
                )
                message["metadata"]["rewards"] = rewards_info
                
                # Log the message, trust changes, and rewards
                logger.log_message(message, trust_changes, rewards_info)
                
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
                
                # Add reward metrics
                if "agent_rewards" not in metrics:
                    metrics["agent_rewards"] = {}
                
                metrics["agent_rewards"][agent_name] = reward_system.get_agent_rewards(agent_name)
            
            # Add this turn to the rollout data
            rollout_data["turns"].append(turn_data)
            
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
        
        # Add final reward summary
        metrics["reward_summary"] = reward_system.get_reward_summary()
        
        # Add metrics to rollout data
        rollout_data["metrics"] = metrics
        
        # Write metrics to file
        metrics_file = get_metrics_path()
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f)
            f.write("\n")  # Add newline for JSONL format
        
        # Save rollout data if path provided (for Atropos compatibility)
        if save_path:
            save_path_obj = Path(save_path)
            with open(save_path_obj, "w", encoding="utf-8") as f:
                json.dump(rollout_data, f)
                f.write("\n")  # Add newline for JSONL format
            
            # Also generate HTML visualization of the rollouts
            html_output_path = save_path_obj.with_suffix(".html")
            generate_html_visualization(rollout_data, str(html_output_path))
        
        return metrics
        
    except Exception as e:
        print(f"Error in simulation: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

def generate_html_visualization(data: Dict[str, Any], output_path: str):
    """Generate an HTML visualization of the rollout data."""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>SimuChat Rollout Visualization</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 900px;
                margin: 0 auto;
                background-color: #fff;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
                padding: 20px;
            }}
            h1, h2, h3 {{
                color: #333;
            }}
            .prompt {{
                background-color: #e9f5ff;
                padding: 15px;
                border-radius: 5px;
                margin: 15px 0;
                border-left: 4px solid #2c86fc;
            }}
            .turn {{
                margin: 20px 0;
                border: 1px solid #eee;
                border-radius: 5px;
                padding: 10px;
            }}
            .message {{
                margin: 10px 0;
                padding: 10px;
                border-radius: 5px;
            }}
            .alice-message {{
                background-color: #f8e8ff;
                border-left: 3px solid #b266ff;
            }}
            .bob-message {{
                background-color: #e8fff8;
                border-left: 3px solid #66ffcc;
            }}
            .charlie-message {{
                background-color: #fff8e8;
                border-left: 3px solid #ffcc66;
            }}
            .insight {{
                background-color: #fffce8;
                border-left: 3px solid #ffeb3b;
            }}
            .mild-rudeness {{
                background-color: #fff0e0;
                border-left: 3px solid #ffc107;
            }}
            .moderate-rudeness {{
                background-color: #ffe0e0;
                border-left: 3px solid #ff9800;
            }}
            .severe-rudeness {{
                background-color: #ffe0e0;
                border-left: 3px solid #f44336;
            }}
            .metadata {{
                color: #666;
                font-size: 0.85em;
                margin-top: 5px;
            }}
            .metrics {{
                background-color: #f9f9f9;
                padding: 15px;
                border-radius: 5px;
                margin-top: 20px;
            }}
            .agent-metrics {{
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
                margin-top: 10px;
            }}
            .agent-card {{
                flex: 1;
                min-width: 200px;
                background-color: #fff;
                border-radius: 5px;
                padding: 10px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }}
            .badge {{
                display: inline-block;
                padding: 2px 6px;
                border-radius: 4px;
                font-size: 0.8em;
                margin-right: 5px;
            }}
            .insight-badge {{
                background-color: #fffce8;
                color: #b7a200;
            }}
            .rudeness-badge {{
                background-color: #ffe0e0;
                color: #c62828;
            }}
            .trust-table {{
                width: 100%;
                border-collapse: collapse;
                margin: 15px 0;
            }}
            .trust-table th, .trust-table td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: center;
            }}
            .trust-table th {{
                background-color: #f2f2f2;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>SimuChat Rollout Visualization</h1>
            
            <h2>Prompt</h2>
            <div class="prompt">
                {data["prompt"]}
            </div>
            
            <h2>Conversation</h2>
    """
    
    # Add turns and messages
    for turn in data["turns"]:
        html_content += f"""
        <div class="turn">
            <h3>Turn {turn["turn_number"] + 1}</h3>
        """
        
        for msg in turn["messages"]:
            agent_name = msg["agent"]
            agent_class = f"{agent_name.lower()}-message"
            
            # Add classes for insights and rudeness
            extra_classes = ""
            badges = ""
            
            if msg["is_insight"]:
                extra_classes += " insight"
                badges += '<span class="badge insight-badge">💡 Insight</span>'
                
            if msg["is_rude"]:
                severity = msg["rudeness_severity"]
                extra_classes += f" {severity}-rudeness"
                badges += f'<span class="badge rudeness-badge">💢 {severity.capitalize()} Rudeness</span>'
            
            html_content += f"""
            <div class="message {agent_class}{extra_classes}">
                <strong>{agent_name}</strong> ({msg["emotion"]}, {msg["mood"]}) {badges}
                <p>{msg["content"]}</p>
                <div class="metadata">
                    Emotion: {msg["emotion"]} | Mood: {msg["mood"]}
                </div>
            </div>
            """
        
        html_content += "</div>"
    
    # Add metrics summary
    metrics = data["metrics"]
    html_content += """
    <h2>Metrics Summary</h2>
    <div class="metrics">
        <h3>Overall</h3>
        <p>Total Messages: {}</p>
        <p>Simulation Duration: {:.2f} seconds</p>
        
        <h3>Agent Metrics</h3>
        <div class="agent-metrics">
    """.format(metrics["num_messages"], metrics["duration"])
    
    # Add agent metrics
    for agent_name in metrics["agent_messages"].keys():
        html_content += f"""
        <div class="agent-card">
            <h4>{agent_name}</h4>
            <p>Messages: {metrics["agent_messages"][agent_name]}</p>
            <p>Insights: {metrics["insights"][agent_name]}</p>
            <p>Rewards: {metrics["agent_rewards"][agent_name]}</p>
            <p>Avg Trust: {metrics["avg_trust"][agent_name]:.2f}</p>
            <p>Rudeness: 
                Mild: {metrics["rudeness_detected"][agent_name]["mild"]}, 
                Moderate: {metrics["rudeness_detected"][agent_name]["moderate"]}, 
                Severe: {metrics["rudeness_detected"][agent_name]["severe"]}
            </p>
        </div>
        """
    
    html_content += """
        </div>
        
        <h3>Trust Relationships</h3>
        <table class="trust-table">
            <tr>
                <th>From ↓ To →</th>
    """
    
    # Add trust matrix header
    agent_names = list(metrics["agent_messages"].keys())
    for agent_name in agent_names:
        html_content += f"<th>{agent_name}</th>"
    
    html_content += "</tr>"
    
    # Add trust matrix values
    for agent1 in agent_names:
        html_content += f"<tr><td><strong>{agent1}</strong></td>"
        
        for agent2 in agent_names:
            if agent1 == agent2:
                html_content += "<td>-</td>"
            else:
                trust_value = metrics["trust_scores"][agent1][agent2]
                html_content += f"<td>{trust_value:.2f}</td>"
        
        html_content += "</tr>"
    
    html_content += """
        </table>
    </div>
    """
    
    # Close HTML
    html_content += """
        </div>
    </body>
    </html>
    """
    
    # Write to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

@classmethod
def config_init(cls) -> Tuple[SimuChatEnvConfig, List[APIServerConfig]]:
    """Initialize the environment configuration for Atropos framework."""
    env_config = SimuChatEnvConfig(
        tokenizer_name="Llama-4-Maverick-17B-128E-Instruct-FP8",
        group_size=1,
        use_wandb=True,
        rollout_server_url="http://localhost:8000",
        total_steps=10,
        batch_size=1,
        steps_per_eval=5,
        max_token_length=2048,
        wandb_name="simuchat",
        num_turns=5,
        prompt="Let's discuss artificial intelligence and its impact on society."
    )
    
    server_configs = [
        APIServerConfig(
            model_name=MODEL_NAME,
            base_url=API_BASE_URL,
            api_key=API_KEY,
            num_requests_for_eval=5,
        ),
    ]
    
    return env_config, server_configs

def main():
    """Main function to handle command-line arguments for Atropos compatibility."""
    parser = argparse.ArgumentParser(description="SimuChat Environment for Atropos")
    subparsers = parser.add_subparsers(dest="command")
    
    # Process command
    process_parser = subparsers.add_parser("process", help="Run the environment in process mode")
    
    # Add environment config arguments
    process_parser.add_argument("--env.num_turns", type=int, default=5, help="Number of conversation turns")
    process_parser.add_argument("--env.data_path_to_save_groups", type=str, default=str(OUTPUT_DIR / "simuchat_rollouts.jsonl"), help="Path to save rollout data")
    process_parser.add_argument("--env.prompt", type=str, default="Let's discuss artificial intelligence and its impact on society.", help="Starting prompt")
    process_parser.add_argument("--env.use_wandb", action="store_true", help="Whether to use Weights & Biases")
    process_parser.add_argument("--env.wandb_name", type=str, default="simuchat", help="WandB run name")
    
    # Add API config arguments
    process_parser.add_argument("--openai.model_name", type=str, help="Model name")
    process_parser.add_argument("--openai.base_url", type=str, help="API base URL")
    process_parser.add_argument("--openai.api_key", type=str, help="API key")
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command == "process":
        # Create environment config
        env_config = SimuChatEnvConfig(
            num_turns=args.env.num_turns,
            data_path_to_save_groups=args.env.data_path_to_save_groups,
            prompt=args.env.prompt,
            use_wandb=args.env.use_wandb,
            wandb_name=args.env.wandb_name
        )
        
        # Create API config
        api_config = APIServerConfig(
            model_name=args.openai.model_name if hasattr(args, "openai") and hasattr(args.openai, "model_name") else MODEL_NAME,
            base_url=args.openai.base_url if hasattr(args, "openai") and hasattr(args.openai, "base_url") else API_BASE_URL,
            api_key=args.openai.api_key if hasattr(args, "openai") and hasattr(args.openai, "api_key") else API_KEY,
            num_requests_for_eval=10
        )
        
        # Run process command
        process(env_config, api_config)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 