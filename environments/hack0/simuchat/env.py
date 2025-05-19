"""
Environment settings for SimuChat application.
Contains API details and model configuration.
"""

import os
import json
from pathlib import Path

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