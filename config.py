"""
Configuration settings for SimuChat application.
Contains API details and model configuration.
"""

# Meta LLaMA API configuration
API_KEY = "LLM|1050039463644017|ZZzxjun1klZ76kW0xu5Zg4BW5-o"  # API key
API_BASE_URL = "https://api.llama.com/v1/chat/completions"  # API endpoint that works
MODEL_NAME = "Llama-4-Maverick-17B-128E-Instruct-FP8"  # Using Maverick model

# Temperature setting for response generation
# Higher values (e.g., 0.7) make output more random
# Lower values (e.g., 0.2) make output more deterministic
TEMPERATURE = 0.6  # Updated to match working test

# Maximum number of messages to keep in history
MAX_HISTORY_LENGTH = 10 