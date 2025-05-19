# SimuChat

A WhatsApp-style fake group chat simulation between 3 AI agents (Alice, Bob, Charlie) using Meta's LLaMA API.

## Description

SimuChat simulates a group chat where three AI agents with distinct personalities respond to each other in a conversation thread. You provide a starting topic, and the agents discuss it in a realistic way, each displaying emotions along with their responses.

### AI Agent Personalities

- **Alice** - Kind and empathetic. Prioritizes emotions, empathy, and harmony.
- **Bob** - Logical and analytical. Focuses on facts, reason, and structured thinking.
- **Charlie** - Bold and competitive. Enjoys debate, leadership, and proving opinions.

## Installation

1. Clone this repository to your local machine
2. Ensure Python 3.7+ is installed
3. Install the required dependencies:

```
pip install -r requirements.txt
```

## Configuration

Before running the application, you need to provide your Meta LLaMA API key:

1. Open the `config.py` file
2. Replace `YOUR_META_API_KEY` with your actual Meta API key using the format:
   ```python
   API_KEY = "LLM|1050039463644017|ZZzxjun1klZ76kW0xu5Zg4BW5-o"
   ```
   The key consists of three parts separated by the pipe (|) character:
   - A service identifier: LLM
   - A numeric ID
   - A token string

## Model Information

This application uses the **Llama-4-Maverick-17B-128E-Instruct-FP8** model, which provides powerful AI capabilities with:
- 3,000 tokens per request
- 1,000,000 tokens monthly quota

## Running the Application

To start the application, run:

```
python main.py
```

1. Enter a starting topic when prompted
2. Watch as Alice, Bob, and Charlie respond to each other
3. After all three agents have responded, you can:
   - Press Enter to continue the conversation
   - Type a new message to steer the discussion
   - Type 'quit' to exit the application

## Adding More Agents

To add more AI agents to the simulation:

1. Open `agents.py`
2. Add a new agent to the `AGENTS` dictionary:
   ```python
   "YourAgentName": {
       "name": "YourAgentName",
       "emoji": "🔮",  # Choose an appropriate emoji
       "system_prompt": "You are YourAgentName, with X, Y, Z personality traits..."
   }
   ```
3. Add the agent name to the `AGENT_NAMES` list in the order you want them to respond

## File Structure

- `main.py` - Entry point of the application
- `config.py` - API configuration settings
- `agents.py` - Agent personality definitions
- `llama_api.py` - LLaMA API interface
- `utils.py` - Utility functions for output formatting and display

## Dependencies

- `requests` - For making API calls to Meta's LLaMA API

## Error Handling

The application includes basic error handling for API failures. If an API call fails, the agent will respond with an error message and the simulation will continue.

## License

This project is open source and available for personal and educational use.

## Disclaimer

This application requires a valid Meta LLaMA API key to function. The API usage might incur costs based on your agreement with Meta. 