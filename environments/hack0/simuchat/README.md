# SimuChat - AI Group Chat with Memory, Trust, and Rewards

SimuChat is a WhatsApp-style group chat simulator that creates conversations between three AI agents (Alice, Bob, and Charlie), each with distinct personalities. The agents maintain memory of past conversations, develop trust relationships with each other, earn rewards for positive interactions, and can have "insight moments" during the discussion.

![SimuChat Screenshot](https://via.placeholder.com/800x450?text=SimuChat+Screenshot)

## Features

- **3 Unique AI Personas**: Alice (empathetic), Bob (logical), and Charlie (competitive)
- **Agent Memory**: Each agent remembers previous messages and builds context
- **Insight Detection**: Identifies when agents change their mind or have realizations
- **Trust Engine**: Trust scores evolve between agents based on agreement/disagreement
- **Rudeness Detection**: Identifies when agents speak rudely to each other, decreasing trust
- **Reward System**: Agents earn points for increasing trust and having insights
- **Emotion + Mood Simulation**: Dynamic emotions and moods affect agent responses
- **Automatic Conversation**: Agents can continue chatting autonomously
- **Configurable Agent Setup**: Easy to add or modify agents
- **Logging**: Detailed conversation logs in JSONL and HTML formats
- **Console and Streamlit UI**: Both terminal and web interfaces available

## Trust and Rudeness

SimuChat implements a realistic trust system where agents' trust in each other can increase or decrease based on interactions:

### Trust Building
- Agreement between agents increases trust
- Similar viewpoints increase trust slightly
- Mentioning another agent positively increases trust

### Trust Breakdown through Rudeness
- SimuChat can detect when agents are rude to each other with three levels of severity:
  - **Mild Rudeness**: Phrases like "that's absurd" or "you're mistaken" (-5% to -10% trust)
  - **Moderate Rudeness**: Phrases like "shut up" or "you're being stupid" (-10% to -20% trust)
  - **Severe Rudeness**: Direct insults like "you're an idiot" (-20% to -30% trust)
  
- Directed rudeness (specifically naming another agent while being rude) causes 50% more trust damage
- Trust changes from rudeness are visually highlighted in the HTML log with red indicators
- The rudeness detection makes conversations more realistic, as agents that behave rudely will lose the trust of others quickly

### Trust Visualization
- Trust is displayed as a percentage between agents
- Trust levels affect agents' moods (defensive, contemplative, supportive, collaborative)
- The HTML logs show trust changes with color-coded indicators
- Significant trust drops from rudeness are highlighted with special formatting

## Demo

Check out a demo of SimuChat in action: [SimuChat Demo](https://youtu.be/placeholder)

## Installation

1. Clone this repository
2. Install the required dependencies:
```
pip install -r requirements.txt
```

## Configuration

Your Meta LLaMA API key should be set in the `env.py` file:

```python
API_KEY = "YOUR_META_API_KEY"  # Format: "LLM|1050039463644017|ZZzxjun1klZ76kW0xu5Zg4BW5-o"
```

## Running SimuChat

### Atropos / Headless Mode

SimuChat can be run programmatically or as part of an automated system:

```python
from environments.hack0.simuchat.env import run

# Run simulation for 5 turns with prompt
metrics = run(
    prompt="Discuss the impact of AI on society", 
    num_turns=5, 
    headless=True
)
```

This will generate:
- `output/chatlog.jsonl` - Full conversation log
- `output/chatlog.html` - HTML formatted log
- `output/metrics.jsonl` - Metrics for WandB compatibility

### Terminal Interface

Run the console application:

```
python main.py
```

Features:
- Enter a topic to start the conversation
- Type 'auto' to start automatic conversation mode (agents continue chatting)
- Type 'stop' to end automatic mode
- Type 'quit' to exit the application

### Streamlit UI (Optional)

Run the Streamlit web interface:

```
streamlit run streamlit_app.py
```

The Streamlit UI includes:
- Toggle for automatic conversation mode
- Real-time trust network visualization
- Reward badges and leaderboard
- Download conversation logs as HTML

## How It Works

1. The user types a starting topic (e.g., "Let's discuss climate change")
2. Alice, Bob, and Charlie respond in sequence, using their unique personalities
3. Each agent maintains memory of previous messages
4. Trust scores between agents evolve based on agreement/disagreement
5. Agents earn rewards for increasing trust and having insights
6. The system tracks "insight moments" when agents change perspective
7. In automatic mode, agents continue chatting with minimal user intervention
8. A log of the conversation is saved in both JSONL and HTML formats

## Metrics and Analysis

SimuChat tracks several metrics during the conversation that can be used for analysis:

| Metric | Description |
|--------|-------------|
| `num_messages` | Total number of messages exchanged |
| `agent_messages` | Number of messages from each agent |
| `insights` | Count of insight moments per agent |
| `emotions` | Distribution of emotions expressed by each agent |
| `moods` | Distribution of moods experienced by each agent |
| `trust_scores` | Trust relationships between every agent pair |
| `agent_rewards` | Points earned by each agent for trust and insights |
| `avg_trust` | Average trust score for each agent |
| `duration` | Total running time of the simulation |

These metrics are saved in `metrics.jsonl` and can be loaded directly into WandB for visualization.

## Agent Reward System

Agents earn points in two ways:
- **+1 point**: When another agent's trust toward them increases
- **+2 points**: When they have an "insight moment" (changing stance, agreeing unexpectedly)

The reward system affects agents in several ways:
- Agents are aware of their rewards and rank in the simulation
- Rewards are displayed in the UI next to messages
- A leaderboard shows the current points for each agent
- Rewards are included in the HTML and JSONL logs

## Adding More Agents

1. Open `agents_config.json`
2. Add a new agent definition:

```json
{
  "name": "Diana",
  "emoji": "🌟",
  "system_prompt": "You are Diana, a visionary and creative AI...",
  "temperature": 0.7,
  "core_emotion": "inspired",
  "mood": "excited",
  "memory_limit": 3,
  "initial_trust": 0.5
}
```

## Project Structure

```
/simuchat/
├── env.py                # Environment settings and run() function
├── agents_config.json    # Agent personalities and config
├── main.py               # Console application entry point
├── llama_api.py          # Meta LLaMA API integration
├── memory.py             # Agent memory management
├── trust.py              # Trust relationship engine
├── rewards.py            # Agent reward system
├── utils.py              # Utility functions
├── logger.py             # Logging utilities
├── streamlit_app.py      # Streamlit UI (optional)
├── output/               # Generated logs
│   ├── chatlog.jsonl     # JSON Lines log
│   ├── chatlog.html      # HTML formatted log
│   ├── metrics.jsonl     # Metrics for WandB
├── README.md             # This file
└── requirements.txt      # Dependencies
```

## Technical Details

- **API**: Uses Meta's LLaMA API (model: `Llama-4-Maverick-17B-128E-Instruct-FP8`)
- **Memory**: Each agent maintains memory of the last 3 messages from other agents
- **Trust**: Trust scores range from 0.0 to 1.0, starting at 0.5
- **Rewards**: Points accumulate based on trust increases and insights
- **Auto-conversation**: Can run up to 50 rounds autonomously
- **Insight Detection**: Uses pattern matching and agreement/disagreement analysis
- **Logging**: Detailed logs are saved for later analysis

## License

This project is available for personal and educational use.

## Acknowledgments

- Built using Meta's LLaMA API
- Inspired by real WhatsApp group chats 