"""
Logger module for SimuChat.
Handles logging conversation data and generating reports.
"""

import json
import time
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import env


class Logger:
    """Class to handle logging of conversation data."""
    
    def __init__(self):
        """Initialize the logger."""
        self.jsonl_path = env.get_jsonl_log_path()
        self.html_path = env.get_html_log_path()
        self.session_id = int(time.time())
        self.log_entries = []
        
        # Initialize log files
        self._ensure_output_directory()
    
    def _ensure_output_directory(self):
        """Ensure the output directory exists."""
        output_dir = env.OUTPUT_DIR
        output_dir.mkdir(exist_ok=True)
    
    def log_message(self, message: Dict[str, Any], trust_snapshot: Optional[Dict] = None, rewards_info: Optional[Dict] = None):
        """Log a message to the log files.
        
        Args:
            message: The message to log
            trust_snapshot: Optional snapshot of current trust scores
            rewards_info: Optional information about rewards earned
        """
        # Create log entry
        timestamp = datetime.datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "session_id": self.session_id,
            "message": message.copy(),
            "trust_snapshot": trust_snapshot
        }
        
        # Add rewards info if provided
        if rewards_info:
            log_entry["rewards_info"] = rewards_info
        
        # Add to memory
        self.log_entries.append(log_entry)
        
        # Append to JSONL file
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
        
        # Update HTML log
        self._update_html_log()
    
    def log_insight(self, agent_name: str, message: str, trust_snapshot: Optional[Dict] = None):
        """Log an insight to the log files.
        
        Args:
            agent_name: The agent who had the insight
            message: The insight message
            trust_snapshot: Optional snapshot of current trust scores
        """
        timestamp = datetime.datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "session_id": self.session_id,
            "type": "insight",
            "agent_name": agent_name,
            "message": message,
            "trust_snapshot": trust_snapshot
        }
        
        # Add to memory
        self.log_entries.append(log_entry)
        
        # Append to JSONL file
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
        
        # Update HTML log
        self._update_html_log()
    
    def _update_html_log(self):
        """Update the HTML log file with current conversation data."""
        # Basic HTML template
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>SimuChat Conversation Log</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 800px;
                    margin: 0 auto;
                    background-color: #fff;
                    border-radius: 8px;
                    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
                    padding: 20px;
                }}
                h1 {{
                    color: #333;
                    border-bottom: 1px solid #eee;
                    padding-bottom: 10px;
                }}
                .message {{
                    padding: 10px 15px;
                    margin: 10px 0;
                    border-radius: 8px;
                }}
                .user-message {{
                    background-color: #e9f5ff;
                    border-left: 3px solid #2c86fc;
                }}
                .agent-message {{
                    background-color: #f0f0f0;
                    border-left: 3px solid #999;
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
                    position: relative;
                }}
                .insight::before {{
                    content: "💡";
                    position: absolute;
                    left: -10px;
                    top: 5px;
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
                .rudeness-indicator {{
                    color: #f44336;
                    font-weight: bold;
                    margin-right: 5px;
                }}
                .metadata {{
                    color: #888;
                    font-size: 0.8em;
                    margin-top: 5px;
                }}
                .trust-data {{
                    background-color: #f9f9f9;
                    margin: 10px 0;
                    padding: 10px;
                    border-radius: 5px;
                }}
                .trust-bar {{
                    height: 15px;
                    background-color: #eee;
                    border-radius: 5px;
                    margin: 5px 0;
                    overflow: hidden;
                }}
                .trust-level {{
                    height: 100%;
                    background-color: #4CAF50;
                }}
                .trust-decrease {{
                    background-color: #f44336;
                }}
                .rewards {{
                    background-color: #fff8e1;
                    padding: 8px 12px;
                    margin: 5px 0;
                    border-radius: 4px;
                    border-left: 3px solid #ffc107;
                }}
                .reward-badge {{
                    display: inline-block;
                    background-color: #ffc107;
                    color: #333;
                    padding: 2px 6px;
                    border-radius: 20px;
                    font-size: 0.8em;
                    margin-right: 5px;
                }}
                .rewards-summary {{
                    background-color: #e8f5e9;
                    margin: 20px 0;
                    padding: 15px;
                    border-radius: 5px;
                    border-left: 3px solid #4CAF50;
                }}
                .rewards-summary h3 {{
                    margin-top: 0;
                    color: #2E7D32;
                }}
                .agent-rewards {{
                    display: flex;
                    align-items: center;
                    margin: 5px 0;
                }}
                .agent-rewards .points {{
                    font-weight: bold;
                    margin-left: 10px;
                }}
                .trust-change {{
                    padding: 3px 8px;
                    border-radius: 4px;
                    font-size: 0.85em;
                    margin-top: 5px;
                    display: inline-block;
                }}
                .trust-increase {{
                    background-color: #e8f5e9;
                    color: #2E7D32;
                }}
                .trust-decrease {{
                    background-color: #ffebee;
                    color: #c62828;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>SimuChat Conversation Log</h1>
                <div id="messages">
                    {messages}
                </div>
            </div>
        </body>
        </html>
        """
        
        messages_html = []
        
        for entry in self.log_entries:
            if "type" in entry and entry["type"] == "insight":
                # Format insight entry
                messages_html.append(f"""
                <div class="message insight">
                    <strong>{entry["agent_name"]} had an insight:</strong>
                    <p>{entry["message"]}</p>
                    <div class="metadata">
                        <span>{entry["timestamp"]}</span>
                    </div>
                </div>
                """)
            elif "message" in entry:
                msg = entry["message"]
                if msg.get("role") == "user":
                    # Format user message
                    messages_html.append(f"""
                    <div class="message user-message">
                        <strong>You:</strong>
                        <p>{msg["content"]}</p>
                        <div class="metadata">
                            <span>{entry["timestamp"]}</span>
                        </div>
                    </div>
                    """)
                elif msg.get("role") == "assistant" and "metadata" in msg:
                    # Format agent message
                    metadata = msg["metadata"]
                    agent_name = metadata.get("agent_name", "Agent")
                    emotion = metadata.get("emotion", "neutral")
                    mood = metadata.get("mood", "")
                    mood_text = f", {mood}" if mood else ""
                    agent_class = f"{agent_name.lower()}-message"
                    
                    is_insight = metadata.get("is_insight", False)
                    insight_class = " insight" if is_insight else ""
                    
                    # Check for rudeness - we need to detect it here for display
                    from utils import detect_rudeness
                    is_rude, rudeness_severity = detect_rudeness(agent_name, msg["content"])
                    
                    rudeness_class = ""
                    rudeness_indicator = ""
                    if is_rude:
                        rudeness_class = f" {rudeness_severity}-rudeness"
                        rudeness_indicator = f'<span class="rudeness-indicator">💢 {rudeness_severity.capitalize()} Rudeness</span>'
                    
                    messages_html.append(f"""
                    <div class="message agent-message {agent_class}{insight_class}{rudeness_class}">
                        <strong>{agent_name} ({emotion}{mood_text}):</strong> {rudeness_indicator}
                        <p>{msg["content"]}</p>
                        <div class="metadata">
                            <span>{entry["timestamp"]}</span>
                        </div>
                    </div>
                    """)
                    
                    # Add reward information if available
                    if "rewards_info" in entry and entry["rewards_info"]["rewards_earned"] > 0:
                        rewards_info = entry["rewards_info"]
                        rewards_html = f"""
                        <div class="rewards">
                            <span class="reward-badge">🏆 +{rewards_info["rewards_earned"]}</span>
                            {', '.join(rewards_info["reasons"])}
                        </div>
                        """
                        messages_html.append(rewards_html)
                
                # Add trust data if available
                if "trust_snapshot" in entry and entry["trust_snapshot"]:
                    trust_html = ["<div class='trust-data'><h3>Trust Network</h3>"]
                    
                    for agent, trust_data in entry["trust_snapshot"].items():
                        trust_html.append(f"<h4>{agent}'s Trust:</h4>")
                        
                        for target, trust_info in trust_data.items():
                            trust_value = trust_info["new_value"]
                            trust_percent = int(trust_value * 100)
                            trust_change = trust_info["change"]
                            reason = trust_info["reason"]
                            
                            # Special display for rudeness-related trust changes
                            rudeness_info = ""
                            if "rudeness" in reason:
                                change_class = "trust-decrease" if trust_change < 0 else "trust-increase"
                                severity = reason.split('_')[-1] if '_' in reason else "unknown"
                                rudeness_info = f"""
                                <div class="trust-change {change_class}">
                                    {round(trust_change * 100)}% trust due to {severity} rudeness
                                </div>
                                """
                            
                            bar_class = "trust-level"
                            if trust_change < -0.05:  # Significant decrease
                                bar_class += " trust-decrease"
                            
                            trust_html.append(f"""
                            <div>
                                <strong>{target}:</strong> {trust_value:.2f}
                                <div class="trust-bar">
                                    <div class="{bar_class}" style="width: {trust_percent}%;"></div>
                                </div>
                                {rudeness_info}
                            </div>
                            """)
                    
                    trust_html.append("</div>")
                    messages_html.append("".join(trust_html))
        
        # Combine all message HTML
        all_messages_html = "".join(messages_html)
        
        # Add rewards summary at the end
        rewards_summary = self._generate_rewards_summary()
        if rewards_summary:
            all_messages_html += rewards_summary
        
        # Replace placeholder in template
        html_content = html_template.format(messages=all_messages_html)
        
        # Write to HTML file
        with open(self.html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
    
    def _generate_rewards_summary(self) -> str:
        """Generate HTML for the rewards summary section.
        
        Returns:
            HTML string for the rewards summary
        """
        # Collect all agents and their rewards from log entries
        agent_rewards = {}
        
        for entry in self.log_entries:
            if "rewards_info" in entry and entry["rewards_info"]["agent"] not in agent_rewards:
                agent_rewards[entry["rewards_info"]["agent"]] = 0
            
            if "rewards_info" in entry:
                agent_name = entry["rewards_info"]["agent"]
                points = entry["rewards_info"]["rewards_earned"]
                agent_rewards[agent_name] = agent_rewards.get(agent_name, 0) + points
        
        if not agent_rewards:
            return ""
        
        # Generate the HTML
        html = ["<div class='rewards-summary'><h3>🏆 Rewards Summary</h3>"]
        
        # Sort agents by points (highest first)
        sorted_agents = sorted(agent_rewards.items(), key=lambda x: x[1], reverse=True)
        
        for agent_name, points in sorted_agents:
            html.append(f"""
            <div class="agent-rewards">
                <strong>{agent_name}:</strong> <span class="points">{points} points</span>
            </div>
            """)
        
        html.append("</div>")
        return "".join(html)
    
    def start_new_session(self):
        """Start a new logging session with a new session ID."""
        self.session_id = int(time.time())
        
    def get_log_entries(self) -> List[Dict[str, Any]]:
        """Get all log entries for the current session.
        
        Returns:
            List of log entries
        """
        return self.log_entries 