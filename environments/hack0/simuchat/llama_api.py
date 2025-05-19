"""
Meta LLaMA API interface for SimuChat application.
Handles API calls to the LLaMA model.
"""

import requests
import json
import sys
import time
from typing import List, Dict, Any, Optional
from env import API_KEY, API_BASE_URL, MODEL_NAME


class LlamaAPIError(Exception):
    """Custom exception for LLaMA API errors."""
    pass


def call_llama_api(messages: List[Dict[str, str]], temperature: float = 0.6, max_retries: int = 3) -> str:
    """
    Makes a call to the Meta LLaMA API.
    
    Args:
        messages: List of message objects with role and content
        temperature: Temperature parameter for generation (0.0 to 1.0)
        max_retries: Maximum number of retry attempts
        
    Returns:
        Response text from the API
        
    Raises:
        LlamaAPIError: If the API call fails after all retries
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Payload for the API
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature
    }
    
    retry_count = 0
    while retry_count < max_retries:
        try:
            print(f"\nAttempting API call (try {retry_count + 1}/{max_retries})...")
            print(f"Using model: {MODEL_NAME}")
            
            response = requests.post(
                API_BASE_URL,
                headers=headers,
                json=payload,
                timeout=60
            )
            
            # Print response status for debugging
            print(f"Response status code: {response.status_code}")
            
            # Check if the request was successful
            response.raise_for_status()
            
            # Parse the response (Meta's format is different from OpenAI)
            result = response.json()
            
            # Extract the generated text based on Meta's response format
            if "completion_message" in result and "content" in result["completion_message"]:
                content = result["completion_message"]["content"]
                # Handle both text and structured content
                if isinstance(content, dict) and "text" in content:
                    return content["text"]
                elif isinstance(content, str):
                    return content
                else:
                    print(f"Unexpected content format: {content}")
                    raise LlamaAPIError("Unexpected content format in API response")
            else:
                print(f"Unexpected response format: {result}")
                raise LlamaAPIError("Unexpected API response format")
                
        except requests.exceptions.RequestException as e:
            print(f"Request Exception: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response content: {e.response.text}")
        except json.JSONDecodeError as e:
            print(f"JSON Decode Error: {str(e)}")
            if 'response' in locals():
                print(f"Response content: {response.text}")
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            print(f"Error type: {type(e).__name__}")
        
        # Increment retry count and wait before retrying
        retry_count += 1
        if retry_count < max_retries:
            wait_time = 2 ** retry_count  # Exponential backoff
            print(f"Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
    
    # If we've exhausted all retries, raise an exception
    raise LlamaAPIError(f"API request failed after {max_retries} attempts")


def get_agent_response(
    agent_name: str,
    agent_system_prompt: str,
    message_history: List[Dict[str, Any]],
    memory_context: str = "",
    temperature: float = 0.6
) -> str:
    """
    Get a response from a specific agent using the LLaMA API.
    
    Args:
        agent_name: Name of the agent
        agent_system_prompt: The system prompt defining the agent's personality
        message_history: The conversation history
        memory_context: Additional context from agent memory
        temperature: Temperature parameter for generation
        
    Returns:
        The agent's response text
    """
    try:
        # Create the messages array for the API
        # Include memory context in the system prompt if provided
        enhanced_prompt = agent_system_prompt
        if memory_context:
            enhanced_prompt = f"{agent_system_prompt}\n\nYour memory context: {memory_context}"
        
        messages = [
            {"role": "system", "content": enhanced_prompt}
        ]
        
        # Add the message history
        for msg in message_history:
            if msg.get("role") in ["user", "assistant"]:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        # Call the API with the agent-specific temperature
        response = call_llama_api(messages, temperature=temperature)
        return response
        
    except LlamaAPIError as e:
        print(f"Error getting response for {agent_name}: {str(e)}")
        return f"Sorry, I couldn't generate a response due to an API error." 