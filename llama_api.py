"""
Meta LLaMA API interface for SimuChat application.
Handles API calls to the LLaMA model.
"""

import requests
import json
import sys
from config import API_KEY, API_BASE_URL, MODEL_NAME, TEMPERATURE


class LlamaAPIError(Exception):
    """Custom exception for LLaMA API errors."""
    pass


def call_llama_api(messages):
    """
    Makes a call to the Meta LLaMA API.
    
    Args:
        messages: List of message objects with role and content
        
    Returns:
        Response text from the API
        
    Raises:
        LlamaAPIError: If the API call fails
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Simplified payload based on successful test
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": TEMPERATURE
    }
    
    try:
        print(f"\nAttempting API call to: {API_BASE_URL}")
        print(f"Using model: {MODEL_NAME}")
        print(f"API Key: {API_KEY[:10]}...{API_KEY[-5:]}")
        
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
        raise LlamaAPIError(f"API request failed: {str(e)}")
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {str(e)}")
        print(f"Response content: {response.text if 'response' in locals() else 'No response'}")
        raise LlamaAPIError("Failed to decode API response as JSON")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error details: {sys.exc_info()}")
        raise LlamaAPIError(f"Unexpected error: {str(e)}")


def get_agent_response(agent_system_prompt, message_history):
    """
    Get a response from a specific agent using the LLaMA API.
    
    Args:
        agent_system_prompt: The system prompt defining the agent's personality
        message_history: The conversation history
        
    Returns:
        The agent's response text
    """
    try:
        # Create the messages array for the API
        messages = [
            {"role": "system", "content": agent_system_prompt}
        ]
        
        # Add the message history
        for msg in message_history:
            if "metadata" not in msg:  # Skip metadata for API call
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        # Call the API
        response = call_llama_api(messages)
        return response
        
    except LlamaAPIError as e:
        print(f"Error: {str(e)}")
        return "Sorry, I couldn't generate a response due to an API error." 