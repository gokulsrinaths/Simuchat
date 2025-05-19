"""
Test script for direct API call to Meta LLaMA API.
"""

import requests
import json

# API configuration
# Try without the "LLM|" prefix, just the raw token portion
API_KEY = "ZZzxjun1k1Z76kW0xu5Zg4BW5-o"
API_BASE_URL = "https://api.llama.com/v1/chat/completions"
MODEL_NAME = "Llama-4-Maverick-17B-128E-Instruct-FP8"

# Simple message
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Say hello!"}
]

# Headers and payload
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "Accept": "text/event-stream"
}

payload = {
    "model": MODEL_NAME,
    "messages": messages,
    "repetition_penalty": 1,
    "temperature": 0.6,
    "top_p": 0.9,
    "max_completion_tokens": 2048,
    "stream": False
}

print("Testing API call with modified key format...")
print(f"API URL: {API_BASE_URL}")
print(f"API Key: {API_KEY}")
print(f"Model: {MODEL_NAME}")
print(f"Payload: {json.dumps(payload, indent=2)}")

try:
    response = requests.post(
        API_BASE_URL,
        headers=headers,
        json=payload,
        timeout=60
    )
    
    print(f"\nResponse Status Code: {response.status_code}")
    print(f"Response Headers: {dict(response.headers)}")
    
    # Print full response text
    print("\nResponse Content:")
    print(response.text)
    
    # Try to parse JSON response if possible
    if response.status_code == 200:
        try:
            result = response.json()
            print("\nParsed JSON Response:")
            print(json.dumps(result, indent=2))
        except json.JSONDecodeError:
            print("\nCould not parse response as JSON")
    
except Exception as e:
    print(f"\nError: {str(e)}")
    print(f"Error Type: {type(e).__name__}")

print("\nTest completed.") 