"""
Test script for Meta LLaMA API using the OpenAI compatible interface.
"""

import os
import json
import requests
from openai import OpenAI

# Set up API key 
API_KEY = "LLM|1050039463644017|ZZzxjun1klZ76kW0xu5Zg4BW5-o"
API_BASE_URL = "https://api.llama.com/v1"

# Try direct request approach first for debugging
print("Testing direct API request...")
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

payload = {
    "model": "Llama-4-Maverick-17B-128E-Instruct-FP8",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello!"}
    ],
    "temperature": 0.6
}

print(f"API URL: {API_BASE_URL}/chat/completions")
print(f"API Key: {API_KEY}")
print(f"Headers: {headers}")
print(f"Payload: {json.dumps(payload, indent=2)}")

try:
    # Make direct request first
    response = requests.post(
        f"{API_BASE_URL}/chat/completions",
        headers=headers,
        json=payload
    )
    
    print(f"\nDirect Request Status Code: {response.status_code}")
    print(f"Response Headers: {response.headers}")
    print(f"Response Content: {response.text}")
    
except Exception as e:
    print(f"\nDirect Request Error: {str(e)}")

# Now try the OpenAI client approach
try:
    print("\n\nTesting with OpenAI client...")
    # Set environment variables
    os.environ["OPENAI_API_KEY"] = API_KEY
    os.environ["OPENAI_BASE_URL"] = API_BASE_URL
    
    # Create client
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"]
    )
    
    # Make a simple completion call
    print("Sending request via OpenAI client...")
    completion = client.chat.completions.create(
        model="Llama-4-Maverick-17B-128E-Instruct-FP8",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello!"}
        ],
        temperature=0.6
    )
    
    print("\nResponse:")
    if completion and hasattr(completion, 'choices') and completion.choices:
        print(completion.choices[0].message.content)
    else:
        print(f"Unexpected response format: {completion}")
    
except Exception as e:
    print(f"\nOpenAI Client Error: {str(e)}")
    print(f"Error Type: {type(e).__name__}")

print("\nTest completed.") 