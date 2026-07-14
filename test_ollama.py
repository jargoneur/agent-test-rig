import os
import requests

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
url = f"{OLLAMA_HOST}/api/generate"

payload = {
    "model": "qwen2.5-coder:3b",
    "prompt": "Write a Python function add(a, b). Return only code.",
    "stream": False,
}

response = requests.post(url, json=payload, timeout=120)
response.raise_for_status()

data = response.json()
print(data["response"])