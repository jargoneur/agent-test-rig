import os

import requests


def main():
    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    url = f"{ollama_host}/api/generate"
    payload = {
        "model": "qwen2.5-coder:3b",
        "prompt": "Write a Python function add(a, b). Return only code.",
        "stream": False,
    }

    response = requests.post(url, json=payload, timeout=120)
    response.raise_for_status()
    print(response.json()["response"])


if __name__ == "__main__":
    main()