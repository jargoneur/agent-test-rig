import os
import requests


class OllamaModel:
    def __init__(self, model_name, host=None):
        self.model_name = model_name
        self.host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    def generate(self, prompt, temperature=0.2, timeout=600):
        url = f"{self.host}/api/generate"

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }

        response = requests.post(url, json=payload, timeout=180)
        response.raise_for_status()

        return response.json()["response"]