import json
import re


class SimpleAgent:
    def __init__(self, model, tools, logger, scaffold, max_steps=8):
        self.model = model
        self.tools = tools
        self.logger = logger
        self.scaffold = scaffold
        self.max_steps = max_steps

    def parse_action(self, response):
        response = response.strip()

        try:
            action = json.loads(response)
        except json.JSONDecode