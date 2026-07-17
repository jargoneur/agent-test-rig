import json
import re
from pathlib import Path


class SimpleAgent:
    def __init__(
        self,
        model,
        tools,
        logger,
        scaffold,
        max_steps=8,
        expected_files=None,
    ):
        self.model = model
        self.tools = tools
        self.logger = logger
        self.scaffold = scaffold
        self.max_steps = max_steps
        self.expected_files = set(expected_files or