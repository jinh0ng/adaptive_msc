# agents/base_agent.py
from llm import call_llm

class BaseAgent:
    def __init__(self, depth: int = 0):
        self.depth = depth

    def call(self, messages: list[dict]) -> str:
        return call_llm(messages)