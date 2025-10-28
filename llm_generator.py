from typing import Dict, Optional
from llm_providers import get_provider


class LLMGenerator:

    def __init__(self, config: Dict):
        self.config = config
        self.provider = get_provider(config)

    def generate(self, prompt: str, system_message: str) -> Optional[str]:
        try:
            return self.provider.generate(prompt, system_message=system_message)
        except Exception as e:
            return None

 