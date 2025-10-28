import time
from typing import Dict, Optional
from .base_provider import BaseLLMProvider
from tenacity import retry, wait_random_exponential, stop_after_attempt

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


class AnthropicProvider(BaseLLMProvider):

    def __init__(self, config: Dict):
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic provider requires anthropic library: pip install anthropic")
        
        super().__init__(config)
        self.client = anthropic.Anthropic(
            api_key=self.config["api_key"],
            timeout=self.config.get("timeout_seconds", 120)
        )

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
    def generate(self, prompt: str, system_message: str) -> Optional[str]:
        try:
            response = self.client.messages.create(
                model=self.config["model_id"],
                max_tokens=self.config.get("max_tokens", 4096),
                temperature=self.config.get("temperature", 0.1),
                system=system_message,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            response_text = response.content[0].text
            if not response_text:
                return None
            
            return response_text
            
        except Exception:
            return None 