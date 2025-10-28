import time
from typing import Dict, Optional
from openai import AzureOpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential
from .base_provider import BaseLLMProvider


class OpenAIProvider(BaseLLMProvider):

    def __init__(self, config: Dict):
        super().__init__(config)
        base_url = self.config.get("base_url", "https://llm-api.amd.com")
        api_version = self.config.get("api_version", "2024-06-01")
        
        headers = {
            'Ocp-Apim-Subscription-Key': self.config["api_key"] 
        }

        self.client = AzureOpenAI(
            api_key='dummy',   
            api_version=api_version,
            base_url=base_url,
            default_headers=headers,
            timeout=self.config.get("timeout_seconds", 120)
        )
        
        self.client.base_url = f'{base_url}/openai/deployments/{self.config["model_id"]}'

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
    def generate(self, prompt: str, system_message: str) -> Optional[str]:
        try:
            messages = [
                {
                    "role": "system",
                    "content": system_message
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]

            response = self.client.chat.completions.create(
                model=self.config["model_id"],
                messages=messages,
                temperature=self.config.get("temperature", 0.1),
                max_tokens=self.config.get("max_tokens", 4096),
                n=1,
                stream=False,
                stop=None,
                presence_penalty=0,
                frequency_penalty=0,
                logit_bias=None,
                user=None
            )
            
            response_text = response.choices[0].message.content
            if not response_text:
                return None

            # print(f"response_text=======>{response_text}")
            return response_text
            
        except Exception:
            return None 