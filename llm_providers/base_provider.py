import json
import re
from abc import ABC, abstractmethod
from typing import Dict, Optional


class BaseLLMProvider(ABC):

    def __init__(self, config: Dict):
        self.config = config

    @abstractmethod
    def generate(self, prompt: str, system_message: str) -> Optional[str]:
        pass


 