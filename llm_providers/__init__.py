from typing import Dict
from .base_provider import BaseLLMProvider
from .openai_provider import OpenAIProvider

try:
    from .anthropic_provider import AnthropicProvider
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


def get_provider(provider_config: Dict) -> BaseLLMProvider:
    provider_name = provider_config.get("provider")
    
    if not provider_name:
        raise ValueError("The configuration is missing the 'provider' field")

    if provider_name == "openai":
        return OpenAIProvider(provider_config)
    elif provider_name == "anthropic":
        if not ANTHROPIC_AVAILABLE:
            raise ValueError("Anthropic provider is not available, please install anthropic library: pip install anthropic")
        return AnthropicProvider(provider_config)
    else:
        supported_providers = ["openai"]
        if ANTHROPIC_AVAILABLE:
            supported_providers.append("anthropic")
        raise ValueError(f"Unsupported provider: {provider_name}. Supported providers: {', '.join(supported_providers)}")

