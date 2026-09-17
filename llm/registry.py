import logging

from llm.base_client import LLMClient
from llm.openai_client import OpenAIClient

PROVIDERS = {
    "openai": {
        "base_url": None,  # 官方 SDK 默认
        "client_cls": OpenAIClient,
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "client_cls": OpenAIClient,
    },
}

def create_client(provider: str, model: str, api_key: str) -> LLMClient:
    if provider not in PROVIDERS:
        raise ValueError(
            f"未知的 provider: {provider!r}，可选：{', '.join(PROVIDERS)}"
        )
    config = PROVIDERS[provider]
    return config["client_cls"](
        model=model,
        api_key=api_key,
        base_url=config["base_url"],
    )