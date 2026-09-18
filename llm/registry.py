import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from llm.base_client import LLMClient
from llm.openai_client import OpenAIClient

logger = logging.getLogger(__name__)

# 锚定到项目根，而不是当前工作目录——否则从子目录启动时读不到 .env。
# load_dotenv 默认不覆盖已存在的环境变量，因此真实环境变量优先于 .env。
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_PROJECT_ROOT / ".env")

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


def _api_key_env_var(provider: str) -> str:
    """API key 的环境变量名约定：<PROVIDER>_API_KEY"""
    return f"{provider.upper()}_API_KEY"


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


def create_client_from_env(provider: str, model: str) -> LLMClient:
    """从环境变量读取 API key 构造客户端"""
    if provider not in PROVIDERS:
        raise ValueError(
            f"未知的 provider: {provider!r}，可选：{', '.join(PROVIDERS)}"
        )

    env_var = _api_key_env_var(provider)
    # strip 掉 .env 里常见的行尾空格与换行；纯空白等同于未设置。
    api_key = (os.environ.get(env_var) or "").strip()
    if not api_key:
        raise ValueError(
            f"环境变量 {env_var} 未设置。请在项目根目录的 .env 中配置"
        )

    logger.debug("使用 %s 构造 provider=%s 的客户端，model=%s", env_var, provider, model)
    return create_client(provider=provider, model=model, api_key=api_key)
