"""registry 的单元测试：provider 路由与校验。

SDK 被替身接管，只断言 provider 配置最终传到 SDK 构造参数上。
"""
import logging
from types import SimpleNamespace

import pytest

import llm.openai_client as openai_client_module
from llm.base_client import LLMClient
from llm.openai_client import OpenAIClient
from llm.registry import PROVIDERS, create_client


@pytest.fixture
def sdk_kwargs(monkeypatch):
    """捕获传给 OpenAI(...) 的构造参数。"""
    captured: dict = {}

    class _FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.chat = SimpleNamespace(completions=None)

    monkeypatch.setattr(openai_client_module, "OpenAI", _FakeOpenAI)
    return captured


def test_returns_llm_client_implementation(sdk_kwargs):
    client = create_client("openai", "gpt-4o-mini", "sk-x")
    assert isinstance(client, LLMClient)
    assert isinstance(client, OpenAIClient)


def test_model_wired_to_client(sdk_kwargs):
    assert create_client("openai", "gpt-4o-mini", "sk-x").model == "gpt-4o-mini"


def test_api_key_forwarded_to_sdk(sdk_kwargs):
    create_client("openai", "gpt-4o-mini", "sk-secret")
    assert sdk_kwargs["api_key"] == "sk-secret"


def test_openai_provider_defers_to_sdk_default_base_url(sdk_kwargs):
    create_client("openai", "gpt-4o-mini", "sk-x")
    assert sdk_kwargs["base_url"] is None


def test_deepseek_provider_routes_to_deepseek_endpoint(sdk_kwargs):
    create_client("deepseek", "deepseek-chat", "sk-x")
    assert sdk_kwargs["base_url"] == "https://api.deepseek.com/v1"


def test_every_registered_provider_is_constructible(sdk_kwargs):
    for provider in PROVIDERS:
        assert isinstance(create_client(provider, "m", "k"), LLMClient)


def test_unknown_provider_raises_value_error(sdk_kwargs):
    with pytest.raises(ValueError) as exc:
        create_client("gemini", "gemini-pro", "sk-x")


def test_unknown_provider_error_lists_available_options(sdk_kwargs):
    with pytest.raises(ValueError) as exc:
        create_client("gemini", "gemini-pro", "sk-x")
    message = str(exc.value)
    assert "gemini" in message
    assert "openai" in message
    assert "deepseek" in message
