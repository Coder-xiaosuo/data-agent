"""registry 的单元测试：provider 路由、校验与 env 取 key。

SDK 被替身接管，只断言 provider 配置最终传到 SDK 构造参数上。
"""
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

import llm.openai_client as openai_client_module
from llm.base_client import LLMClient
from llm.openai_client import OpenAIClient
from llm.registry import (
    PROVIDERS,
    _PROJECT_ROOT,
    create_client,
    create_client_from_env,
)


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


# --------------------------------------------------------------------------
# create_client_from_env
# --------------------------------------------------------------------------


def test_project_root_is_anchored_where_dotenv_lives(sdk_kwargs):
    """_PROJECT_ROOT 由 parents[1] 推导，若 registry.py 移位会静默读不到 .env。"""
    assert (_PROJECT_ROOT / "pytest.ini").is_file()
    assert _PROJECT_ROOT == Path(__file__).resolve().parents[1]


def test_from_env_reads_provider_specific_variable(sdk_kwargs, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-from-env")
    create_client_from_env("deepseek", "deepseek-chat")
    assert sdk_kwargs["api_key"] == "sk-from-env"


def test_from_env_routes_to_provider_endpoint(sdk_kwargs, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-from-env")
    create_client_from_env("deepseek", "deepseek-chat")
    assert sdk_kwargs["base_url"] == "https://api.deepseek.com/v1"


def test_from_env_uses_distinct_variable_per_provider(sdk_kwargs, monkeypatch):
    """openai 的 key 不能读到 deepseek 的变量。"""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek")
    with pytest.raises(ValueError):
        create_client_from_env("openai", "gpt-4o-mini")


def test_from_env_returns_llm_client(sdk_kwargs, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-from-env")
    assert isinstance(create_client_from_env("deepseek", "deepseek-chat"), LLMClient)


def test_from_env_missing_key_raises_value_error(sdk_kwargs, monkeypatch):
    """缺 key 必须快速失败，不能带着空 key 发起注定 401 的请求。"""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(ValueError):
        create_client_from_env("deepseek", "deepseek-chat")


def test_from_env_missing_key_error_names_the_variable(sdk_kwargs, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(ValueError) as exc:
        create_client_from_env("deepseek", "deepseek-chat")
    assert "DEEPSEEK_API_KEY" in str(exc.value)


@pytest.mark.parametrize("blank", ["", "   "])
def test_from_env_blank_key_treated_as_missing(sdk_kwargs, monkeypatch, blank):
    monkeypatch.setenv("DEEPSEEK_API_KEY", blank)
    with pytest.raises(ValueError):
        create_client_from_env("deepseek", "deepseek-chat")


def test_from_env_rejects_unknown_provider(sdk_kwargs):
    with pytest.raises(ValueError) as exc:
        create_client_from_env("gemini", "gemini-pro")
    assert "gemini" in str(exc.value)


def test_from_env_never_calls_sdk_when_key_missing(sdk_kwargs, monkeypatch):
    """校验必须先于构造，否则会先用空 key 建出一个坏客户端。"""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(ValueError):
        create_client_from_env("deepseek", "deepseek-chat")
    assert sdk_kwargs == {}
