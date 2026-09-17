"""OpenAIClient 的单元测试：请求构造 + 响应归一化 + 异常翻译。

不发起真实网络请求——OpenAI SDK 被替身接管，只断言参数透传与归一化结果。
"""

from types import SimpleNamespace

import httpx
import openai
import pytest

import llm.openai_client as openai_client_module
from llm.base_client import LLMClient
from utils.expiation.errors import LLMAuthError, LLMError, LLMRequestError, LLMTransientError
from llm.openai_client import OpenAIClient
from schemas.llm import FinishReason, LLMResponse, Usage

_MSG = [{"role": "user", "content": "上月订单量是多少"}]

_HTTP_REQUEST = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")


def make_vendor_error(class_name: str, status_code: int):
    """构造一个真实的 openai SDK 异常，用于验证翻译逻辑。"""
    cls = getattr(openai, class_name)
    return cls(
        f"stub {class_name}",
        response=httpx.Response(status_code, request=_HTTP_REQUEST),
        body=None,
    )



class _FakeCompletions:
    """记录每次 create() 的入参，并返回预设响应或抛出预设异常。"""

    def __init__(self, response, error=None):
        self._response = response
        self._error = error
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return self._response


class _Harness:
    def __init__(self, client, completions, sdk_kwargs):
        self.client = client
        self.completions = completions
        self.sdk_kwargs = sdk_kwargs

    @property
    def last_request(self) -> dict:
        assert self.completions.calls, "chat() 未发起任何请求"
        return self.completions.calls[-1]


@pytest.fixture
def build(monkeypatch):
    """构造一个连通路都被替身的 OpenAIClient。"""

    def _build(response=None, *, error=None, model="gpt-4o-mini", api_key="test-key",
               base_url=None):
        completions = _FakeCompletions(response, error=error)
        sdk_kwargs: dict = {}

        class _FakeOpenAI:
            def __init__(self, **kwargs):
                sdk_kwargs.update(kwargs)
                self.chat = SimpleNamespace(completions=completions)

        monkeypatch.setattr(openai_client_module, "OpenAI", _FakeOpenAI)
        client = OpenAIClient(model=model, api_key=api_key, base_url=base_url)
        return _Harness(client, completions, sdk_kwargs)

    return _build


def make_usage(prompt=10, completion=5, total=15):
    return SimpleNamespace(
        prompt_tokens=prompt, completion_tokens=completion, total_tokens=total
    )


def make_tool_call(call_id="call_1", name="run_sql", arguments='{"sql": "SELECT 1"}'):
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=arguments))


def make_response(*, content=None, tool_calls=None, finish_reason="stop", usage=None,
                  model="gpt-4o-mini", raw=None):
    """伪造一次 OpenAI ChatCompletion 响应，字段与 SDK 返回对象对齐。"""
    payload = {"id": "chatcmpl-stub"} if raw is None else raw
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(content=content, tool_calls=tool_calls),
            )
        ],
        usage=usage,
        model=model,
        model_dump=lambda: payload,
    )


# --------------------------------------------------------------------------
# 抽象契约
# --------------------------------------------------------------------------


def test_llm_client_is_abstract():
    with pytest.raises(TypeError):
        LLMClient()


def test_openai_client_implements_llm_client(build):
    assert isinstance(build(make_response(content="ok")).client, LLMClient)


def test_sdk_constructed_with_api_key_and_base_url(build):
    h = build(make_response(), api_key="sk-test", base_url="https://example.com/v1")
    assert h.sdk_kwargs == {"api_key": "sk-test", "base_url": "https://example.com/v1"}


# --------------------------------------------------------------------------
# 请求构造
# --------------------------------------------------------------------------


def test_chat_sends_model_and_messages(build):
    h = build(make_response(content="ok"))
    h.client.chat(_MSG)
    assert h.last_request["model"] == "gpt-4o-mini"
    assert h.last_request["messages"] == _MSG


def test_chat_omits_tools_when_not_passed(build):
    h = build(make_response(content="ok"))
    h.client.chat(_MSG)
    assert "tools" not in h.last_request


def test_chat_omits_tools_when_none(build):
    h = build(make_response(content="ok"))
    h.client.chat(_MSG, tools=None)
    assert "tools" not in h.last_request


def test_chat_omits_tools_when_empty_list(build):
    """空 tools 会污染请求，部分兼容服务商直接报 400。"""
    h = build(make_response(content="ok"))
    h.client.chat(_MSG, tools=[])
    assert "tools" not in h.last_request


def test_chat_forwards_tools_when_provided(build):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "run_sql",
                "description": "执行只读 SQL",
                "parameters": {"type": "object", "properties": {"sql": {"type": "string"}}},
            },
        }
    ]
    h = build(make_response(content="ok"))
    h.client.chat(_MSG, tools=tools)
    assert h.last_request["tools"] == tools


def test_client_reusable_across_calls(build):
    """agent 循环会反复调用同一个 client。"""
    h = build(make_response(content="ok"))
    h.client.chat(_MSG)
    h.client.chat(_MSG)
    assert len(h.completions.calls) == 2


# --------------------------------------------------------------------------
# 响应归一化：基础字段
# --------------------------------------------------------------------------


def test_chat_returns_llm_response(build):
    h = build(make_response(content="ok"))
    assert isinstance(h.client.chat(_MSG), LLMResponse)


def test_content_preserved(build):
    h = build(make_response(content="销售额为 1200"))
    assert h.client.chat(_MSG).content == "销售额为 1200"


def test_model_and_raw_data_preserved(build):
    raw = {"id": "chatcmpl-1", "object": "chat.completion"}
    h = build(make_response(content="ok", model="deepseek-chat", raw=raw))
    resp = h.client.chat(_MSG)
    assert resp.model == "deepseek-chat"
    assert resp.raw_data == raw


def test_usage_parsed(build):
    h = build(make_response(usage=make_usage(prompt=11, completion=7, total=18)))
    assert h.client.chat(_MSG).usage == Usage(
        prompt_tokens=11, completion_tokens=7, total_tokens=18
    )


def test_missing_usage_degrades_to_zero(build):
    """部分兼容服务商不返回 usage，这里是空值风险点。"""
    h = build(make_response(content="ok", usage=None))
    assert h.client.chat(_MSG).usage == Usage()


# --------------------------------------------------------------------------
# 响应归一化：finish_reason
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw_finish_reason", "expected"),
    [
        ("stop", FinishReason.STOP),
        ("tool_calls", FinishReason.TOOL_CALLS),
        ("length", FinishReason.LENGTH),
        ("content_filter", FinishReason.CONTENT_FILTER),
        ("function_call", FinishReason.UNKNOWN),  # 旧版字段，未纳入映射
        ("", FinishReason.UNKNOWN),
        (None, FinishReason.UNKNOWN),
    ],
)
def test_finish_reason_mapped_to_enum(build, raw_finish_reason, expected):
    h = build(make_response(content="x", finish_reason=raw_finish_reason))
    assert h.client.chat(_MSG).finish_reason is expected


# --------------------------------------------------------------------------
# 响应归一化：tool_calls
# --------------------------------------------------------------------------


def test_absent_tool_calls_yield_none(build):
    h = build(make_response(content="最终结论"))
    resp = h.client.chat(_MSG)
    assert resp.tool_calls is None
    assert resp.has_tool_calls is False


def test_has_tool_calls_true_when_present(build):
    h = build(
        make_response(tool_calls=[make_tool_call()], finish_reason="tool_calls")
    )
    resp = h.client.chat(_MSG)
    assert resp.has_tool_calls is True
    assert resp.content is None


def test_tool_calls_parsed_preserving_order_and_id(build):
    calls = [
        make_tool_call("call_a", "read_schema", "{}"),
        make_tool_call("call_b", "run_sql", '{"sql": "SELECT 1"}'),
    ]
    resp = build(make_response(tool_calls=calls, finish_reason="tool_calls")).client.chat(_MSG)
    assert [c.id for c in resp.tool_calls] == ["call_a", "call_b"]
    assert [c.name for c in resp.tool_calls] == ["read_schema", "run_sql"]


def test_tool_call_arguments_parsed_as_dict(build):
    """arguments 在原始响应里是 JSON 字符串，必须解成 dict。"""
    h = build(make_response(tool_calls=[make_tool_call(arguments='{"sql": "SELECT 1"}')]))
    arguments = h.client.chat(_MSG).tool_calls[0].arguments
    assert isinstance(arguments, dict)
    assert arguments == {"sql": "SELECT 1"}


def test_nested_and_non_ascii_arguments_round_trip(build):
    raw = '{"filters": {"region": "华东"}, "limit": 100}'
    h = build(make_response(tool_calls=[make_tool_call(name="run_sql", arguments=raw)]))
    assert h.client.chat(_MSG).tool_calls[0].arguments == {
        "filters": {"region": "华东"},
        "limit": 100,
    }


@pytest.mark.parametrize("absent_arguments", ["", None])
def test_absent_arguments_are_treated_as_no_params(build, absent_arguments):
    """工具可以没有参数，缺省是合法情况，不应视为错误。"""
    h = build(make_response(tool_calls=[make_tool_call(arguments=absent_arguments)]))
    assert h.client.chat(_MSG).tool_calls[0].arguments == {}


@pytest.mark.parametrize(
    "bad_arguments",
    [
        "not-json",          # 非 JSON
        '{"sql": "SELECT',   # 截断
        "[1, 2]",            # 合法 JSON 但不是对象
        "123",               # 标量
        "null",              # null
        '"just a string"',   # 字符串
    ],
)
def test_malformed_arguments_raise_request_error(build, bad_arguments):
    """模型破坏了工具契约时必须抛出，不能静默退化成空参数。"""
    h = build(make_response(tool_calls=[make_tool_call(arguments=bad_arguments)]))
    with pytest.raises(LLMRequestError):
        h.client.chat(_MSG)


def test_malformed_arguments_error_names_the_tool(build):
    """并行的多个工具调用里，报错必须指明是哪个工具，否则无法定位。"""
    calls = [
        make_tool_call("call_a", "read_schema", "{}"),
        make_tool_call("call_b", "run_sql", "not-json"),
    ]
    h = build(make_response(tool_calls=calls))
    with pytest.raises(LLMRequestError) as exc:
        h.client.chat(_MSG)
    assert "run_sql" in str(exc.value)


# --------------------------------------------------------------------------
# 异常翻译（厂商异常 -> 统一异常）
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "vendor_error",
    [
        make_vendor_error("RateLimitError", 429),
        make_vendor_error("InternalServerError", 500),
        openai.APITimeoutError(request=_HTTP_REQUEST),
        openai.APIConnectionError(request=_HTTP_REQUEST),
    ],
)
def test_transient_vendor_errors_map_to_transient(build, vendor_error):
    h = build(error=vendor_error)
    with pytest.raises(LLMTransientError) as exc:
        h.client.chat(_MSG)
    assert exc.value.retryable is True


@pytest.mark.parametrize(
    "vendor_error",
    [
        make_vendor_error("AuthenticationError", 401),
        make_vendor_error("PermissionDeniedError", 403),
    ],
)
def test_auth_vendor_errors_map_to_auth(build, vendor_error):
    h = build(error=vendor_error)
    with pytest.raises(LLMAuthError) as exc:
        h.client.chat(_MSG)
    assert exc.value.retryable is False


@pytest.mark.parametrize(
    "vendor_error",
    [
        make_vendor_error("BadRequestError", 400),
        make_vendor_error("NotFoundError", 404),
        make_vendor_error("UnprocessableEntityError", 422),
    ],
)
def test_request_vendor_errors_map_to_request(build, vendor_error):
    h = build(error=vendor_error)
    with pytest.raises(LLMRequestError):
        h.client.chat(_MSG)


def test_unknown_vendor_error_falls_back_to_base(build):
    """兜底只到 SDK 的根异常，不吞掉编程错误。"""
    h = build(error=openai.APIError("unexpected", request=_HTTP_REQUEST, body=None))
    with pytest.raises(LLMError) as exc:
        h.client.chat(_MSG)
    assert type(exc.value) is LLMError


def test_translated_error_keeps_original_as_cause(build):
    """翻译不能丢原始异常，否则排查时没有厂商侧的上下文。"""
    original = make_vendor_error("RateLimitError", 429)
    h = build(error=original)
    with pytest.raises(LLMTransientError) as exc:
        h.client.chat(_MSG)
    assert exc.value.__cause__ is original


def test_programming_errors_are_not_swallowed(build):
    """非厂商异常必须原样冒泡，不能被包装成"模型不可用"。"""
    h = build(error=KeyError("boom"))
    with pytest.raises(KeyError):
        h.client.chat(_MSG)


@pytest.mark.parametrize(
    ("error_cls", "expected_type", "expected_retryable"),
    [
        (LLMError, "llm_error", False),
        (LLMTransientError, "llm_transient", True),
        (LLMAuthError, "llm_auth", False),
        (LLMRequestError, "llm_request", False),
    ],
)
def test_error_contract_for_harness_recording(error_cls, expected_type, expected_retryable):
    """error_type / retryable 是 agent 记录 ErrorRecord 与决策恢复的依据。"""
    err = error_cls("boom")
    assert err.error_type == expected_type
    assert err.retryable is expected_retryable
    assert isinstance(err, LLMError)
    assert err.message == "boom"


def test_error_type_tokens_are_unique():
    """error_type 会写进 ErrorRecord，与 SQL 层共用命名空间，不能重复。"""
    tokens = {c.error_type for c in (LLMError, LLMTransientError, LLMAuthError, LLMRequestError)}
    assert len(tokens) == 4

