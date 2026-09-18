"""OpenAIClient 的单元测试：请求构造 + 响应归一化 + 异常翻译。

不发起真实网络请求——OpenAI SDK 被替身接管，只断言参数透传与归一化结果。
"""

from types import SimpleNamespace

import openai
import pytest

import llm.openai_client as openai_client_module
from llm.base_client import LLMClient
from llm.openai_client import DEFAULT_TIMEOUT_SECONDS, OpenAIClient
from llm.streaming import aggregate_stream
from schemas.llm import FinishReason, LLMResponse, LLMStreamChunk, Usage
from utils.exception.errors import (
    LLMAuthError,
    LLMError,
    LLMRequestError,
    LLMTransientError,
)

_MSG = [{"role": "user", "content": "上月订单量是多少"}]


def _fake_http_response(status_code: int):
    """SDK 的 APIStatusError 只对 response 做 duck typing，用替身即可。

    这样测试不必绑定具体 HTTP 库（openai 2.x 用 httpx，3.x 已换成 httpx2）。
    """
    return SimpleNamespace(
        status_code=status_code,
        headers={},
        request=None,
        text="",
        json=lambda: {},
    )


def make_vendor_error(class_name: str, status_code: int):
    """构造一个真实的 openai SDK 异常，用于验证翻译逻辑。"""
    cls = getattr(openai, class_name)
    return cls(f"stub {class_name}", response=_fake_http_response(status_code), body=None)



class _FakeStream:
    """替身流：可迭代、可 close，并记录是否被关闭。"""

    def __init__(self, chunks, error=None):
        self._chunks = list(chunks)
        self._error = error
        self.closed = False
        self.iterated = 0

    def __iter__(self):
        for chunk in self._chunks:
            self.iterated += 1
            yield chunk
        if self._error is not None:
            raise self._error

    def close(self):
        self.closed = True


class _FakeCompletions:
    """记录每次 create() 的入参，并返回预设响应/流，或抛出预设异常。"""

    def __init__(self, response, error=None, stream_chunks=None, stream_error=None):
        self._response = response
        self._error = error
        self._stream_chunks = stream_chunks or []
        self._stream_error = stream_error
        self.calls: list[dict] = []
        self.last_stream: _FakeStream | None = None

    def create(self, **kwargs):
        self.calls.append(kwargs)
        # error 表示创建请求时就失败（非流式与流式都一样），stream_error 表示迭代中途失败。
        if self._error is not None:
            raise self._error
        if kwargs.get("stream"):
            self.last_stream = _FakeStream(self._stream_chunks, error=self._stream_error)
            return self.last_stream
        return self._response


class _Harness:
    def __init__(self, client, completions, sdk_kwargs):
        self.client = client
        self.completions = completions
        self.sdk_kwargs = sdk_kwargs

    @property
    def last_request(self) -> dict:
        assert self.completions.calls, "尚未发起任何请求"
        return self.completions.calls[-1]

    @property
    def last_stream(self) -> _FakeStream:
        assert self.completions.last_stream is not None, "未创建流"
        return self.completions.last_stream


@pytest.fixture
def build(monkeypatch):
    """构造一个连通路都被替身的 OpenAIClient。"""

    def _build(response=None, *, error=None, stream_chunks=None, stream_error=None,
               model="gpt-4o-mini", api_key="test-key", base_url=None):
        completions = _FakeCompletions(
            response, error=error, stream_chunks=stream_chunks, stream_error=stream_error
        )
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


def make_stream_chunk(*, content=None, tool_call_deltas=None, finish_reason=None,
                      usage=None, model="gpt-4o-mini", choices=None):
    """伪造一个流式分片，字段与 SDK 的 ChatCompletionChunk 对齐。

    choices=None 表示"只带 usage 的收尾分片"，SDK 在这种情况下返回空 choices 列表。
    """
    if choices is None:
        choices = [
            SimpleNamespace(
                finish_reason=finish_reason,
                delta=SimpleNamespace(content=content, tool_calls=tool_call_deltas),
            )
        ]
    return SimpleNamespace(
        choices=choices,
        usage=usage,
        model=model,
        model_dump=lambda: {"id": "chunk-stub", "model": model},
    )


def make_tool_call_delta(index=0, call_id=None, name=None, arguments=None):
    fn = None if (name is None and arguments is None) else SimpleNamespace(
        name=name, arguments=arguments
    )
    return SimpleNamespace(index=index, id=call_id, function=fn)


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
# 请求构造：超时与采样参数
# --------------------------------------------------------------------------


def test_timeout_defaults_to_module_default(build):
    """SDK 默认超时长达 10 分钟，必须覆盖成受控值，否则循环里等于没有保护。"""
    h = build(make_response(content="ok"))
    h.client.chat(_MSG)
    assert h.last_request["timeout"] == DEFAULT_TIMEOUT_SECONDS


def test_explicit_timeout_overrides_default(build):
    h = build(make_response(content="ok"))
    h.client.chat(_MSG, timeout=5.0)
    assert h.last_request["timeout"] == 5.0


def test_sampling_params_omitted_when_not_given(build):
    """不传就交给服务商默认值，不能用固定值覆盖模型自身推荐配置。"""
    h = build(make_response(content="ok"))
    h.client.chat(_MSG)
    assert "temperature" not in h.last_request
    assert "max_tokens" not in h.last_request


def test_temperature_passed_through(build):
    h = build(make_response(content="ok"))
    h.client.chat(_MSG, temperature=0.2)
    assert h.last_request["temperature"] == 0.2


def test_max_tokens_passed_through(build):
    h = build(make_response(content="ok"))
    h.client.chat(_MSG, max_tokens=512)
    assert h.last_request["max_tokens"] == 512


def test_zero_temperature_is_forwarded(build):
    """确定性输出要求 temperature=0，0.0 是假值，实现必须用 is not None 判断。"""
    h = build(make_response(content="ok"))
    h.client.chat(_MSG, temperature=0.0)
    assert h.last_request["temperature"] == 0.0


def test_zero_max_tokens_is_forwarded(build):
    """max_tokens=0 同样是假值，不能被当作未设置丢弃。"""
    h = build(make_response(content="ok"))
    h.client.chat(_MSG, max_tokens=0)
    assert h.last_request["max_tokens"] == 0


def test_params_combine_with_tools(build):
    tools = [{"type": "function", "function": {"name": "run_sql", "parameters": {}}}]
    h = build(make_response(content="ok"))
    h.client.chat(_MSG, tools=tools, temperature=0.1, max_tokens=256, timeout=30.0)
    assert h.last_request["tools"] == tools
    assert h.last_request["temperature"] == 0.1
    assert h.last_request["max_tokens"] == 256
    assert h.last_request["timeout"] == 30.0


def test_timeout_error_is_translated_to_transient(build):
    """超时保护触发后要落到可重试类别，让循环能决定重试。"""
    h = build(error=openai.APITimeoutError(request=None))
    with pytest.raises(LLMTransientError):
        h.client.chat(_MSG, timeout=1.0)



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
        openai.APITimeoutError(request=None),
        openai.APIConnectionError(request=None),
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
    h = build(error=openai.APIError("unexpected", request=None, body=None))
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


# --------------------------------------------------------------------------
# chat：tool_choice / response_format
# --------------------------------------------------------------------------


_TOOLS = [{"type": "function", "function": {"name": "run_sql", "parameters": {}}}]


def test_tool_choice_omitted_when_not_given(build):
    h = build(make_response(content="ok"))
    h.client.chat(_MSG, tools=_TOOLS)
    assert "tool_choice" not in h.last_request


def test_tool_choice_passed_through(build):
    h = build(make_response(content="ok"))
    h.client.chat(_MSG, tools=_TOOLS, tool_choice="required")
    assert h.last_request["tool_choice"] == "required"


def test_tool_choice_accepts_function_object(build):
    """指定具体函数时 tool_choice 是对象，不能被当成字符串处理。"""
    choice = {"type": "function", "function": {"name": "run_sql"}}
    h = build(make_response(content="ok"))
    h.client.chat(_MSG, tools=_TOOLS, tool_choice=choice)
    assert h.last_request["tool_choice"] == choice


def test_tool_choice_without_tools_raises(build):
    """tool_choice 脱离 tools 会被服务商拒绝，本地就该给出明确错误。"""
    h = build(make_response(content="ok"))
    with pytest.raises(ValueError):
        h.client.chat(_MSG, tool_choice="required")


def test_response_format_omitted_when_not_given(build):
    h = build(make_response(content="ok"))
    h.client.chat(_MSG)
    assert "response_format" not in h.last_request


def test_response_format_passed_through(build):
    h = build(make_response(content="ok"))
    h.client.chat(_MSG, response_format={"type": "json_object"})
    assert h.last_request["response_format"] == {"type": "json_object"}


# --------------------------------------------------------------------------
# chat_stream：请求构造
# --------------------------------------------------------------------------


def test_chat_stream_sets_stream_flag(build):
    h = build(stream_chunks=[make_stream_chunk(content="hi")])
    list(h.client.chat_stream(_MSG))
    assert h.last_request["stream"] is True


def test_chat_stream_requests_usage_by_default(build):
    """不请求 usage 的话流式拿不到 token 消耗，AgentState 就记不了账。"""
    h = build(stream_chunks=[make_stream_chunk(content="hi")])
    list(h.client.chat_stream(_MSG))
    assert h.last_request["stream_options"] == {"include_usage": True}


def test_chat_stream_can_disable_usage(build):
    h = build(stream_chunks=[make_stream_chunk(content="hi")])
    list(h.client.chat_stream(_MSG, include_usage=False))
    assert "stream_options" not in h.last_request


def test_chat_stream_shares_params_with_chat(build):
    h = build(stream_chunks=[make_stream_chunk(content="hi")])
    list(h.client.chat_stream(
        _MSG,
        tools=_TOOLS,
        tool_choice="auto",
        temperature=0.3,
        max_tokens=64,
        response_format={"type": "json_object"},
        timeout=12.0,
    ))
    req = h.last_request
    assert req["model"] == "gpt-4o-mini"
    assert req["tools"] == _TOOLS
    assert req["tool_choice"] == "auto"
    assert req["temperature"] == 0.3
    assert req["max_tokens"] == 64
    assert req["response_format"] == {"type": "json_object"}
    assert req["timeout"] == 12.0


def test_chat_stream_applies_default_timeout(build):
    h = build(stream_chunks=[make_stream_chunk(content="hi")])
    list(h.client.chat_stream(_MSG))
    assert h.last_request["timeout"] == DEFAULT_TIMEOUT_SECONDS


def test_chat_stream_tool_choice_without_tools_raises(build):
    h = build(stream_chunks=[make_stream_chunk(content="hi")])
    with pytest.raises(ValueError):
        list(h.client.chat_stream(_MSG, tool_choice="required"))


# --------------------------------------------------------------------------
# chat_stream：分片归一化
# --------------------------------------------------------------------------


def test_chat_stream_is_lazy(build):
    """生成器在首次迭代前不应发出请求，否则调用方无法先做参数校验。"""
    h = build(stream_chunks=[make_stream_chunk(content="hi")])
    h.client.chat_stream(_MSG)
    assert h.completions.calls == []


def test_chat_stream_yields_stream_chunks(build):
    h = build(stream_chunks=[
        make_stream_chunk(content="上"),
        make_stream_chunk(content="月"),
        make_stream_chunk(content=None, finish_reason="stop"),
    ])
    chunks = list(h.client.chat_stream(_MSG))
    assert [c.content_delta for c in chunks] == ["上", "月", None]
    assert all(isinstance(c, LLMStreamChunk) for c in chunks)


def test_chat_stream_finish_reason_only_on_last_chunk(build):
    """中间分片没有 finish_reason，必须保持 None，不能用 UNKNOWN 冒充。"""
    h = build(stream_chunks=[
        make_stream_chunk(content="上"),
        make_stream_chunk(content=None, finish_reason="stop"),
    ])
    chunks = list(h.client.chat_stream(_MSG))
    assert [c.finish_reason for c in chunks] == [None, FinishReason.STOP]


def test_chat_stream_maps_unknown_finish_reason(build):
    h = build(stream_chunks=[make_stream_chunk(finish_reason="weird")])
    assert list(h.client.chat_stream(_MSG))[0].finish_reason is FinishReason.UNKNOWN


def test_chat_stream_usage_only_on_final_chunk(build):
    """usage 只出现在末尾的空 choices 收尾分片上，解析时不能假设 choices 非空。"""
    h = build(stream_chunks=[
        make_stream_chunk(content="ok"),
        make_stream_chunk(choices=[], usage=make_usage(prompt=7, completion=2, total=9)),
    ])
    chunks = list(h.client.chat_stream(_MSG))
    assert chunks[0].usage is None
    assert chunks[1].usage == Usage(prompt_tokens=7, completion_tokens=2, total_tokens=9)


def test_chat_stream_parses_tool_call_fragments(build):
    h = build(stream_chunks=[
        make_stream_chunk(tool_call_deltas=[make_tool_call_delta(0, call_id="call_1", name="run_sql", arguments='{"sql"')]),
        make_stream_chunk(tool_call_deltas=[make_tool_call_delta(0, arguments=': "SELECT 1"}')]),
        make_stream_chunk(finish_reason="tool_calls"),
    ])
    chunks = list(h.client.chat_stream(_MSG))
    first = chunks[0].tool_call_deltas[0]
    assert (first.index, first.id, first.name, first.arguments_delta) == (0, "call_1", "run_sql", '{"sql"')
    second = chunks[1].tool_call_deltas[0]
    assert (second.index, second.id, second.name, second.arguments_delta) == (0, None, None, ': "SELECT 1"}')
    assert chunks[1].has_tool_call_deltas is True
    assert chunks[0].has_tool_call_deltas is True


def test_chat_stream_content_delta_absent_for_usage_chunk(build):
    h = build(stream_chunks=[make_stream_chunk(choices=[], usage=make_usage())])
    assert list(h.client.chat_stream(_MSG))[0].content_delta is None


def test_chat_stream_model_recorded(build):
    h = build(stream_chunks=[make_stream_chunk(content="ok", model="deepseek-flash")])
    assert list(h.client.chat_stream(_MSG))[0].model == "deepseek-flash"


def test_chat_stream_closes_connection_on_completion(build):
    h = build(stream_chunks=[make_stream_chunk(content="ok")])
    list(h.client.chat_stream(_MSG))
    assert h.last_stream.closed is True


def test_chat_stream_closes_connection_on_early_break(build):
    """调用方中途 break 也必须释放连接，否则会泄漏。"""
    h = build(stream_chunks=[make_stream_chunk(content="a"), make_stream_chunk(content="b")])
    for chunk in h.client.chat_stream(_MSG):
        break
    assert h.last_stream.closed is True


def test_chat_stream_translates_vendor_error_during_iteration(build):
    h = build(stream_chunks=[], stream_error=make_vendor_error("RateLimitError", 429))
    with pytest.raises(LLMTransientError):
        list(h.client.chat_stream(_MSG))


def test_chat_stream_translates_error_raised_at_creation(build):
    h = build(error=make_vendor_error("AuthenticationError", 401))
    with pytest.raises(LLMAuthError):
        list(h.client.chat_stream(_MSG))


def test_chat_stream_programming_errors_not_swallowed(build):
    h = build(stream_chunks=[], stream_error=KeyError("boom"))
    with pytest.raises(KeyError):
        list(h.client.chat_stream(_MSG))


# --------------------------------------------------------------------------
# 等价性：流式聚合结果必须与一次性调用一致
# --------------------------------------------------------------------------


def test_stream_aggregate_equals_chat_for_text(build):
    """核心不变式：流式只改变怎么展示，不改变怎么决策。"""
    h = build(
        response=make_response(content="上月订单量为 1200", finish_reason="stop",
                                usage=make_usage(9, 3, 12), model="deepseek-flash"),
        stream_chunks=[
            make_stream_chunk(content="上月", model="deepseek-flash"),
            make_stream_chunk(content="订单量为 1200", model="deepseek-flash"),
            make_stream_chunk(finish_reason="stop", model="deepseek-flash"),
            make_stream_chunk(choices=[], usage=make_usage(9, 3, 12), model="deepseek-flash"),
        ],
    )
    streamed = aggregate_stream(h.client.chat_stream(_MSG))
    direct = h.client.chat(_MSG)

    assert streamed.content == direct.content
    assert streamed.finish_reason is direct.finish_reason
    assert streamed.usage == direct.usage
    assert streamed.model == direct.model
    assert streamed.tool_calls == direct.tool_calls


def test_stream_aggregate_equals_chat_for_tool_calls(build):
    """工具调用轮次的等价性——分片拼接的结果必须与一次性返回的 arguments 相同。"""
    h = build(
        response=make_response(
            tool_calls=[make_tool_call("call_1", "run_sql", '{"sql": "SELECT COUNT(*) FROM orders"}')],
            finish_reason="tool_calls",
            usage=make_usage(20, 8, 28),
        ),
        stream_chunks=[
            make_stream_chunk(tool_call_deltas=[make_tool_call_delta(0, call_id="call_1", name="run_sql", arguments='{"sql": "SELECT')]),
            make_stream_chunk(tool_call_deltas=[make_tool_call_delta(0, arguments=' COUNT(*) FROM orders"}')]),
            make_stream_chunk(finish_reason="tool_calls"),
            make_stream_chunk(choices=[], usage=make_usage(20, 8, 28)),
        ],
    )
    streamed = aggregate_stream(h.client.chat_stream(_MSG))
    direct = h.client.chat(_MSG)

    assert streamed.content is None
    assert streamed.finish_reason is FinishReason.TOOL_CALLS
    assert streamed.usage == direct.usage
    assert streamed.tool_calls == direct.tool_calls
    assert streamed.tool_calls[0].arguments == {"sql": "SELECT COUNT(*) FROM orders"}


