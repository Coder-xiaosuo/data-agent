"""aggregate_stream 的单元测试：把流式分片还原成完整响应。

这一层的价值就是"流式结果与一次性结果一致"，所以测试重心在
工具调用分片的拼接、以及最终产物与 chat() 返回形态的等价性。
"""

import pytest

from llm.streaming import aggregate_stream, parse_tool_arguments
from schemas.llm import (
    FinishReason,
    LLMResponse,
    LLMStreamChunk,
    ToolCall,
    ToolCallDelta,
    Usage,
)
from utils.exception.errors import LLMRequestError


def chunk(*, content=None, deltas=(), finish_reason=None, usage=None, model="gpt-4o-mini"):
    return LLMStreamChunk(
        content_delta=content,
        tool_call_deltas=list(deltas),
        finish_reason=finish_reason,
        usage=usage,
        model=model,
    )


def delta(index=0, call_id=None, name=None, arguments=None):
    return ToolCallDelta(index=index, id=call_id, name=name, arguments_delta=arguments)


# --------------------------------------------------------------------------
# 文本内容
# --------------------------------------------------------------------------


def test_returns_llm_response():
    assert isinstance(aggregate_stream([chunk(content="a")]), LLMResponse)


def test_content_fragments_concatenated():
    resp = aggregate_stream([
        chunk(content="上月"),
        chunk(content="订单量"),
        chunk(content="为 1200"),
        chunk(finish_reason=FinishReason.STOP),
    ])
    assert resp.content == "上月订单量为 1200"


def test_relative_order_preserved():
    resp = aggregate_stream([chunk(content=str(i)) for i in range(10)])
    assert resp.content == "0123456789"


def test_empty_stream_yields_empty_response():
    resp = aggregate_stream([])
    assert resp.content is None
    assert resp.tool_calls is None
    assert resp.finish_reason is FinishReason.UNKNOWN


def test_no_content_fragments_normalizes_to_none():
    """只有工具调用时 content 应为 None，与 chat() 在工具轮次的返回一致。"""
    resp = aggregate_stream([
        chunk(deltas=[delta(0, "call_1", "run_sql", "{}")]),
        chunk(finish_reason=FinishReason.TOOL_CALLS),
    ])
    assert resp.content is None


def test_empty_string_content_fragments_normalize_to_none():
    """空串分片不能被拼成空字符串，否则与"无内容"的语义混淆。"""
    resp = aggregate_stream([chunk(content=""), chunk(content=None)])
    assert resp.content is None


def test_whitespace_content_is_kept():
    """空白是模型真实输出，不能当空内容丢掉。"""
    assert aggregate_stream([chunk(content=" ")]).content == " "


# --------------------------------------------------------------------------
# 结束原因与用量
# --------------------------------------------------------------------------


def test_finish_reason_taken_from_stream():
    resp = aggregate_stream([chunk(content="x"), chunk(finish_reason=FinishReason.STOP)])
    assert resp.finish_reason is FinishReason.STOP


def test_intermediate_none_finish_reason_does_not_override():
    resp = aggregate_stream([
        chunk(finish_reason=FinishReason.LENGTH),
        chunk(content="x"),
    ])
    assert resp.finish_reason is FinishReason.LENGTH


def test_last_finish_reason_wins():
    resp = aggregate_stream([
        chunk(finish_reason=FinishReason.LENGTH),
        chunk(finish_reason=FinishReason.STOP),
    ])
    assert resp.finish_reason is FinishReason.STOP


def test_usage_taken_from_trailing_chunk():
    resp = aggregate_stream([
        chunk(content="ok"),
        chunk(usage=Usage(prompt_tokens=3, completion_tokens=4, total_tokens=7)),
    ])
    assert resp.usage == Usage(prompt_tokens=3, completion_tokens=4, total_tokens=7)


def test_missing_usage_defaults_to_zero():
    """未开启 include_usage 时没有用量分片，应给零值而非 None。"""
    resp = aggregate_stream([chunk(content="ok")])
    assert resp.usage == Usage()


def test_model_taken_from_stream():
    assert aggregate_stream([chunk(content="x", model="deepseek-flash")]).model == "deepseek-flash"


# --------------------------------------------------------------------------
# 工具调用分片拼接
# --------------------------------------------------------------------------


def test_single_tool_call_arguments_reassembled():
    """首个分片给 id 与 name，后续分片只给 arguments 碎片。"""
    resp = aggregate_stream([
        chunk(deltas=[delta(0, "call_1", "run_sql", '{"sql"')]),
        chunk(deltas=[delta(0, arguments=': "SELECT 1"}')]),
        chunk(finish_reason=FinishReason.TOOL_CALLS),
    ])
    assert resp.tool_calls == [
        ToolCall(id="call_1", name="run_sql", arguments={"sql": "SELECT 1"})
    ]
    assert resp.has_tool_calls is True


def test_many_argument_fragments_reassembled():
    fragments = ['{"sq', 'l": "SELECT', ' COUNT(*) ', 'FROM orders"}']
    chunks = [chunk(deltas=[delta(0, "call_1", "run_sql", fragments[0])])]
    chunks += [chunk(deltas=[delta(0, arguments=f)]) for f in fragments[1:]]
    resp = aggregate_stream(chunks)
    assert resp.tool_calls[0].arguments == {"sql": "SELECT COUNT(*) FROM orders"}


def test_parallel_tool_calls_grouped_by_index():
    """并行工具调用的分片会交错到达，必须按 index 归类而不是按到达顺序。"""
    resp = aggregate_stream([
        chunk(deltas=[
            delta(0, "call_a", "read_schema", '{"table"'),
            delta(1, "call_b", "run_sql", '{"sql"'),
        ]),
        chunk(deltas=[
            delta(0, arguments=': "orders"}'),
            delta(1, arguments=': "SELECT 1"}'),
        ]),
        chunk(finish_reason=FinishReason.TOOL_CALLS),
    ])
    assert [c.id for c in resp.tool_calls] == ["call_a", "call_b"]
    assert resp.tool_calls[0].arguments == {"table": "orders"}
    assert resp.tool_calls[1].arguments == {"sql": "SELECT 1"}


def test_tool_calls_sorted_by_index_not_arrival_order():
    resp = aggregate_stream([
        chunk(deltas=[delta(1, "call_b", "run_sql", "{}")]),
        chunk(deltas=[delta(0, "call_a", "read_schema", "{}")]),
    ])
    assert [c.id for c in resp.tool_calls] == ["call_a", "call_b"]


def test_tool_call_with_no_arguments_is_legal():
    resp = aggregate_stream([chunk(deltas=[delta(0, "call_1", "list_tables", None)])])
    assert resp.tool_calls[0].arguments == {}


def test_duplicate_id_fragments_do_not_duplicate_call():
    """有的服务商每个分片都重复带 id，不能因此拆成多个调用。"""
    resp = aggregate_stream([
        chunk(deltas=[delta(0, "call_1", "run_sql", '{"a"')]),
        chunk(deltas=[delta(0, "call_1", "run_sql", ': 1}')]),
    ])
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].arguments == {"a": 1}


def test_malformed_reassembled_arguments_raise():
    """切片拼接后仍非法 JSON，说明模型破坏了契约，必须抛出让验证层感知。"""
    resp_chunks = [
        chunk(deltas=[delta(0, "call_1", "run_sql", "not-json")]),
    ]
    with pytest.raises(LLMRequestError):
        aggregate_stream(resp_chunks)


def test_incomplete_tool_call_raises():
    """只有 arguments 分片却始终没有 id / name，无法拼回调用。"""
    with pytest.raises(LLMRequestError):
        aggregate_stream([chunk(deltas=[delta(0, arguments='{"sql": "SELECT 1"}')])])


def test_arguments_not_object_raises():
    with pytest.raises(LLMRequestError):
        aggregate_stream([chunk(deltas=[delta(0, "call_1", "run_sql", "[1, 2]")])])


def test_non_ascii_and_nested_arguments_round_trip():
    resp = aggregate_stream([
        chunk(deltas=[delta(0, "call_1", "run_sql", '{"filters": {"region": "华东"}}')]),
    ])
    assert resp.tool_calls[0].arguments == {"filters": {"region": "华东"}}


# --------------------------------------------------------------------------
# parse_tool_arguments（非流式与流式共用的解析入口）
# --------------------------------------------------------------------------


def test_parse_accepts_valid_object():
    assert parse_tool_arguments('{"sql": "SELECT 1"}', tool_name="run_sql") == {
        "sql": "SELECT 1"
    }


@pytest.mark.parametrize("absent", [None, ""])
def test_parse_treats_absent_as_no_params(absent):
    assert parse_tool_arguments(absent, tool_name="run_sql") == {}


@pytest.mark.parametrize("bad", ["not-json", '{"sql": "SELECT', "[1]", "123", '"str"'])
def test_parse_rejects_non_object_or_malformed(bad):
    with pytest.raises(LLMRequestError):
        parse_tool_arguments(bad, tool_name="run_sql")


def test_parse_error_names_the_tool():
    with pytest.raises(LLMRequestError) as exc:
        parse_tool_arguments("bad", tool_name="run_sql")
    assert "run_sql" in str(exc.value)
