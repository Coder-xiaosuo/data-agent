"""流式响应的聚合。

把一串 LLMStreamChunk 还原成与 chat() 同形的 LLMResponse，
让 agent 层不必自己处理工具调用分片的拼接——那是最容易写错的地方。
"""

import json
from typing import Iterable

from schemas.llm import FinishReason, LLMResponse, LLMStreamChunk, ToolCall, Usage
from utils.exception.errors import LLMRequestError


def parse_tool_arguments(raw: str | None, *, tool_name: str) -> dict:
    """把工具参数从 JSON 字符串解析为 dict。

    非流式传完整 JSON，流式传各分片拼接后的 JSON，共用同一套校验。

    缺省（None / 空串）视为该工具无参数，是合法情况；
    但解析失败说明模型破坏了工具契约，必须抛出让上层感知，
    否则空参数会被当成"模型真的没给参数"而静默误判。
    """
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMRequestError(
            f"工具 {tool_name} 的参数不是合法 JSON：{raw!r}"
        ) from exc
    if not isinstance(parsed, dict):
        raise LLMRequestError(
            f"工具 {tool_name} 的参数不是 JSON 对象：{raw!r}"
        )
    return parsed


def aggregate_stream(chunks: Iterable[LLMStreamChunk]) -> LLMResponse:
    """把流式分片聚合为一次完整响应。

    聚合结果与对同一请求调用 chat() 的返回一致，便于 agent 层统一处理：
    流式只影响"怎么展示"，不影响"怎么决策"。

    工具调用按 index 归类：id / name 取首个出现的分片，arguments 片段按到达顺序拼接。

    Raises:
        LLMRequestError: 拼接后的工具参数不是合法 JSON 对象，或分片缺少 id / name。
    """
    content_parts: list[str] = []
    # index -> {id, name, arguments 片段}
    tool_call_parts: dict[int, dict] = {}
    finish_reason = FinishReason.UNKNOWN
    usage = Usage()
    model = ""

    for chunk in chunks:
        if chunk.content_delta:
            content_parts.append(chunk.content_delta)

        for delta in chunk.tool_call_deltas:
            entry = tool_call_parts.setdefault(
                delta.index, {"id": None, "name": None, "arguments": []}
            )
            if delta.id:
                entry["id"] = delta.id
            if delta.name:
                entry["name"] = delta.name
            if delta.arguments_delta:
                entry["arguments"].append(delta.arguments_delta)

        if chunk.finish_reason is not None:
            finish_reason = chunk.finish_reason
        if chunk.usage is not None:
            usage = chunk.usage
        if chunk.model:
            model = chunk.model

    return LLMResponse(
        # 无内容片段时归一化为 None，与 chat() 在工具调用轮次的返回值保持一致。
        content="".join(content_parts) or None,
        tool_calls=_build_tool_calls(tool_call_parts),
        finish_reason=finish_reason,
        usage=usage,
        model=model,
        # 流式不保留原始分片：逐片留存会随输出线性增长，与流式的省内存初衷相悖。
        raw_data=None,
    )


def _build_tool_calls(tool_call_parts: dict[int, dict]) -> list[ToolCall] | None:
    if not tool_call_parts:
        return None

    tool_calls = []
    for index in sorted(tool_call_parts):
        entry = tool_call_parts[index]
        name = entry["name"]
        if not entry["id"] or not name:
            raise LLMRequestError(
                f"工具调用分片不完整：index={index} 缺少 id 或 name，无法拼回完整调用"
            )
        tool_calls.append(
            ToolCall(
                id=entry["id"],
                name=name,
                arguments=parse_tool_arguments(
                    "".join(entry["arguments"]), tool_name=name
                ),
            )
        )
    return tool_calls
