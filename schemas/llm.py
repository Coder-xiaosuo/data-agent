# schemas/llm.py

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


# 能力枚举
class ModelCapability(str, Enum):
    TOOL_CALLING = "tool_calling"
    STREAMING = "streaming"
    JSON_MODE = "json_mode"


class FinishReason(str, Enum):
    """归一化后的结束原因。不同服务商的原始值映射到这里。"""
    STOP = "stop"                    # 正常结束
    TOOL_CALLS = "tool_calls"        # 模型请求调用工具
    LENGTH = "length"                # 达到 max_tokens
    CONTENT_FILTER = "content_filter"  # 内容被过滤
    UNKNOWN = "unknown"              # 无法识别的原始值


# 工具调用
class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


#Token 消耗
class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


#统一响应
class LLMResponse(BaseModel):
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    finish_reason: FinishReason = FinishReason.UNKNOWN
    usage: Usage = Field(default_factory=Usage)
    model: str = ""                  # 实际使用的模型名，便于追踪
    raw_data: dict | None = None          # 原始响应，调试用，不参与业务逻辑

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


# 流式：工具调用的单个分片
class ToolCallDelta(BaseModel):
    """工具调用的增量分片。

    流式响应里一次工具调用的信息是拆开到达的：首个分片带 id 与 name，
    后续分片只带 arguments 的字符串碎片。`index` 是拼回完整调用的唯一依据。
    """
    index: int
    id: str | None = None
    name: str | None = None
    arguments_delta: str | None = None   # JSON 字符串的片段，不是完整 JSON


# 流式：单个增量分片
class LLMStreamChunk(BaseModel):
    """一次流式响应的增量分片。

    注意 finish_reason 与 usage 只在特定分片上出现：
    - finish_reason 仅出现在最后一个内容分片
    - usage 仅在开启 include_usage 时的末尾空分片（其 choices 为空列表）
    """
    content_delta: str | None = None
    tool_call_deltas: list[ToolCallDelta] = Field(default_factory=list)
    finish_reason: FinishReason | None = None   # 未到达结束分片时为 None，区别于 UNKNOWN
    usage: Usage | None = None                  # 未携带用量时为 None，区别于全零
    model: str = ""
    raw_data: dict | None = None

    @property
    def has_tool_call_deltas(self) -> bool:
        return bool(self.tool_call_deltas)
