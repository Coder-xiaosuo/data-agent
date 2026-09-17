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