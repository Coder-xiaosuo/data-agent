import logging
from contextlib import closing, contextmanager
from typing import Any, Iterator

import openai
from openai import OpenAI

from llm.base_client import LLMClient
from llm.streaming import parse_tool_arguments
from schemas.llm import (
    FinishReason,
    LLMResponse,
    LLMStreamChunk,
    ToolCall,
    ToolCallDelta,
    Usage,
)
from utils.exception.errors import (
    LLMAuthError,
    LLMError,
    LLMRequestError,
    LLMTransientError,
)

logger = logging.getLogger(__name__)

_FINISH_REASON_MAP = {
    "stop": FinishReason.STOP,
    "tool_calls": FinishReason.TOOL_CALLS,
    "length": FinishReason.LENGTH,
    "content_filter": FinishReason.CONTENT_FILTER,
}

# 厂商异常 -> 统一异常的归类。分类依据是恢复动作，不是 HTTP 状态码本身。
_TRANSIENT_ERRORS = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
)
_AUTH_ERRORS = (
    openai.AuthenticationError,
    openai.PermissionDeniedError,
)
_REQUEST_ERRORS = (
    openai.BadRequestError,
    openai.NotFoundError,
    openai.UnprocessableEntityError,
)

# SDK 默认超时长达 10 分钟，对 agent 循环来说等于没有保护。
# 超时发生后 SDK 抛 APITimeoutError，会被归到 LLMTransientError，由上层决定是否重试。
DEFAULT_TIMEOUT_SECONDS = 60.0


class OpenAIClient(LLMClient):
    """适配所有 OpenAI 兼容接口的服务商。"""

    def __init__(self, model: str, api_key: str, base_url: str | None = None):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def chat(
        self,
        messages,
        tools=None,
        *,
        tool_choice: str | dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> LLMResponse:
        kwargs = self._build_kwargs(
            messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            timeout=timeout,
        )
        with self._translate_api_errors():
            response = self.client.chat.completions.create(**kwargs)
        return self._to_llm_response(response)

    def chat_stream(
        self,
        messages,
        tools=None,
        *,
        tool_choice: str | dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        timeout: float | None = None,
        include_usage: bool = True,
    ) -> Iterator[LLMStreamChunk]:
        kwargs = self._build_kwargs(
            messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            timeout=timeout,
        )
        kwargs["stream"] = True
        if include_usage:
            #token 消耗开关
            kwargs["stream_options"] = {"include_usage": True}

        #生成器体首次迭代执行
        with self._translate_api_errors():
            stream = self.client.chat.completions.create(**kwargs)
            # closing 保证提前 break 或抛异常时也释放底层连接，不泄漏。
            with closing(stream):
                for raw_chunk in stream:
                    yield self._to_stream_chunk(raw_chunk)

    def _build_kwargs(
        self,
        messages,
        *,
        tools,
        tool_choice,
        temperature,
        max_tokens,
        response_format,
        timeout,
    ) -> dict:
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "timeout": DEFAULT_TIMEOUT_SECONDS if timeout is None else timeout,
        }
        if tools:
            kwargs["tools"] = tools
            # tool_choice 只有在声明了 tools 时才有意义，单独传会被服务商拒绝。
            if tool_choice is not None:
                kwargs["tool_choice"] = tool_choice
        elif tool_choice is not None:
            raise ValueError("tool_choice 需要配合 tools 一起使用，否则请求会被服务商拒绝")
        # 采样与格式参数按需透传：不传就交给服务商默认值，避免用固定值覆盖掉模型自身配置。
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if response_format is not None:
            kwargs["response_format"] = response_format
        return kwargs

    @contextmanager
    def _translate_api_errors(self):
        """调用边界：只在这里把厂商异常翻译成统一异常"""
        try:
            yield
        except _TRANSIENT_ERRORS as exc:
            logger.debug("模型调用瞬时失败：%r", exc)
            raise LLMTransientError(f"模型服务暂时不可用：{exc}") from exc
        except _AUTH_ERRORS as exc:
            logger.debug("模型鉴权失败：%r", exc)
            raise LLMAuthError(f"模型鉴权失败：{exc}") from exc
        except _REQUEST_ERRORS as exc:
            logger.debug("模型拒绝请求：%r", exc)
            raise LLMRequestError(f"模型拒绝了该请求：{exc}") from exc
        except openai.OpenAIError as exc:
            logger.debug("模型调用失败：%r", exc)
            raise LLMError(f"模型调用失败：{exc}") from exc

    def _to_llm_response(self, response) -> LLMResponse:
        choice = response.choices[0]
        message = choice.message

        tool_calls = None
        if message.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=parse_tool_arguments(
                        tc.function.arguments, tool_name=tc.function.name
                    ),
                )
                for tc in message.tool_calls
            ]

        return LLMResponse(
            content=message.content,
            tool_calls=tool_calls,
            finish_reason=_FINISH_REASON_MAP.get(
                choice.finish_reason, FinishReason.UNKNOWN
            ),
            usage=self._parse_usage(response.usage),
            model=response.model,
            raw_data=response.model_dump(),
        )

    def _to_stream_chunk(self, chunk) -> LLMStreamChunk:
        # 开启 include_usage 后，末尾会有一个只带 usage、choices 为空的收尾分片。
        choice = chunk.choices[0] if chunk.choices else None

        content_delta = None
        tool_call_deltas: list[ToolCallDelta] = []
        finish_reason = None
        if choice is not None:
            delta = choice.delta
            content_delta = delta.content
            if choice.finish_reason is not None:
                finish_reason = _FINISH_REASON_MAP.get(
                    choice.finish_reason, FinishReason.UNKNOWN
                )
            if delta.tool_calls:
                tool_call_deltas = [
                    ToolCallDelta(
                        index=tc.index,
                        id=tc.id,
                        name=tc.function.name if tc.function else None,
                        arguments_delta=tc.function.arguments if tc.function else None,
                    )
                    for tc in delta.tool_calls
                ]

        return LLMStreamChunk(
            content_delta=content_delta,
            tool_call_deltas=tool_call_deltas,
            finish_reason=finish_reason,
            usage=self._parse_usage(chunk.usage) if chunk.usage is not None else None,
            model=chunk.model or "",
            raw_data=chunk.model_dump(),
        )

    @staticmethod
    def _parse_usage(usage) -> Usage:
        if usage is None:
            return Usage()
        return Usage(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
        )
