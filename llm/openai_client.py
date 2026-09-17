import json

import openai
from openai import OpenAI

from llm.base_client import LLMClient
from utils.expiation.errors import (
    LLMAuthError,
    LLMError,
    LLMRequestError,
    LLMTransientError,
)
from schemas.llm import FinishReason, LLMResponse, ToolCall, Usage

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


class OpenAIClient(LLMClient):
    """适配所有 OpenAI 兼容接口的服务商。"""

    def __init__(self, model: str, api_key: str, base_url: str | None = None):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def chat(self, messages, tools=None) -> LLMResponse:
        kwargs = {"model": self.model, "messages": messages}
        if tools:
            kwargs["tools"] = tools

        response = self._create_completion(kwargs)
        return self._to_llm_response(response)

    def _create_completion(self, kwargs: dict):
        """调用边界：只在这里把厂商异常翻译成统一异常。

        这层只做翻译，不做重试——是否重试属于 agent 循环的策略。
        用 `raise ... from exc` 保留原始异常链，便于排查。
        """
        try:
            return self.client.chat.completions.create(**kwargs)
        except _TRANSIENT_ERRORS as exc:
            raise LLMTransientError(f"模型服务暂时不可用：{exc}") from exc
        except _AUTH_ERRORS as exc:
            raise LLMAuthError(f"模型鉴权失败：{exc}") from exc
        except _REQUEST_ERRORS as exc:
            raise LLMRequestError(f"模型拒绝了该请求：{exc}") from exc
        except openai.OpenAIError as exc:
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
                    arguments=self._parse_arguments(
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

    @staticmethod
    def _parse_arguments(raw: str | None, *, tool_name: str) -> dict:
        """模型返回的 arguments 是 JSON 字符串。

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

    @staticmethod
    def _parse_usage(usage) -> Usage:
        if usage is None:
            return Usage()
        return Usage(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
        )
