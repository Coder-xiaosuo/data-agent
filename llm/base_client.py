
from abc import ABC, abstractmethod
from typing import Any, Iterator

from schemas.llm import LLMResponse, LLMStreamChunk


class LLMClient(ABC):
    """模型适配层契约。

    实现方必须把厂商 SDK 的异常翻译成 utils.exception.errors 里的统一异常后再抛出，
    调用方（agent / harness）不需要、也不应该 import 任何厂商 SDK 的异常类型。
    是否重试等恢复策略由调用方决定。

    流式与非流式共用同一套参数与归一化结果：流式只改变"怎么展示"，
    不改变"怎么决策"——chat_stream() 的分片经 aggregate_stream() 聚合后，
    与同一请求调用 chat() 的返回一致。
    """

    @abstractmethod
    def chat(
            self,
            messages: list[dict],
            tools: list[dict] | None = None,
            *,
            tool_choice: str | dict[str, Any] | None = None,
            temperature: float | None = None,
            max_tokens: int | None = None,
            response_format: dict[str, Any] | None = None,
            timeout: float | None = None,
    ) -> LLMResponse:
        """发起一次对话补全，等待完整响应。

        Args:
            messages: OpenAI 格式的消息列表。
            tools: 工具 schema 列表；为空或 None 时不携带 tools 字段。
            tool_choice: 工具调用策略（auto / none / required 或指定函数对象）；
                必须与 tools 同时使用。
            temperature: 采样温度；None 表示不传，用服务商默认值。
            max_tokens: 生成上限；None 表示不传，用服务商默认值。
            response_format: 输出格式约束，如 {"type": "json_object"}。
            timeout: 单次请求超时秒数；None 表示用实现方的默认值。

        Raises:
            LLMTransientError: 限流、超时、连接中断、服务端 5xx，可重试。
            LLMAuthError: 鉴权或权限失败，不可重试。
            LLMRequestError: 请求或模型配置有误、工具参数非法，不可重试。
            LLMError: 其余调用失败。
        """
        ...

    @abstractmethod
    def chat_stream(
            self,
            messages: list[dict],
            tools: list[dict] | None = None,
            *,
            tool_choice: str | dict[str, Any] | None = None,
            temperature: float | None = None,
            max_tokens: int | None = None,
            response_format: dict[str, Any] | None = None,
            timeout: float | None = None,
            include_usage: bool = True,
    ) -> Iterator[LLMStreamChunk]:
        """发起一次流式对话补全，逐个产出增量分片。

        参数含义与 chat() 一致。返回的是惰性生成器：请求在首次迭代时才发出，
        因此异常也在迭代过程中抛出，调用方需要把迭代本身放在 try 内。

        Args:
            include_usage: 是否请求服务商在末尾分片返回 token 用量；
                关闭后 LLMStreamChunk.usage 将始终为 None。

        Raises:
            与 chat() 相同的统一异常。
        """
        ...
