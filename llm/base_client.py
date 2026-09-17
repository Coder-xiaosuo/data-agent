
from abc import ABC, abstractmethod
from schemas.llm import LLMResponse


class LLMClient(ABC):
    """模型适配层契约。

    实现方必须把厂商 SDK 的异常翻译成 llm.errors 里的统一异常后再抛出，
    是否重试等恢复策略由调用方决定。
    """

    @abstractmethod
    def chat(
            self,
            messages: list[dict],
            tools: list[dict] | None = None,
    ) -> LLMResponse:
        ...
