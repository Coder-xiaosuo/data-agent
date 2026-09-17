"""LLM 层统一异常。

这一层是唯一知道厂商 SDK 细节的地方，因此把厂商异常翻译成这里的统一类型，
上层（agent / harness）只依赖这些类型，不依赖 openai SDK。

分类标准只有一条：**是否对应不同的恢复动作**。
- LLMTransientError：可重试
- LLMAuthError：配置问题，不可重试，快速失败
- LLMRequestError：请求或模型配置有误，不可重试

`error_type` 用于填充 AgentState 里的 ErrorRecord.error_type；
带 `llm_` 前缀是为了和 SQL 层的 error_type（syntax_error 等）共用同一命名空间时不冲突。
"""


class LLMError(Exception):
    """LLM 层统一异常基类。"""

    error_type: str = "llm_error"
    retryable: bool = False

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class LLMTransientError(LLMError):
    """限流、超时、连接中断、服务端 5xx。可重试。"""

    error_type = "llm_transient"
    retryable = True


class LLMAuthError(LLMError):
    """API key 无效、权限不足。配置问题，重试无用。"""

    error_type = "llm_auth"
    retryable = False


class LLMRequestError(LLMError):
    """请求被拒：模型名不存在、参数非法、工具参数不是合法 JSON 等。"""

    error_type = "llm_request"
    retryable = False
