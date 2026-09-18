import importlib.util
import openai

print("openai:", openai.__version__)
for name in [
    "OpenAIError", "APIError", "APIStatusError",
    "RateLimitError", "APITimeoutError", "APIConnectionError", "InternalServerError",
    "AuthenticationError", "PermissionDeniedError",
    "BadRequestError", "NotFoundError", "UnprocessableEntityError",
]:
    print(f"{name:26s}", "OK" if hasattr(openai, name) else "MISSING")

print("dotenv:", "installed" if importlib.util.find_spec("dotenv") else "MISSING")
print("pydantic_settings:", "installed" if importlib.util.find_spec("pydantic_settings") else "MISSING")
