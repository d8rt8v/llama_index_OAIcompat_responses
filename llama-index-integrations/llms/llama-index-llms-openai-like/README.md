# LlamaIndex Llms Integration: OpenAI Like

`pip install llama-index-llms-openai-like`

This package is a thin wrapper around the OpenAI API. It is designed to be used with the OpenAI API, but can be used with any OpenAI-compatible API.

## Classes

This integration provides two classes:

1. **OpenAILike** - For standard OpenAI-compatible APIs using chat/completions endpoints
2. **OpenAILikeResponses** - For OpenAI-compatible APIs that support the `/responses` endpoint

## Usage

### Basic OpenAI-compatible API (OpenAILike)

```python
from llama_index.llms.openai_like import OpenAILike

llm = OpenAILike(
    model="model-name",
    api_base="http://localhost:1234/v1",
    api_key="fake",
    # Explicitly set the context window to match the model's context window
    context_window=128000,
    # Controls whether the model uses chat or completion endpoint
    is_chat_model=True,
    # Controls whether the model supports function calling
    is_function_calling_model=False,
)

response = llm.complete("Hello World!")
print(response.text)
```

### OpenAI-compatible API with Responses support (OpenAILikeResponses)

For OpenAI-compatible servers that support the `/responses` API endpoint (similar to OpenAI's responses API), use `OpenAILikeResponses`:

```python
from llama_index.llms.openai_like import OpenAILikeResponses

llm = OpenAILikeResponses(
    model="gpt-4o-mini",
    api_base="https://your-openai-compatible-api.com/v1",
    api_key="your-api-key",
    context_window=128000,
    is_chat_model=True,
    is_function_calling_model=True,
    
    # Responses-specific parameters
    max_output_tokens=1000,
    instructions="You are a helpful assistant.",
    track_previous_responses=True,
    built_in_tools=[{"type": "web_search"}],
    user="user_id",
)

response = llm.complete("Write a short story")
print(response.text)
```

### Key Features of OpenAILikeResponses

- **Built-in Tools**: Support for built-in tools like web search, code interpreter, etc.
- **Response Tracking**: Track previous responses for conversational context
- **Instructions**: Set global instructions for the model
- **Advanced Function Calling**: Enhanced function calling with parallel execution support
- **Response Storage**: Optional storage of responses in the provider's system
- **Streaming Support**: Full streaming support for both chat and completion

### Function Calling with OpenAILikeResponses

```python
from llama_index.core.tools import FunctionTool

def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Search results for: {query}"

search_tool = FunctionTool.from_defaults(fn=search_web)

response = llm.chat_with_tools(
    tools=[search_tool],
    user_msg="Search for the latest AI developments",
    tool_required=True
)
```

## When to Use Which Class

- Use **OpenAILike** for standard OpenAI-compatible APIs that use `/chat/completions` or `/completions` endpoints
- Use **OpenAILikeResponses** for OpenAI-compatible APIs that support the `/responses` endpoint and you want to leverage advanced features like built-in tools, response tracking, and enhanced function calling
