from typing import (
    Any,
    AsyncGenerator,
    Dict,
    Generator,
    List,
    Optional,
    Sequence,
    Type,
    Union,
)

import httpx
from openai import AsyncOpenAI
from openai import OpenAI as SyncOpenAI

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.constants import DEFAULT_CONTEXT_WINDOW, DEFAULT_TEMPERATURE
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.core.llms.llm import Model
from llama_index.core.prompts import PromptTemplate
from llama_index.core.program.utils import FlexibleModel  
from llama_index.core.types import PydanticProgramMode
from llama_index.llms.openai.base import Tokenizer
from llama_index.llms.openai.responses import OpenAIResponses, DEFAULT_OPENAI_MODEL
from llama_index.llms.openai.utils import (
    resolve_openai_credentials,
    to_openai_message_dicts,
    is_json_schema_supported,
)


class OpenAILikeResponses(OpenAIResponses):
    """
    OpenAI-like Responses LLM with structured output support.

    This class extends OpenAIResponses to support the OpenAI /responses API for
    OpenAI-compatible servers. It provides the same responses API functionality
    but allows for different API endpoints and custom configurations.

    Features:
    - Support for OpenAI /responses API 
    - Structured output with Pydantic models
    - Function calling support
    - Streaming capabilities
    - Full async support
    - Custom API base URLs and authentication

    Args:
        model: name of the model to use.
        api_base: The base URL for the API.
        api_key: API key for authentication.
        temperature: a float from 0 to 1 controlling randomness in generation.
        max_output_tokens: the maximum number of tokens to generate.
        reasoning_options: Optional dictionary to configure reasoning for O1 models.
        include: Additional output data to include in the model response.
        instructions: Instructions for the model to follow.
        track_previous_responses: Whether to track previous responses.
        store: Whether to store previous responses in OpenAI's storage.
        built_in_tools: The built-in tools to use for the model to augment responses.
        truncation: Whether to auto-truncate the input if it exceeds the model's context window.
        user: An optional identifier to help track the user's requests for abuse.
        strict: Whether to enforce strict validation of the structured output.
        context_window: The context window to use for the api.
        is_chat_model: Whether the model uses the chat or completion endpoint.
        is_function_calling_model: Whether the model supports OpenAI function calling/tools.
        pydantic_program_mode: Mode for structured output (DEFAULT, OPENAI_JSON, LLM).
        additional_kwargs: Add additional parameters to OpenAI request body.
        max_retries: How many times to retry the API call if it fails.
        timeout: How long to wait, in seconds, for an API call before failing.
        default_headers: override the default headers for API requests.
        http_client: pass in your own httpx.Client instance.
        async_http_client: pass in your own httpx.AsyncClient instance.

    Examples:
        `pip install llama-index-llms-openai-like`

        Basic usage:
        ```python
        from llama_index.llms.openai_like import OpenAILikeResponses

        llm = OpenAILikeResponses(
            model="my-model",
            api_base="https://my-openai-compatible-api.com/v1",
            api_key="my-api-key",
            context_window=128000,
            is_chat_model=True,
            is_function_calling_model=True,
        )

        response = llm.complete("Hi, write a short story")
        print(response.text)
        ```

        Structured output with Pydantic models:
        ```python
        from pydantic import BaseModel, Field

        class PersonInfo(BaseModel):
            name: str = Field(description="Person's name")
            age: int = Field(description="Person's age")

        structured_llm = llm.as_structured_llm(PersonInfo)
        response = structured_llm.complete("Tell me about Alice, age 25")
        person_data = response.raw  # PersonInfo object
        print(f"Name: {person_data.name}, Age: {person_data.age}")
        ```

    """

    # OpenAI-like specific fields not inherited from OpenAIResponses
    context_window: Optional[int] = Field(
        default=DEFAULT_CONTEXT_WINDOW,
        description="The context window to use for the api.",
    )
    is_chat_model: bool = Field(
        default=True,
        description="Whether the model uses the chat or completion endpoint.",
    )
    is_function_calling_model: bool = Field(
        default=True,
        description="Whether the model supports OpenAI function calling/tools over the API.",
    )
    tokenizer: Union[Tokenizer, str, None] = Field(
        default=None,
        description=(
            "An instance of a tokenizer object that has an encode method, or the name"
            " of a tokenizer model from Hugging Face. If left as None, then this"
            " disables inference of max_tokens."
        ),
    )
    pydantic_program_mode: PydanticProgramMode = Field(
        default=PydanticProgramMode.DEFAULT,
        description="Pydantic program mode for structured output.",
    )

    def __init__(
        self,
        model: str = DEFAULT_OPENAI_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_output_tokens: Optional[int] = None,
        reasoning_options: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None,
        instructions: Optional[str] = None,
        track_previous_responses: bool = False,
        store: bool = False,
        built_in_tools: Optional[List[dict]] = None,
        truncation: str = "disabled",
        user: Optional[str] = None,
        previous_response_id: Optional[str] = None,
        call_metadata: Optional[Dict[str, Any]] = None,
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
        # OpenAI-like specific parameters
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        context_window: Optional[int] = None,
        is_chat_model: bool = True,
        is_function_calling_model: bool = True,
        max_retries: int = 3,
        timeout: float = 60.0,
        default_headers: Optional[Dict[str, str]] = None,
        tokenizer: Union[Tokenizer, str, None] = None,
        strict: bool = False,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        http_client: Optional[httpx.Client] = None,
        async_http_client: Optional[httpx.AsyncClient] = None,
        openai_client: Optional[SyncOpenAI] = None,
        async_openai_client: Optional[AsyncOpenAI] = None,
        **kwargs: Any,
    ) -> None:
        # Set OpenAI-like specific fields before calling parent constructor
        self.context_window = context_window or DEFAULT_CONTEXT_WINDOW
        self.is_chat_model = is_chat_model
        self.is_function_calling_model = is_function_calling_model
        self.tokenizer = tokenizer
        self.pydantic_program_mode = pydantic_program_mode
        
        # Call parent constructor with all the standard OpenAI parameters
        super().__init__(
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            reasoning_options=reasoning_options,
            include=include,
            instructions=instructions,
            track_previous_responses=track_previous_responses,
            store=store,
            built_in_tools=built_in_tools,
            truncation=truncation,
            user=user,
            previous_response_id=previous_response_id,
            call_metadata=call_metadata,
            strict=strict,
            additional_kwargs=additional_kwargs,
            max_retries=max_retries,
            timeout=timeout,
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
            default_headers=default_headers,
            http_client=http_client,
            async_http_client=async_http_client,
            openai_client=openai_client,
            async_openai_client=async_openai_client,
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        return "openai_like_responses_llm"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.max_output_tokens or -1,
            is_chat_model=self.is_chat_model,
            is_function_calling_model=self.is_function_calling_model,
            model_name=self.model,
        )

    @property
    def _tokenizer(self) -> Optional[Tokenizer]:
        """Get tokenizer for this model."""
        if isinstance(self.tokenizer, str):
            try:
                from transformers import AutoTokenizer
                return AutoTokenizer.from_pretrained(self.tokenizer)
            except ImportError:
                return None
        return self.tokenizer

    # ===== Structured Output Methods =====
    def _should_use_structure_outputs(self) -> bool:
        """Check if structured output should be used."""
        return (
            getattr(self, "pydantic_program_mode", PydanticProgramMode.DEFAULT) == PydanticProgramMode.DEFAULT
            and is_json_schema_supported(self.model)
        )

    def _prepare_schema(
        self, llm_kwargs: Optional[Dict[str, Any]], output_cls: Type[Model]
    ) -> Dict[str, Any]:
        """Prepare schema for structured output."""
        try:
            from openai.resources.beta.chat.completions import _type_to_response_format
            response_format = _type_to_response_format(output_cls)
        except ImportError:
            # Fallback for older OpenAI client versions or unsupported formats
            response_format = {"type": "json_object"}

        llm_kwargs = llm_kwargs or {}
        llm_kwargs["response_format"] = response_format
        if "tool_choice" in llm_kwargs:
            del llm_kwargs["tool_choice"]
        return llm_kwargs

    def structured_predict(
        self,
        output_cls: Type[Model],
        prompt: PromptTemplate,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        **prompt_args: Any,
    ) -> Model:
        """Structured predict using responses API."""
        llm_kwargs = llm_kwargs or {}

        if self._should_use_structure_outputs():
            messages = self._extend_messages(prompt.format_messages(**prompt_args))
            llm_kwargs = self._prepare_schema(llm_kwargs, output_cls)
            response = self.chat(messages, **llm_kwargs)
            return output_cls.model_validate_json(str(response.message.content))

        # Fallback to function calling for structured outputs
        llm_kwargs["tool_choice"] = (
            "required" if "tool_choice" not in llm_kwargs else llm_kwargs["tool_choice"]
        )
        return super().structured_predict(
            output_cls, prompt, llm_kwargs=llm_kwargs, **prompt_args
        )

    async def astructured_predict(
        self,
        output_cls: Type[Model],
        prompt: PromptTemplate,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        **prompt_args: Any,
    ) -> Model:
        """Async structured predict using responses API."""
        llm_kwargs = llm_kwargs or {}

        if self._should_use_structure_outputs():
            messages = self._extend_messages(prompt.format_messages(**prompt_args))
            llm_kwargs = self._prepare_schema(llm_kwargs, output_cls)
            response = await self.achat(messages, **llm_kwargs)
            return output_cls.model_validate_json(str(response.message.content))

        # Fallback to function calling for structured outputs
        llm_kwargs["tool_choice"] = (
            "required" if "tool_choice" not in llm_kwargs else llm_kwargs["tool_choice"]
        )
        return await super().astructured_predict(
            output_cls, prompt, llm_kwargs=llm_kwargs, **prompt_args
        )

    def stream_structured_predict(
        self,
        output_cls: Type[Model],
        prompt: PromptTemplate,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        **prompt_args: Any,
    ) -> Generator[Union[Model, FlexibleModel], None, None]:
        """Stream structured predict using responses API."""
        llm_kwargs = llm_kwargs or {}

        return super().stream_structured_predict(
            output_cls, prompt, llm_kwargs=llm_kwargs, **prompt_args
        )

    async def astream_structured_predict(
        self,
        output_cls: Type[Model],
        prompt: PromptTemplate,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        **prompt_args: Any,
    ) -> AsyncGenerator[Union[Model, FlexibleModel], None]:
        """Async stream structured predict using responses API.""" 
        llm_kwargs = llm_kwargs or {}
        return await super().astream_structured_predict(
            output_cls, prompt, llm_kwargs=llm_kwargs, **prompt_args
        )

    def _extend_messages(self, messages: List[ChatMessage]) -> List[ChatMessage]:
        """Extend messages with any additional context if needed."""
        return messages
