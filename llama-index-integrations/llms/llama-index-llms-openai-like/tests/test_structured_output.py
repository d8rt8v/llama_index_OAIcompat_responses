import pytest
from unittest.mock import patch, MagicMock
from pydantic import BaseModel, Field

from llama_index.llms.openai_like.responses import OpenAILikeResponses
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.prompts import PromptTemplate


class TestPydanticModel(BaseModel):
    name: str = Field(description="A person's name")
    age: int = Field(description="A person's age") 


@pytest.fixture
def structured_llm():
    """Create OpenAILikeResponses instance for structured output testing."""
    with (
        patch("llama_index.llms.openai.base.SyncOpenAI"),
        patch("llama_index.llms.openai.base.AsyncOpenAI"),
    ):
        return OpenAILikeResponses(
            model="gpt-4o", 
            api_key="fake-key",
            api_base="https://test-api.com/v1",
            is_chat_model=True,
            is_function_calling_model=True,
        )


def test_structured_output_creation(structured_llm):
    """Test that we can create a structured LLM."""
    sllm = structured_llm.as_structured_llm(TestPydanticModel)
    assert sllm is not None
    assert sllm.output_cls == TestPydanticModel


def test_should_use_structure_outputs(structured_llm):
    """Test _should_use_structure_outputs method."""
    # Mock is_json_schema_supported to return True
    with patch('llama_index.llms.openai_like.responses.is_json_schema_supported', return_value=True):
        assert structured_llm._should_use_structure_outputs() is True
        
    # Test with unsupported model
    with patch('llama_index.llms.openai_like.responses.is_json_schema_supported', return_value=False):
        assert structured_llm._should_use_structure_outputs() is False


def test_prepare_schema(structured_llm):
    """Test _prepare_schema method."""
    llm_kwargs = {"temperature": 0.7, "tool_choice": "auto"}
    
    with patch('llama_index.llms.openai_like.responses._type_to_response_format') as mock_format:
        mock_format.return_value = {"type": "json_object"}
        
        result = structured_llm._prepare_schema(llm_kwargs, TestPydanticModel)
        
        assert "response_format" in result
        assert "tool_choice" not in result  # Should be removed
        assert result["temperature"] == 0.7
        mock_format.assert_called_once_with(TestPydanticModel)


@patch("llama_index.llms.openai.base.SyncOpenAI")
def test_structured_predict_with_json_mode(mock_sync_openai, structured_llm):
    """Test structured_predict using JSON mode."""
    # Mock the chat response
    mock_response = MagicMock()
    mock_response.message.content = '{"name": "Alice", "age": 25}'
    
    # Mock the chat method
    structured_llm.chat = MagicMock(return_value=mock_response)
    
    # Mock _should_use_structure_outputs to return True
    structured_llm._should_use_structure_outputs = MagicMock(return_value=True)
    structured_llm._extend_messages = MagicMock(return_value=[ChatMessage(role=MessageRole.USER, content="test")])
    structured_llm._prepare_schema = MagicMock(return_value={"response_format": {"type": "json_object"}})
    
    prompt = PromptTemplate("Create a person with name Alice and age 25")
    result = structured_llm.structured_predict(TestPydanticModel, prompt)
    
    assert isinstance(result, TestPydanticModel)
    assert result.name == "Alice"
    assert result.age == 25
    

@patch("llama_index.llms.openai.base.AsyncOpenAI")
@pytest.mark.asyncio
async def test_astructured_predict_with_json_mode(mock_async_openai, structured_llm):
    """Test async structured_predict using JSON mode."""
    # Mock the async chat response
    mock_response = MagicMock()
    mock_response.message.content = '{"name": "Bob", "age": 30}'
    
    # Mock the achat method as async
    async def mock_achat(*args, **kwargs):
        return mock_response
    
    structured_llm.achat = mock_achat
    
    # Mock _should_use_structure_outputs to return True
    structured_llm._should_use_structure_outputs = MagicMock(return_value=True)
    structured_llm._extend_messages = MagicMock(return_value=[ChatMessage(role=MessageRole.USER, content="test")])
    structured_llm._prepare_schema = MagicMock(return_value={"response_format": {"type": "json_object"}})
    
    prompt = PromptTemplate("Create a person with name Bob and age 30")
    result = await structured_llm.astructured_predict(TestPydanticModel, prompt)
    
    assert isinstance(result, TestPydanticModel)
    assert result.name == "Bob"
    assert result.age == 30


def test_structured_predict_fallback_to_function_calling(structured_llm):
    """Test structured_predict falls back to function calling when JSON mode is not supported."""
    # Mock _should_use_structure_outputs to return False
    structured_llm._should_use_structure_outputs = MagicMock(return_value=False)
    
    # Mock the super() call
    with patch.object(OpenAILikeResponses.__bases__[0], 'structured_predict') as mock_super:
        mock_super.return_value = TestPydanticModel(name="Charlie", age=35)
        
        prompt = PromptTemplate("Create a person")
        result = structured_llm.structured_predict(TestPydanticModel, prompt)
        
        assert isinstance(result, TestPydanticModel)
        assert result.name == "Charlie"
        assert result.age == 35
        
        # Verify that super().structured_predict was called with tool_choice required
        mock_super.assert_called_once()
        args, kwargs = mock_super.call_args
        assert kwargs["llm_kwargs"]["tool_choice"] == "required"