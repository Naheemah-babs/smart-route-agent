import pytest
from unittest.mock import patch
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.models import Gemini
from google.adk.models.llm_response import LlmResponse
from google.genai import types

from app.agent import root_agent

class MockResponse:
    def __init__(self, text):
        self.text = text

class MockModels:
    def __init__(self, text):
        self.text = text
    def generate_content(self, **kwargs):
        return MockResponse(self.text)

class MockClient:
    def __init__(self, text="shipping"):
        self.models = MockModels(text)

async def mock_generate_content_async(self, llm_request, stream=False):
    yield LlmResponse(
        content=types.Content(
            role="model",
            parts=[types.Part.from_text(text="Mocked FAQ Answer: Shipping rates are $5 flat.")]
        )
    )

@pytest.mark.asyncio
async def test_shipping_query_routing() -> None:
    session_service = InMemorySessionService()
    session = session_service.create_session_sync(user_id="test_user", app_name="test")
    runner = Runner(agent=root_agent, session_service=session_service, app_name="test")

    with patch("app.agent.genai.Client", return_value=MockClient("shipping")), \
         patch.object(Gemini, "generate_content_async", mock_generate_content_async):

        message = types.Content(
            role="user", parts=[types.Part.from_text(text="What are the shipping rates?")]
        )

        events = list(
            runner.run(
                new_message=message,
                user_id="test_user",
                session_id=session.id,
                run_config=RunConfig(streaming_mode=StreamingMode.SSE),
            )
        )

        has_faq_response = False
        for event in events:
            if (
                event.content
                and event.content.parts
                and any("Mocked FAQ Answer" in part.text for part in event.content.parts if part.text)
            ):
                has_faq_response = True
                break

        assert has_faq_response, "Expected routing to shipping FAQ agent and receiving mocked response"

@pytest.mark.asyncio
async def test_unrelated_query_routing() -> None:
    session_service = InMemorySessionService()
    session = session_service.create_session_sync(user_id="test_user", app_name="test")
    runner = Runner(agent=root_agent, session_service=session_service, app_name="test")

    with patch("app.agent.genai.Client", return_value=MockClient("unrelated")):
        message = types.Content(
            role="user", parts=[types.Part.from_text(text="Why is the sky blue?")]
        )

        events = list(
            runner.run(
                new_message=message,
                user_id="test_user",
                session_id=session.id,
                run_config=RunConfig(streaming_mode=StreamingMode.SSE),
            )
        )

        has_decline_response = False
        for event in events:
            if (
                event.content
                and event.content.parts
                and any("I am sorry, but I can only assist with shipping-related inquiries" in part.text for part in event.content.parts if part.text)
            ):
                has_decline_response = True
                break

        assert has_decline_response, "Expected routing to decline node and receiving decline message"
