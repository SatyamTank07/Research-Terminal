"""Unit and Integration Tests for Unified POST /chat Endpoint.

Validates:
1. ChatRequest schema defaults stream=True and accepts stream=False.
2. Endpoint routing:
   - POST /chat with stream=False returns synchronous JSON ChatResponse.
   - POST /chat with stream=True (or omitted default) returns text/event-stream.
   - Dropped endpoint POST /chat/stream returns 404.
3. Stream chat service:
   - Yields milestone status events, token deltas, and final result event.
4. Refactored helper extraction:
   - _prepare_chat_turn and _finalize_assistant_turn persistence.
"""

import asyncio
import json
import unittest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from app.main import app
from app.models import ChatMessage, Conversation, User
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat_service import (
    _finalize_assistant_turn,
    _prepare_chat_turn,
    process_chat,
    stream_chat_service,
)


class TestUnifiedChatEndpoint(unittest.TestCase):
    """Test suite verifying the consolidated POST /chat endpoint architecture."""

    def setUp(self):
        self.client = TestClient(app)

    def test_01_chat_request_schema_defaults(self):
        """Verify ChatRequest defaults stream=True when omitted."""
        req_default = ChatRequest(message="Analyze Apple")
        self.assertTrue(req_default.stream)
        self.assertEqual(req_default.agent_type, "multi_agent")

        req_sync = ChatRequest(message="Analyze Apple", stream=False)
        self.assertFalse(req_sync.stream)

    def test_02_dropped_chat_stream_endpoint_returns_404(self):
        """Verify that the redundant POST /chat/stream endpoint is dropped and returns 404."""
        response = self.client.post(
            "/chat/stream",
            json={"message": "Analyze Tesla"},
        )
        self.assertEqual(response.status_code, 404)

    @patch("app.routers.chat.get_single_user")
    @patch("app.routers.chat.process_chat")
    def test_03_post_chat_sync_mode(self, mock_process_chat, mock_get_user):
        """Verify POST /chat with stream=False returns JSON ChatResponse."""
        mock_user = MagicMock(spec=User)
        mock_user.id = 1
        mock_user.is_active = True
        mock_get_user.return_value = mock_user

        mock_process_chat.return_value = ChatResponse(
            response="Valuation report content",
            conversation_id="conv-123",
            message_id="msg-456",
            sources=[{"title": "10-K", "snippet": "Item 8 data"}],
        )

        response = self.client.post(
            "/chat",
            json={
                "message": "Calculate TSLA DCF",
                "conversation_id": "conv-123",
                "stream": False,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "application/json")
        data = response.json()
        self.assertEqual(data["response"], "Valuation report content")
        self.assertEqual(data["conversation_id"], "conv-123")
        self.assertEqual(data["message_id"], "msg-456")
        self.assertEqual(len(data["sources"]), 1)
        mock_process_chat.assert_called_once()

    @patch("app.routers.chat.get_single_user")
    def test_04_post_chat_streaming_mode(self, mock_get_user):
        """Verify POST /chat with stream=True returns StreamingResponse text/event-stream."""
        mock_user = MagicMock(spec=User)
        mock_user.id = 1
        mock_user.is_active = True
        mock_get_user.return_value = mock_user

        async def fake_stream(*args, **kwargs):
            yield 'data: {"type": "status", "node": "supervisor", "message": "Triaging"}\n\n'
            yield 'data: {"type": "token", "delta": "Analysis"}\n\n'
            yield 'data: {"type": "result", "response": "Analysis", "conversation_id": "c1", "message_id": "m1"}\n\n'

        with patch("app.routers.chat.stream_chat_service", side_effect=fake_stream):
            response = self.client.post(
                "/chat",
                json={
                    "message": "Analyze NVDA",
                    "stream": True,
                },
            )

            self.assertEqual(response.status_code, 200)
            self.assertIn("text/event-stream", response.headers["content-type"])
            content = response.text
            self.assertIn('"type": "status"', content)
            self.assertIn('"type": "token"', content)
            self.assertIn('"type": "result"', content)

    @patch("app.services.chat_service._prepare_chat_turn")
    @patch("app.services.chat_service._finalize_assistant_turn")
    @patch("app.services.chat_service.AgentRegistry.get")
    def test_05_stream_chat_service_yields_token_lifecycle(
        self, mock_registry_get, mock_finalize, mock_prepare
    ):
        """Verify stream_chat_service executes 3-phase streaming (status -> token -> result)."""
        mock_conv = MagicMock(spec=Conversation)
        mock_conv.id = "conv-abc"
        mock_conv.session_state = {}
        mock_user = MagicMock(spec=User)
        mock_user.id = 1
        mock_prepare.return_value = (mock_conv, MagicMock(), [{"role": "user", "content": "hi"}])

        mock_assistant_msg = MagicMock(spec=ChatMessage)
        mock_assistant_msg.id = "msg-xyz"
        mock_finalize.return_value = mock_assistant_msg

        # Mock agent with astream_run yielding a milestone and final result without tokens_streamed
        mock_agent = MagicMock()

        async def mock_astream_run(**kwargs):
            yield {"type": "status", "node": "supervisor", "message": "Resolved AAPL"}
            yield {
                "type": "result",
                "response": "Apple is undervalued with strong moat",
                "sources": [],
                "tokens_streamed": False,
            }

        mock_agent.astream_run = mock_astream_run
        mock_registry_get.return_value = mock_agent

        req = ChatRequest(message="Analyze AAPL", stream=True)
        db = MagicMock()

        loop = asyncio.new_event_loop()
        events = []
        try:
            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
                async def run_gen():
                    async for chunk in stream_chat_service(req, mock_user, db):
                        if chunk.startswith("data: "):
                            events.append(json.loads(chunk[6:].strip()))

                loop.run_until_complete(run_gen())
        finally:
            loop.close()

        event_types = [e["type"] for e in events]
        self.assertIn("status", event_types)
        self.assertIn("token", event_types)
        self.assertIn("result", event_types)

        # Confirm tokens contain words from the response
        token_deltas = "".join(e["delta"] for e in events if e["type"] == "token")
        self.assertEqual(token_deltas, "Apple is undervalued with strong moat")

        # Confirm final result payload
        result_event = next(e for e in events if e["type"] == "result")
        self.assertEqual(result_event["conversation_id"], "conv-abc")
        self.assertEqual(result_event["message_id"], "msg-xyz")
        self.assertEqual(result_event["response"], "Apple is undervalued with strong moat")

    @patch("app.services.chat_service._prepare_chat_turn")
    @patch("app.services.chat_service._finalize_assistant_turn")
    @patch("app.services.chat_service.AgentRegistry.get")
    def test_06_process_chat_service_unit(self, mock_registry_get, mock_finalize, mock_prepare):
        """Verify process_chat executes synchronously with zero real external API calls."""
        from app.agents.base import AgentOutput

        mock_conv = MagicMock(spec=Conversation)
        mock_conv.id = "conv-def"
        mock_conv.session_state = {}
        mock_user = MagicMock(spec=User)
        mock_user.id = 2
        mock_prepare.return_value = (mock_conv, MagicMock(), [{"role": "user", "content": "Analyze Tesla"}])

        mock_assistant_msg = MagicMock(spec=ChatMessage)
        mock_assistant_msg.id = "msg-tsla-1"
        mock_finalize.return_value = mock_assistant_msg

        mock_agent = MagicMock()
        mock_agent.run.return_value = AgentOutput(
            content="Tesla Valuation: DCF fair value is $245.50.",
            sources=[{"ticker": "TSLA", "item": "Item 8"}],
            updated_session_state={"ticker": "TSLA"},
        )
        mock_registry_get.return_value = mock_agent

        req = ChatRequest(message="Analyze Tesla", stream=False)
        db = MagicMock()

        with patch.dict("os.environ", {"OPENAI_API_KEY": "mocked-test-key"}):
            resp = process_chat(req, mock_user, db)

        self.assertIsInstance(resp, ChatResponse)
        self.assertEqual(resp.response, "Tesla Valuation: DCF fair value is $245.50.")
        self.assertEqual(resp.conversation_id, "conv-def")
        self.assertEqual(resp.message_id, "msg-tsla-1")
        self.assertEqual(len(resp.sources), 1)
        mock_agent.run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
