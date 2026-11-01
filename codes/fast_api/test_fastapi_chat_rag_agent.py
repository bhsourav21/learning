"""Automated tests for fastapi_chat_rag_agent.py using FastAPI's TestClient.

The real LLM is slow (20-40s per call), so most tests swap it for a fake that
answers instantly. This keeps the tests fast and the answers predictable.

Note: importing the app embeds the 5 documents once, so Ollama must be running.

Run:
    cd fast_api && uv run pytest test_fastapi_chat_rag_agent.py -v
"""
from types import SimpleNamespace

import httpx
import openai
import pytest
from fastapi.testclient import TestClient

import fastapi_chat_rag_agent as app_module

# raise_server_exceptions=False -> unexpected errors come back as a 500 response
# (like a real server) instead of crashing the test.
client = TestClient(app_module.app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Helpers to build fake LLM messages
# ---------------------------------------------------------------------------

def fake_answer(text: str):
    """A fake LLM message with a final answer and no tool calls."""
    return SimpleNamespace(content=text, tool_calls=None)


def fake_tool_call(name: str, arguments: str):
    """A fake LLM message that asks to call one tool."""
    call = SimpleNamespace(id="call_1", function=SimpleNamespace(name=name, arguments=arguments))
    return SimpleNamespace(content=None, tool_calls=[call])


@pytest.fixture
def fake_llm(monkeypatch):
    """Replace ask_llm with a fake that returns the given replies, in order.

    Usage in a test:  fake_llm(fake_answer("hi"))
    """
    def set_replies(*replies):
        replies = list(replies)
        monkeypatch.setattr(app_module, "ask_llm", lambda messages, tools=None: replies.pop(0))
    return set_replies


@pytest.fixture
def llm_down(monkeypatch):
    """Make every call to Ollama fail, as if Ollama was not running."""
    def raise_connection_error(*args, **kwargs):
        raise openai.APIConnectionError(request=httpx.Request("POST", "http://localhost:11434"))

    monkeypatch.setattr(app_module.client.chat.completions, "create", raise_connection_error)
    monkeypatch.setattr(app_module.client.embeddings, "create", raise_connection_error)


# ---------------------------------------------------------------------------
# POST /chat
# ---------------------------------------------------------------------------

def test_chat_200(fake_llm):
    fake_llm(fake_answer("Python is a programming language."))
    response = client.post("/chat", json={"prompt": "What is Python?"})
    assert response.status_code == 200
    assert response.json() == {"answer": "Python is a programming language."}


def test_chat_400_empty_prompt():
    response = client.post("/chat", json={"prompt": "   "})
    assert response.status_code == 400
    assert response.json() == {"error": "Bad Request", "detail": "'prompt' must not be empty"}


def test_chat_422_missing_field():
    response = client.post("/chat", json={"text": "wrong field name"})
    assert response.status_code == 422
    assert response.json()["error"] == "Validation Error"
    assert "prompt" in response.json()["detail"]


def test_chat_422_wrong_type():
    response = client.post("/chat", json={"prompt": 123})
    assert response.status_code == 422


def test_chat_500_llm_down(llm_down):
    response = client.post("/chat", json={"prompt": "Hi"})
    assert response.status_code == 500
    assert response.json()["error"] == "Internal Server Error"
    assert "LLM call failed" in response.json()["detail"]


# ---------------------------------------------------------------------------
# POST /rag-query
# ---------------------------------------------------------------------------

def test_rag_200(fake_llm, monkeypatch):
    # Make the "question" embedding identical to the Uvicorn document,
    # so search() must return that document first.
    uvicorn_index = 4
    monkeypatch.setattr(app_module, "embed", lambda texts: app_module.DOC_VECTORS[[uvicorn_index]])
    fake_llm(fake_answer("Uvicorn is an ASGI server."))

    response = client.post("/rag-query", json={"question": "What is Uvicorn?", "top_k": 2})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Uvicorn is an ASGI server."
    assert len(body["sources"]) == 2
    assert body["sources"][0] == app_module.DOCUMENTS[uvicorn_index]


def test_rag_400_empty_question():
    response = client.post("/rag-query", json={"question": ""})
    assert response.status_code == 400


@pytest.mark.parametrize("top_k", [0, -1, 99])
def test_rag_400_bad_top_k(top_k):
    response = client.post("/rag-query", json={"question": "Hi", "top_k": top_k})
    assert response.status_code == 400
    assert "top_k" in response.json()["detail"]


def test_rag_422_wrong_type():
    response = client.post("/rag-query", json={"question": "Hi", "top_k": "two"})
    assert response.status_code == 422


def test_rag_500_embedding_down(llm_down):
    response = client.post("/rag-query", json={"question": "Hi"})
    assert response.status_code == 500
    assert "Embedding failed" in response.json()["detail"]


# ---------------------------------------------------------------------------
# POST /agent-run
# ---------------------------------------------------------------------------

def test_agent_200_uses_tools(fake_llm):
    # The fake LLM: first asks for add(12, 30), then multiply(42, 3), then answers.
    fake_llm(
        fake_tool_call("add", '{"a": 12, "b": 30}'),
        fake_tool_call("multiply", '{"a": 42, "b": 3}'),
        fake_answer("The answer is 126."),
    )

    response = client.post("/agent-run", json={"task": "What is (12 + 30) * 3?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "The answer is 126."
    assert body["steps"] == [
        {"tool": "add", "arguments": {"a": 12, "b": 30}, "result": 42.0},
        {"tool": "multiply", "arguments": {"a": 42, "b": 3}, "result": 126.0},
    ]


def test_agent_200_stops_at_max_steps(fake_llm):
    # The fake LLM keeps asking for tools and never answers.
    fake_llm(
        fake_tool_call("add", '{"a": 1, "b": 1}'),
        fake_tool_call("add", '{"a": 1, "b": 1}'),
    )
    response = client.post("/agent-run", json={"task": "Loop forever", "max_steps": 2})
    assert response.status_code == 200
    assert response.json()["answer"] == "Stopped: reached max_steps."
    assert len(response.json()["steps"]) == 2


def test_agent_400_empty_task():
    response = client.post("/agent-run", json={"task": " "})
    assert response.status_code == 400


@pytest.mark.parametrize("max_steps", [0, 11])
def test_agent_400_bad_max_steps(max_steps):
    response = client.post("/agent-run", json={"task": "Hi", "max_steps": max_steps})
    assert response.status_code == 400


def test_agent_422_missing_task():
    response = client.post("/agent-run", json={"max_steps": 3})
    assert response.status_code == 422


def test_agent_500_unknown_tool(fake_llm):
    fake_llm(fake_tool_call("divide", '{"a": 1, "b": 2}'))
    response = client.post("/agent-run", json={"task": "What is 1 / 2?"})
    assert response.status_code == 500
    assert "unknown tool 'divide'" in response.json()["detail"]


def test_agent_500_llm_down(llm_down):
    response = client.post("/agent-run", json={"task": "Hi"})
    assert response.status_code == 500


# ---------------------------------------------------------------------------
# Unit tests for the plain Python tool functions
# ---------------------------------------------------------------------------

def test_tools():
    assert app_module.add(2, 3) == 5
    assert app_module.multiply(4, 2.5) == 10


# Lets you also run the tests with:  uv run python3 test_fastapi_chat_rag_agent.py
# (without this block, running the file directly defines the tests but never runs them)
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
