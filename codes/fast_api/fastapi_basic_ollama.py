from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from openai import OpenAI
from pydantic import BaseModel

# Free local LLM via Ollama (OpenAI-compatible API). No API key needed.
MODEL = "qwen3:4b"
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

app = FastAPI(title="Basic LLM API")


# Request body for POST /chat
class ChatRequest(BaseModel):
    prompt: str


# Response body for POST /chat
class ChatResponse(BaseModel):
    answer: str


@app.get("/")
def home():
    """Health check."""
    return {"status": "ok", "model": MODEL}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Send a prompt, get the full answer back as JSON."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": req.prompt}],
    )
    return ChatResponse(answer=response.choices[0].message.content)


def token_generator(prompt: str):
    """Yield the answer piece by piece as the LLM writes it."""
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    for chunk in stream:
        text = chunk.choices[0].delta.content
        if text:
            yield text


@app.get("/stream")
def stream(prompt: str):
    """Stream the answer token by token."""
    return StreamingResponse(token_generator(prompt), media_type="text/plain")


# To run:
# 1. Make sure Ollama is running:  ollama serve   (and: ollama pull qwen3:4b)
# 2. cd fast_api && uv run uvicorn fastapi_basic_ollama:app --reload
# 3. Open http://localhost:8000/docs to try the endpoints in the browser
#
# Or from the terminal:
#   curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{"prompt": "What is Python?"}'
#   curl -N "http://localhost:8000/stream?prompt=What+is+Python"
