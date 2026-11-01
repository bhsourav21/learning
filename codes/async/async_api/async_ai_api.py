import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import AsyncOpenAI

# Free local LLM via Ollama (OpenAI-compatible API). No API key needed.
MODEL = "qwen3:4b"
MAX_PARALLEL = 5
limiter = asyncio.Semaphore(MAX_PARALLEL)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # One shared client = reused connections (fast). Created once at startup.
    app.state.llm = AsyncOpenAI( # 1. runs once at startup
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        timeout=200.0,
        max_retries=0,
    )
    yield  # 2. server handles requests here
    await app.state.llm.close() # clean shutdown

app = FastAPI(lifespan=lifespan)

class Prompt(BaseModel):
    text: str

class BatchPrompt(BaseModel):
    texts: list[str]

async def ask_llm(text: str) -> str:
    async with limiter: # wait here if 5 calls are running
        r = await app.state.llm.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": text}],
        )
        return r.choices[0].message.content

@app.post("/generate")
async def generate(p: Prompt):
    try:
        async with asyncio.timeout(20):
            answer = await ask_llm(p.text)
    except TimeoutError:
        raise HTTPException(504, "Model took too long")
    return {"answer": answer}

@app.post("/batch")
async def batch(b: BatchPrompt):
    answers = await asyncio.gather(*(ask_llm(t) for t in b.texts))
    return {"answers": answers}

async def fake_token_stream(text: str):
    for word in f"You asked: {text}".split():
        await asyncio.sleep(0.2) # model "thinking per token"
        yield f"data: {word}\n\n"

@app.post("/stream")
async def stream(p: Prompt):
    return StreamingResponse(fake_token_stream(p.text),
                             media_type="text/event-stream")


def blocking_sentiment(text: str) -> str:
    import time; time.sleep(1)
    return "positive" if "good" in text else "neutral"

app.post("/sentiment")
async def sentiment(p: Prompt):
    label = await asyncio.to_thread(blocking_sentiment, p.text)
    return {"label": label}

