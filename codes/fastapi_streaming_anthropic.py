from dotenv import load_dotenv
import anthropic
from anthropic import AsyncAnthropic
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import asyncio

load_dotenv()
app = FastAPI()
client = AsyncAnthropic()

async def token_generator(prompt: str):
    """Yield SSE-formatted chunks for the browser."""
    async with client.messages.stream(
        model="claude-haiku-4-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        async  for text in stream.text_stream:
            # Server-Sent Event format: 'data: <payload>\n\n'
            yield f"data: {text}\n\n"
    yield "data: [DONE]\n\n"

@app.get("/stream")
async def stream_endpoint(prompt: str):
    return StreamingResponse(
        token_generator(prompt),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no", # Disable nginx buffering
        },
    )


# To run:
# 1. uv run uvicorn fastapi_streaming_openai:app --reload
# 2. http://localhost:8000/stream?prompt=What+is+Python