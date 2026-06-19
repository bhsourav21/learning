from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import asyncio

load_dotenv()
app = FastAPI()
client = AsyncOpenAI()

async def token_generator(prompt: str):
    """Yield SSE-formatted chunks for the browser."""
    async with client.chat.completions.stream(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        async for event in stream:
            if event.type == "content.delta":
                # Server-Sent Event format: 'data: <payload>\n\n'
                yield f"data: {event.delta}\n\n"
        yield "data: [DONE]\n\n"  # sent once after the stream ends

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
