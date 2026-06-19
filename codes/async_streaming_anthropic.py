from dotenv import load_dotenv
import anthropic
import asyncio

load_dotenv()
async_client = anthropic.AsyncAnthropic()

async def stream_response(prompt: str) -> str:
    """Stream a response and return the full text."""
    full_text = ""
    
    async with async_client.messages.stream(
        model="claude-haiku-4-5",
        max_tokens=1024,
        messages=[
            {
                "role": "user", "content": prompt
            }
        ],
    ) as stream:
        async for text in stream.text_stream:
            print(text, end="", flush=True)
            full_text += text
    print()
    return full_text


async def stream_multiple(prompts: list[str]):
    """Stream multiple completions CONCURRENTLY."""
    tasks = [stream_response(p) for p in prompts]
    results = await asyncio.gather(*tasks)
    return results

# Run it
asyncio.run(stream_response("What is the event loop in Python?"))
asyncio.run(stream_multiple(
    [
        "What is the event loop in Python in one line?",
        "What is the array in Python in one line?",
        "What is the enumerate in Python in one line?",
    ]
))

# Using asyncio.gather() lets you stream from multiple prompts 
# truly concurrently — all requests run
# in parallel on the same event loop, with no threads required.