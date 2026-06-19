from dotenv import load_dotenv
import openai
import asyncio

load_dotenv()
async_client = openai.AsyncOpenAI()

async def stream_response(prompt: str) -> str:
    """Stream a response and return the full text."""
    full_text = ""
    
    async with async_client.chat.completions.stream(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user", "content": prompt
            }
        ],
    ) as stream:
        async for event in stream:
            if event.type == "content.delta":
                print(event.delta, end="", flush=True)
                full_text += event.delta
    print()
    return full_text


async def stream_multiple(prompts: list[str]):
    """Stream multiple completions CONCURRENTLY."""
    tasks = [stream_response(p) for p in prompts]
    results = await asyncio.gather(*tasks)
    return results

# Run it
asyncio.run(stream_response("What is the event loop in Python?"))
# asyncio.run(stream_multiple(
#     [
#         "What is the event loop in Python in one line?",
#         "What is the array in Python in one line?",
#         "What is the enumerate in Python in one line?",
#     ]
# ))

# Using asyncio.gather() lets you stream from multiple prompts 
# truly concurrently — all requests run
# in parallel on the same event loop, with no threads required.