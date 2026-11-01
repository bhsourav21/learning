import asyncio
from openai import AsyncOpenAI

MODEL = "qwen3:4b"

async def ask(client, text):
    r = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": text}],
    )
    return r.choices[0].message.content

async def main():
    async with AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama", timeout=300, max_retries=0) as client:
        answers = await asyncio.gather(
            ask(client, "What is async?"),
            ask(client, "What is an event loop?"),
        )
        print(answers)

        # read the stream token by token
        stream = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "hello"}],
            stream=True,
        )
        async for chunk in stream:
            token = chunk.choices[0].delta.content
            if token:
                print(token, end="", flush=True)
        print()

print(asyncio.run(main()))
