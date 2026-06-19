from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam

load_dotenv()

client = OpenAI()

# Streaming: tokens arrive one chunk at a time
stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user", 
            "content": "Write a short poem about python"
        }
    ],
    stream=True # <-- this enables streaming
)

# Iterate over each chunk as it arrives
for chunk in stream:
    print("chunk:")
    print(chunk)
    # Each chunk has a list of 'choices'
    delta = chunk.choices[0].delta

    # delta.content holds the new token(s); None when there's nothing yet
    if delta.content is not None:
        print(delta.content, end="", flush=True)

print() # newline after streaming finishes