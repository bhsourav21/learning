from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam

load_dotenv()
client = OpenAI()

collected_chunks = []
full_text = ""

# Streaming: tokens arrive one chunk at a time
stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user", 
            "content": "Explain recursion breifly"
        }
    ],
    stream=True # <-- this enables streaming
)

# Iterate over each chunk as it arrives
for chunk in stream:
    delta = chunk.choices[0].delta
    finish_reason = chunk.choices[0].finish_reason

if finish_reason is not None:
    print(f"stream finish reason: {finish_reason}")


# Possible values:
# 'stop' → model finished naturally
# 'length' → hit max_tokens limit
# 'tool_calls' → model called a tool/function
# 'content_filter'→ content was filtered
# 'function_call' → deprecated, use tool_calls