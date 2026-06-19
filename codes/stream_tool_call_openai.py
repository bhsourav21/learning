# When the model decides to call a tool, the function name and JSON 
# arguments arrive in streaming
# chunks too — they don't wait for the full JSON to be ready. 
# You must buffer the argument fragments
# and parse them once finish_reason == 'tool_calls'.

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
import json

load_dotenv()
client = OpenAI()

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["city"]
            }
        }
    }
]

tool_call_id = None
function_name = None
args_buffer = ""

with client.chat.completions.stream(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is the weather in Tokyo?"}],
    tools=tools,
    ) as stream:
    for event in stream:
        # stream() yields ChunkEvent objects — access the raw chunk via event.chunk
        if event.type != "chunk":
            continue
        chunk = event.chunk
        delta = chunk.choices[0].delta
        # Tool call chunks arrive through delta.tool_calls
        if delta.tool_calls:
            tc = delta.tool_calls[0]
            # First chunk sets the id and function name
            if tc.id:
                tool_call_id = tc.id
            if tc.function.name:
                function_name = tc.function.name
            # Subsequent chunks stream the JSON arguments
            if tc.function.arguments:
                args_buffer += tc.function.arguments
                print(tc.function.arguments, end="", flush=True)
        reason = chunk.choices[0].finish_reason
        if reason == "tool_calls":
            args = json.loads(args_buffer)
            print(f"\n\nTool call id: {tool_call_id}")
            print(f"\n\nTool called : {function_name}")
            print(f"\n\nArguments : {json.dumps(args, indent=2)}")
