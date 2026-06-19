from dotenv import load_dotenv
import anthropic
import json

load_dotenv()

client = anthropic.Anthropic()

tools = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City and state, e.g. San Francisco, CA"
                }
            },
        "required": ["location"]
        }
    }
]

args_buffer = ""
function_name = None

with client.messages.stream(
    model="claude-haiku-4-5",
    max_tokens=1024,
    tools=tools,
    messages=[{"role": "user", "content": "What is the weather in Tokyo?"}],
    ) as stream:
    for event in stream:
        # Tool input arrives as 'input_json_delta' events
        if hasattr(event, "type") and event.type == "content_block_delta":
            delta = event.delta
            if delta.type == "input_json_delta":
                args_buffer += delta.partial_json
                print("delta.partial_json:")
                print(delta.partial_json, end="", flush=True)
        # content_block_start tells us the tool name
        if hasattr(event, "type") and event.type == "content_block_start":
            cb = event.content_block
            if cb.type == "tool_use":
                function_name = cb.name
                print(f"\nClaude is calling tool: {function_name}")

final = stream.get_final_message()
if final.stop_reason == "tool_use":
    parsed_args = json.loads(args_buffer)
    print(f"\nTool : {function_name}")
    print(f"Arguments : {json.dumps(parsed_args, indent=2)}")