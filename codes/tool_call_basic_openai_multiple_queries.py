from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
import json
import requests

load_dotenv()
client = OpenAI()

# 1. Define a list of callable tools for the model
tools = [
    {
        "type": "function",
        "name": "get_current_weather",
        "description": "Get current weather of a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "A location or city like Paris, Tokyo etc",
                },
            },
            "required": ["location"],
        },
    },
]

def get_current_weather(city: str) -> dict:
    url = f"https://wttr.in/{city}"
    response = requests.get(url, params={"format": "j1"})
    response.raise_for_status()
    data = response.json()["current_condition"][0]
    
    return {
        "temperature_c": data["temp_C"],
        "temperature_f": data["temp_F"],
        "humidity": data["humidity"],
        "description": data["weatherDesc"][0]["value"],
        "wind_speed_kmph": data["windspeedKmph"],
        "feels_like_c": data["FeelsLikeC"],
    }

# Create a running input list we will add to over time
input_list = [
    {"role": "user", "content": (
        "Please answer all of these:\n"
        "1. What is the weather of Beaverton, OR?\n"
        "2. What is the capital of France?\n"
        "3. How is the weather in Delhi?\n"
        "4. Who is the captain of the Indian test squad?\n"
        "5. Who invented the C programming language?"
    )}
]

# 2. Prompt the model with tools defined
response = client.responses.create(
    model="gpt-4o-mini",
    tools=tools,
    input=input_list,
)
print('--------------------------------------')
print(f"tool call response:{response.output}")

# 3 & 4. Execute tools and collect results
tool_outputs = []
for item in response.output:
    if item.type == "function_call" and item.name == "get_current_weather":
        location = json.loads(item.arguments)["location"]
        curr_weather = get_current_weather(location)
        curr_weather["location"] = location  # helps model match result to the right city
        tool_outputs.append({
            "type": "function_call_output",
            "call_id": item.call_id,
            "output": json.dumps(curr_weather),
        })

print('--------------------------------------')
print("Tool outputs:")
print(tool_outputs)

# Use previous_response_id so OpenAI manages conversation state server-side
# Only pass the new tool outputs, not the full history
response = client.responses.create(
    model="gpt-4o-mini",
    tools=tools,
    previous_response_id=response.id,
    input=tool_outputs,
)

# 5. The model should be able to give a response!
print('--------------------------------------')
print("Final output:")
print(response.model_dump_json(indent=2))
print("\n" + response.output_text)