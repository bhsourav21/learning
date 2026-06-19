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
    {"role": "user", "content": "What is the weather of Beaverton, OR?"}
]

# 2. Prompt the model with tools defined
response = client.responses.create(
    model="gpt-4o-mini",
    tools=tools,
    input=input_list,
)
print('--------------------------------------')
print(f"tool call response:{response.output}")

# Save function call outputs for subsequent requests
input_list += response.output
print('--------------------------------------')
print(f"input_list appended with tool call response: {input_list}")

for item in response.output:
    if item.type == "function_call":
        if item.name == "get_current_weather":
            # 3. Execute the function logic for get_horoscope
            location = json.loads(item.arguments)["location"]
            curr_weather = get_current_weather(location)
            
            # 4. Provide function call results to the model
            input_list.append({
                "type": "function_call_output",
                "call_id": item.call_id,
                "output": json.dumps(curr_weather),
            })

print('--------------------------------------')
print("Final input:")
print(input_list)

response = client.responses.create(
    model="gpt-4o-mini",
    # instructions="Respond only with weather fetched by a tool.",
    input=input_list,
)

# 5. The model should be able to give a response!
print('--------------------------------------')
print("Final output:")
print(response.model_dump_json(indent=2))
print("\n" + response.output_text)