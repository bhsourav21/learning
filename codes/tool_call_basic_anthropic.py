from dotenv import load_dotenv
import anthropic
import json
import requests

load_dotenv()
client = anthropic.Anthropic()

# 1. Define a list of callable tools for the model
tools = [
    {
        "name": "get_current_weather",
        "description": "Get current weather of a location",
        "input_schema": {
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
response = client.messages.create(
    model="claude-haiku-4-5",
    tools=tools,
    max_tokens = 1024,
    messages=input_list,
)
print('--------------------------------------')
print(f"tool call response:{response.content}")

# # Save the full assistant turn as a proper message
# input_list.append({"role": "assistant", "content": response.content})
# print('--------------------------------------')
# print(f"input_list appended with tool call response: {input_list}")

# for block in response.content:
#     if block.type == "tool_use":
#         if block.name == "get_current_weather":
#             # 3. Execute the function logic for get_horoscope
#             location = block.input["location"]
#             curr_weather = get_current_weather(location)
            
#             # 4. Provide function call results to the model
#             input_list.append({
#                 "role": "user", 
#                 "content": [
#                     {
#                         "type": "tool_result", 
#                         "tool_use_id": block.id, 
#                         "content": json.dumps(curr_weather)
#                     }
#                 ]
#             })

# print('--------------------------------------')
# print("Final input:")
# print(input_list)

# response = client.messages.create(
#     model="claude-haiku-4-5",
#     max_tokens = 1024,
#     # system="Respond only with weather fetched by a tool.",
#     messages=input_list,
# )

# # 5. The model should be able to give a response!
# print('--------------------------------------')
# print("Final output:")
# print(response.model_dump_json(indent=2))
# print("\n" + response.content[0].text)