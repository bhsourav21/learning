from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

system_prompt = (
    "You are a math expert. "
    "End your answer with: Final Answer: <number><unit>"
)

# Few-Shot CoT: show the AI HOW to reason with examples
few_shot_prompt = '''
Example 1:
Q: A store has 20 apples. They sell 8. Then they get 5 more. How many apples?
A: Start with 20 apples.
Sell 8: 20 - 8 = 12 apples.
Get 5 more: 12 + 5 = 17 apples.
Answer: 17 apples.

Example 2:
Q: A jar has 30 sweets. 3 children each take 4 sweets. How many are left?
A: Each child takes 4 sweets. 3 children: 3 x 4 = 12 sweets taken.
Remaining: 30 - 12 = 18 sweets.
Answer: 18 sweets.

Now answer this:
Q: A parking lot has 50 spaces. 18 cars are parked.
Then 7 cars leave and 12 new cars arrive. How many free spaces are left?
'''

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o-mini",
    max_tokens=1024,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": few_shot_prompt},
    ],
)

print(response.choices[0].message.content)