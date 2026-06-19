import openai
import anthropic
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

class LLMClient:
    def __init__(self, provider: str, model: str):
        self.provider = provider
        self.model = model

        if provider == "anthropic":
            self.client_anthropic = anthropic.Anthropic()
        elif provider == "openai":
            self.client_openai = OpenAI()
        else:
            raise ValueError(f"Unknown provided: {provider}")
    
    def chat(self, 
             messages: list[dict], 
             system: str = None, 
             max_tokens: int = 1024
    ):
        if self.provider == "anthropic":
            kwargs = dict(
                model = self.model,
                messages = messages,
                max_tokens = max_tokens
            )
            if system:
                kwargs["system"] = system
                resp = self.client_anthropic.messages.create(**kwargs)
                return resp.content[0].text
        elif self.provider == "openai":
            if system:
                messages = [
                    {
                        "role": "system",
                        "content": system
                    }
                ] + messages
                resp = self.client_openai.chat.completions.create(
                    model = self.model,
                    messages = messages,
                    max_tokens = max_tokens
                )
                return resp.choices[0].message.content
            

anthropic_client = LLMClient("anthropic", "claude-haiku-4-5")
openai_client = LLMClient("openai", "gpt-4o-mini")

response_anthropic = anthropic_client.chat(
    messages=[
        {
            "role": "user",
            "content": "What is 2+2?"
        }
    ],
    system="Answer concisely",
    max_tokens=512
)

response_openai = openai_client.chat(
    messages=[
        {
            "role": "user",
            "content": "What is 2+2?"
        }
    ],
    system="Answer concisely",
    max_tokens=512
)

print(f"response_anthropic:{response_anthropic}")
print(f"response_openai:{response_openai}")


