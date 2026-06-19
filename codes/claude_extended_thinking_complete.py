"""
Complete extended thinking example.
Asks Claude a complex question and shows both thinking and answer.
"""

from dotenv import load_dotenv
import anthropic

load_dotenv()
client = anthropic.Anthropic()

def ask_claude_with_thinking(question: str, budget: int = 10000) -> dict:
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens = budget + 4000,
        thinking={
            "type": "enabled",
            "budget_tokens": budget
        },
        messages=[
            {
                "role": "user",
                "content": question
            }
        ],        
    )

    result = {
        "thinking": None,
        "answer": None,
        "thinking_tokens": 0
    }

    for block in response.content:
        if block.type == "thinking":
            result["thinking"] = block.thinking
        elif block.type == "text":
            result["answer"] = block.text

    
    # Token usage
    if hasattr(response.usage, "output_tokens_details"):
        result["thinking_tokens"] = (
            response.usage.output_tokens_details.thinking_tokens
        )
    
    return result

if __name__ == "__main__":
    question = """
    I have a list of 1 million numbers. I want to find all pairs of numbers
    that sum to a target value T. What is the most efficient algorithm,
    and what is its time and space complexity?
    """
    print(f"Question: {question.strip()}\n")
    print("Asking Claude with extended thinking (budget: 8,000 tokens)...")
    print("=" * 60)
    result = ask_claude_with_thinking(question, budget=8000)
    print(f"Thinking tokens used: {result['thinking_tokens']}")
    print("\n--- Claude's Thinking (summarised) ---")
    if result["thinking"]:
        # Show first 500 chars of thinking
        preview = result["thinking"][:500]
        print(preview + "..." if len(result["thinking"]) > 500 else preview)
    
    print("\n--- Final Answer ---")
    print(result["answer"])

