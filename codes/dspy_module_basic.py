import dspy
from ddgs import DDGS

lm = dspy.LM(
    "ollama_chat/gemma3:4b",
    api_base="http://localhost:11434",
    api_key="",
)
dspy.configure(lm=lm)

dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)

def web_search(query: str) -> str:
    """Real web search (DuckDuckGo, no api key required)"""
    try:
        results = list(DDGS().text(query, max_results=3))
    except Exception as e:
        raise RuntimeError(f"web_search failed: {e}")

    print(f"[web_search] live DuckDuckGo results for '{query}':")
    if not results:
        print("  (no results)")
        return "No results found"

    for r in results:
        print(f"  - {r['title']}: {r['body']}")

    return "\n".join(f"- {r['title']}: {r['body']}" for r in results)


sig = "question -> answer"

fast = dspy.Predict(sig)
thinking = dspy.ChainOfThought(sig)
agent = dspy.ReAct(sig, tools=[web_search])

for m in (fast, thinking, agent):
    print(m(question="How many time zones does Russia have today?").answer)

def web_search(query: str) -> str:
    """Real web search (DuckDuckGo, no api key required)"""
    try:
        results = list(DDGS().text(query, max_results=3))
    except Exception as e:
        raise RuntimeError(f"web_search failed: {e}")

    print(f"[web_search] live DuckDuckGo results for '{query}':")
    if not results:
        print("  (no results)")
        return "No results found"

    for r in results:
        print(f"  - {r['title']}: {r['body']}")

    return "\n".join(f"- {r['title']}: {r['body']}" for r in results)

