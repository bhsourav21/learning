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

agent = dspy.ReAct(
    "question -> reply",
    tools = [web_search],
    max_iters=3
)

out = agent(question="Where the BRICS 2026 event was held?")

print(f"response:{out.reply}")
print(f"trajectory:{out.trajectory}")
