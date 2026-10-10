import os
import sys
import operator
from typing import Annotated, Literal, TypedDict
from ddgs import DDGS
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.types import Command
from pydantic import BaseModel, Field

MODEL = "qwen3:4b"
MAX_REVISIONS = 2
MAX_STEPS = 10

llm = ChatOpenAI(
    model=MODEL,
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    temperature=0,
    max_tokens=3000,
)

class AgentState(TypedDict):
    topic: str
    messages: Annotated[list, add_messages]

    # Artifacts produced by each specialist (last-write-wins, single writer each)
    research: str
    sources: list[str]
    draft: str
    critique: str

    # Control state — not content. This is what makes the cycles finite.
    revisions: int              # how many drafts the writer has produced
    critique_of_revision: int   # which draft the current critique refers to
    steps: int                  # how many times the supervisor has dispatched

    route_log: Annotated[list[str], operator.add]

@tool
def web_search(query: str, max_results: int = 5) -> dict:
    """Search the web for current facts and return findings plus source URLs.

    Uses DuckDuckGo via ddgs — no API key required.
    """
    try:
        results = list(DDGS().text(query, max_results=max_results))
    except Exception as e:
        raise RuntimeError(f"web_search failed: {e}")

    if not results:
        return {"findings": "No results found", "sources": []}

    return {
        "findings": "\n".join(f"- {r['title']}: {r['body']}" for r in results),
        # Sources are returned alongside the prose so the critic can check claims
        # against them instead of only vibe-checking the writing.
        "sources": [r["href"] for r in results if r.get("href")],
    }

class Route(BaseModel):
    next: Literal["researcher", "writer", "critic", "FINISH"] = Field(
        description="Which specialist runs next, or FINISH if the work is done"
    )
    reason: str = Field(description="One sentence explaining the choice")

ROUTER_SYSTEM = """You are the supervisor of a three-agent content team.

Specialists you can dispatch to:
  researcher - runs a web search and produces factual notes on the topic
  writer     - turns the research into a blog post draft; on a rerun it revises
               the existing draft using the critique
  critic     - reviews the current draft against the research for accuracy,
               structure and clarity

Reply FINISH when the draft is good enough to ship.

Your one hard decision is what to do after a critique arrives: send the draft
back to the writer, or FINISH. Weigh whether another rewrite would actually
fix what the critique identifies, and respect the revision budget you are told
about — a draft that is merely imperfect is not worth the last revision."""

def router(state: AgentState) -> Command[Literal["researcher", "writer", "critic", "__end__"]]:
    """Supervisor node: one LLM call, then resolve it against the state"""
    steps = state.get("steps", 0)
    revisions = state.get(  "revisions", 0)

    if steps > MAX_STEPS:
        log = f"router -> END [STEP BUDGET spent ({steps}/{MAX_STEPS})]"
        print(f"  {log}")
        return Command(goto=END, update={"route_log": [log]})

    critique_is_current = (
        bool(state.get("critique"))
        and state.get("critique_of_revision", -1) == revisions
    )

    decision: Route = llm.with_structured_output(Route).invoke(
        [
            {"role": "system", "content": ROUTER_SYSTEM},
            {
                "role": "user",
                "content": f"""Topic: {state['topic']}

Current state of the work:
  research  : {"DONE" if state.get("research") else "MISSING"}
  draft     : {"DONE (draft #" + str(revisions) + ")" if state.get("draft") else "MISSING"}
  critique  : {"CURRENT" if critique_is_current else ("STALE" if state.get("critique") else "MISSING")}

Revision budget: {revisions} of {MAX_REVISIONS} drafts used.

Latest critique (empty if none yet):
{state.get("critique", "") or "(none)"}

Which specialist should run next?""",
            },
        ]
    )

    if not state.get("research"):
        nxt, why = "researcher", "no research yet"
    elif not state.get("draft"):
        nxt, why = "writer", "no draft yet"
    elif not critique_is_current:
        nxt, why = "critic", f"draft #{revisions} not yet reviewed"
    elif revisions >= MAX_REVISIONS:
        nxt, why = "FINISH", f"revision budget spent ({revisions}/{MAX_REVISIONS})"
    else:
        # Everything exists, the critique is current, and budget remains. THIS is
        # the point where the LLM's answer actually decides something: revise the
        # draft, dig up more research, or ship it. "critic" is excluded because
        # re-reviewing an unchanged draft is meaningless work.
        nxt = decision.next if decision.next in ("researcher", "writer", "FINISH") else "FINISH"
        why = None  # the LLM's own reason is logged instead

    if why is None:
        log = f"router -> {nxt} ({decision.reason})"
    else:
        log = f"router -> {nxt} [forced: {why}; LLM said '{decision.next}']"
    print(f"  {log}")

    return Command(
        goto=END if nxt == "FINISH" else nxt,
        update={"steps": steps + 1, "route_log": [log]},
    )
    
def researcher(state: AgentState) -> AgentState:
    """Web search -> factual notes. Writes research + sources."""
    result = web_search.invoke(({"query": state["topic"]}))

    notes = llm.invoke(
        [
            {
                "role": "system",
                "content": "You are a research assistant. Condense the search results "
                "into tight factual notes a writer can build on. Bullet points, "
                "concrete facts and numbers only, no filler, no opinions.",
            },
            {
                "role": "user",
                "content": f"Topic: {state['topic']}\n\nSearch results:\n{result['findings']}",
            },
        ]
    ).content

    return {
        "research": notes,
        "sources": result["sources"],
        "messages": [AIMessage(content=f"Research notes:\n{notes}", name="researcher")],
        "route_log": [f"researcher: gathered notes from {len(result['sources'])} sources"],
    }


def writer(state: AgentState) -> AgentState:
    """Research (+ critique on a rerun) -> draft. Owns the revisions counter."""
    revisions = state.get("revisions", 0)
    is_revision = revisions > 0 and bool(state.get("draft"))

    if is_revision:
        task = (
            f"Revise the draft below to address every point in the critique. "
            f"Keep what already works.\n\n"
            f"CURRENT DRAFT:\n{state['draft']}\n\n"
            f"CRITIQUE TO ADDRESS:\n{state['critique']}"
        )
    else:
        task = "Write the first draft of the blog post."

    draft = llm.invoke(
        [
            {
                "role": "system",
                "content": "You are a technical blog writer. Produce a clear, well-structured "
                "post of roughly 400-500 words grounded strictly in the research notes. "
                "Do not invent facts. Output the post only — no preamble.",
            },
            {
                "role": "user",
                "content": f"Topic: {state['topic']}\n\n"
                f"RESEARCH NOTES:\n{state['research']}\n\n{task}",
            },
        ]
    ).content

    label = "revised draft" if is_revision else "first draft"
    return {
        "draft": draft,
        "revisions": revisions + 1,
        "messages": [AIMessage(content=f"Produced {label}.", name="writer")],
        "route_log": [f"writer: produced draft #{revisions + 1} ({label})"],
    }

def critic(state: AgentState) -> AgentState:
    """Draft + research -> critique. Stamps which draft it reviewed."""
    critique = llm.invoke(
        [
            {
                "role": "system",
                "content": "You are a demanding editor. Review the draft against the research "
                "notes for factual accuracy, structure, and clarity. Give at most four "
                "specific, actionable points. If the draft is genuinely ready to publish, "
                "say so plainly and explain why in one sentence.",
            },
            {
                "role": "user",
                "content": f"Topic: {state['topic']}\n\n"
                f"RESEARCH NOTES:\n{state['research']}\n\n"
                f"DRAFT TO REVIEW:\n{state['draft']}",
            },
        ]
    ).content

    revisions = state.get("revisions", 0)
    return {
        "critique": critique,
        # Stamping the revision number is how the router tells a fresh critique
        # from one written about an older draft.
        "critique_of_revision": revisions,
        "messages": [AIMessage(content=f"Critique of draft #{revisions}:\n{critique}", name="critic")],
        "route_log": [f"critic: reviewed draft #{revisions}"],
    }

def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("router", router)
    graph.add_node("researcher", researcher)
    graph.add_node("writer", writer)
    graph.add_node("critic", critic)

    graph.add_edge(START, "router")
    graph.add_edge("researcher", "router")
    graph.add_edge("writer", "router")
    graph.add_edge("critic", "router")

    return graph.compile()

# build_graph() - WITH the parentheses. Without them `app` is the function
# object itself, and app.invoke(...) raises AttributeError.
app = build_graph()

def initial_state(topic: str) -> AgentState:
    return {
        "topic": topic,
        "messages": [HumanMessage(content=f"Research {topic}, write a blog post, then review it.")],
        "research": "",
        "sources": [],
        "draft": "",
        "critique": "",
        "revisions": 0,
        "critique_of_revision": -1,
        "steps": 0,
        "route_log": [],
    }

def main():
    topic = sys.argv[1] if len(sys.argv) > 1 else (
        "the LangGraph supervisor pattern for multi-agent orchestration"
    )

    print("=" * 70)
    print("  Supervisor Multi-Agent Graph — researcher / writer / critic")
    print("=" * 70)
    print(f"\nTopic: {topic}")
    print(f"Revision budget: {MAX_REVISIONS} drafts\n")
    print("Routing trace:")

    result = app.invoke(initial_state(topic))

    print("\n" + "=" * 70)
    print("  EXECUTION TRACE")
    print("=" * 70)
    for line in result["route_log"]:
        print(" ", line)

    print("\n" + "=" * 70)
    print("  AGENT-TO-AGENT TRANSCRIPT (state['messages'])")
    print("=" * 70)
    for m in result["messages"]:
        who = getattr(m, "name", None) or type(m).__name__
        preview = m.content[:120].replace("\n", " ")
        print(f"  [{who}] {preview}{'...' if len(m.content) > 120 else ''}")

    print("\n" + "=" * 70)
    print(f"  FINAL BLOG POST (draft #{result['revisions']})")
    print("=" * 70)
    print(result["draft"])

    print("\n" + "=" * 70)
    print("  FINAL CRITIQUE")
    print("=" * 70)
    print(result["critique"])

    print("\n" + "=" * 70)
    print("  SOURCES")
    print("=" * 70)
    for s in result["sources"]:
        print(" -", s)


if __name__ == "__main__":
    main()