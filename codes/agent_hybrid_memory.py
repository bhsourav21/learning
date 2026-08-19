"""
Hybrid memory: combines the two previous strategies. The last 5 messages
are kept verbatim in-context (sliding window, like agent_in_context_memory.py).
Anything older is embedded and stored in Pinecone, and on each turn the top-3
semantically relevant older turns are retrieved (like agent_external_memory.py).
Every call to the model sees: system prompt + top-3 retrieved older turns
(if any) + the last 5 in-context messages + the current user message.
"""

from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()

MODEL = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"
SYSTEM_PROMPT = "You are a helpful assistant."

INDEX_NAME = "bhsourav17-pinecone-index"
NAMESPACE = "agent-hybrid-memory-demo"
TOP_K = 3
WINDOW_SIZE = 5  # number of most recent messages kept verbatim in-context

openai_client = OpenAI()
pc = Pinecone()  # reads PINECONE_API_KEY from env automatically
index = pc.Index(INDEX_NAME)


@dataclass
class ConversationTurn:
    user: str


def get_embedding(text: str) -> list[float]:
    response = openai_client.embeddings.create(input=[text], model=EMBED_MODEL)
    return response.data[0].embedding


def retrieve_relevant_turns(query_embedding: list[float], top_k: int = TOP_K) -> list[dict]:
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        namespace=NAMESPACE,
    )
    return [match["metadata"] for match in results.matches]


def store_turn(turn_index: int, role: str, text: str, embedding: list[float]) -> None:
    index.upsert(
        vectors=[{
            "id": f"turn-{turn_index}-{role}",
            "values": embedding,
            "metadata": {"role": role, "text": text, "turn_index": turn_index},
        }],
        namespace=NAMESPACE,
    )


def call_llm(messages: list[dict]) -> str:
    response = openai_client.chat.completions.create(model=MODEL, messages=messages)
    return response.choices[0].message.content


def run_conversation(conversation_turns: list[ConversationTurn]) -> None:
    # Clean slate so re-running the demo doesn't mix in stale vectors
    try:
        index.delete(delete_all=True, namespace=NAMESPACE)
    except Exception:
        pass

    # In-context window: holds the WINDOW_SIZE most recent messages verbatim,
    # e.g. [{"turn_index": 18, "role": "user", "content": ...}, ...]
    window: list[dict] = []
    # Nothing is archived until the window first overflows, so querying
    # Pinecone before that point is guaranteed to return nothing -- skip it
    archive_has_data = False

    for turn_index, turn in enumerate(conversation_turns, start=1):
        # Embed the current user message so it can be used both to search
        # Pinecone below and (reused, no extra API call) to store this turn later
        user_embedding = get_embedding(turn.user)

        # Semantic search covers only what has already been evicted from the
        # window -- e.g. at turn 17 ("remind me my name?") the window holds
        # turns 15-17 only, so turn 1 ("my name is Sourav") must be retrieved
        retrieved = retrieve_relevant_turns(user_embedding) if archive_has_data else []

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Retrieved memory is per-turn dynamic data, so it's folded into the
        # user message rather than a second system message, same as external memory
        user_content = turn.user
        if retrieved:
            memory_context = "\n".join(
                f"- ({m['role']}) {m['text']}" for m in retrieved
            )
            user_content = (
                f"Relevant past conversation turns (retrieved via semantic search):\n{memory_context}\n\n"
                f"Current message: {turn.user}"
            )

        # The last WINDOW_SIZE messages ride along verbatim, between the
        # retrieved memory and the current turn -- no retrieval needed for these
        messages.extend({"role": m["role"], "content": m["content"]} for m in window)
        messages.append({"role": "user", "content": user_content})

        response = call_llm(messages)

        print(f"--- Turn {turn_index} ---")
        if retrieved:
            print("Retrieved memory:")
            for m in retrieved:
                print(f"  [{m['turn_index']}:{m['role']}] {m['text']}")
        print(f"User: {turn.user}\nAssistant: {response}\n")

        # Grow the window with the RAW turn (not the retrieval-augmented version --
        # that augmentation is a call-time artifact, not part of the transcript)
        window.append({"turn_index": turn_index, "role": "user", "content": turn.user})
        window.append({"turn_index": turn_index, "role": "assistant", "content": response})

        # Evict oldest messages past WINDOW_SIZE, archiving each to Pinecone so
        # it remains reachable later via semantic search instead of vanishing.
        # The evicted user message reuses user_embedding only if it's this
        # turn's own message; anything older needs a fresh embedding call.
        while len(window) > WINDOW_SIZE:
            evicted = window.pop(0)
            if evicted["turn_index"] == turn_index and evicted["role"] == "user":
                evicted_embedding = user_embedding
            else:
                evicted_embedding = get_embedding(evicted["content"])
            store_turn(evicted["turn_index"], evicted["role"], evicted["content"], evicted_embedding)
            archive_has_data = True


if __name__ == "__main__":
    conversation_turns = [
        ConversationTurn(user="Hi, my name is Sourav."),
        ConversationTurn(user="What's a good book on distributed systems?"),
        ConversationTurn(user="Why did you pick that one specifically?"),
        ConversationTurn(user="I mostly work in Python, does that matter?"),
        ConversationTurn(user="What's the difference between Paxos and Raft?"),
        ConversationTurn(user="Which one is easier to implement from scratch?"),
        ConversationTurn(user="Any good repos with a reference Raft implementation?"),
        ConversationTurn(user="Switching topics -- what's a CAP theorem in one sentence?"),
        ConversationTurn(user="Give me an example of a system that favors AP over CP."),
        ConversationTurn(user="And one that favors CP over AP?"),
        ConversationTurn(user="How does that tie back to Raft's leader election?"),
        ConversationTurn(user="What happens during a network partition in Raft?"),
        ConversationTurn(user="Can a split-brain scenario happen in Raft?"),
        ConversationTurn(user="What's the role of the term number in Raft?"),
        ConversationTurn(user="How is that different from Paxos's ballot numbers?"),
        ConversationTurn(user="Okay, let's go back to the book recommendation -- what was it called again?"),
        ConversationTurn(user="Can you remind me what my name is?"),
        ConversationTurn(user="What was the first topic we discussed today?"),
        ConversationTurn(user="Summarize our whole conversation in three bullet points."),
        ConversationTurn(user="Thanks, that's all for now."),
    ]
    run_conversation(conversation_turns)
