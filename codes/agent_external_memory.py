"""
External memory: instead of resending the whole transcript, every turn
(user + assistant) is embedded and upserted into a Pinecone index. On each
new turn, the user's message is embedded and used to semantically retrieve
the 3 most relevant past turns, which are injected into the prompt as
context alongside the system prompt and the current message.
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
NAMESPACE = "agent-external-memory-demo"
TOP_K = 3

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

    # e.g. turn_index=15, turn.user="How is that different from Paxos's ballot numbers?"
    for turn_index, turn in enumerate(conversation_turns, start=1):
        # Embed the current user message so it can be used both to search
        # Pinecone below and (reused, no extra API call) to store this turn later
        user_embedding = get_embedding(turn.user)

        # Semantic search, not recency: may surface turn 5 ("Paxos vs Raft") and
        # turn 14 ("term number in Raft") over the more recent turn 8 (CAP theorem)
        retrieved = retrieve_relevant_turns(user_embedding)

        # Start a FRESH messages list every turn -- unlike the in-context version,
        # there is no growing transcript; only system prompt + retrieved memory + current message
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Retrieved memory is per-turn dynamic data, not a stable instruction,
        # so it's folded into the user message rather than a second system message
        user_content = turn.user
        if retrieved:
            # Format the 3 retrieved turns as a bullet list, e.g.:
            #   - (assistant) Paxos and Raft both solve consensus...
            #   - (user) What's the role of the term number in Raft?
            memory_context = "\n".join(
                f"- ({m['role']}) {m['text']}" for m in retrieved
            )
            user_content = (
                f"Relevant past conversation turns (retrieved via semantic search):\n{memory_context}\n\n"
                f"Current message: {turn.user}"
            )
        messages.append({"role": "user", "content": user_content})

        # Sends [system, current-user-message-with-memory-context] to gpt-4o-mini
        response = call_llm(messages)

        print(f"--- Turn {turn_index} ---")
        if retrieved:
            # Log exactly what memory the model saw, e.g. "[14:user] What's the role..."
            print("Retrieved memory:")
            for m in retrieved:
                print(f"  [{m['turn_index']}:{m['role']}] {m['text']}")
        print(f"User: {turn.user}\nAssistant: {response}\n")

        # Persist this turn's user + assistant messages as external memory so
        # they become searchable for all future turns
        store_turn(turn_index, "user", turn.user, user_embedding)
        assistant_embedding = get_embedding(response)
        store_turn(turn_index, "assistant", response, assistant_embedding)


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
