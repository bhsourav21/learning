"""
Sliding-window context management: a 30-turn conversation where only the last
10 messages of history are ever sent to the model.

Unlike `agent_in_context_memory.py`, which truncates by *token budget* and pins
a system prompt at index 0, this script truncates by *message count* and pins
nothing at all. Every token in the request is subject to eviction:

    request = [last 10 messages of history] + [current user message]

There is deliberately no system prompt, so the windowed prompt-token curve is
genuinely flat with no fixed per-call prefix inflating it. The full transcript
is still kept in `history` -- only the *view* sent to the model is windowed --
so each turn can also report what resending everything *would* have cost. That
contrast (flat vs. quadratic) is the whole point of the exercise.

Each turn also dumps the request in the flattened wire format the tokenizer
actually sees -- <|start|>user<|message|>Hello<|end|> -- annotated with the token
cost of every message, so the per-message and per-request overheads in
`count_tokens` are visible rather than theoretical.

The tradeoff is visible on screen: the recall probes near the end ("what was my
name again?") fall outside the window and the model cannot answer them. That is
the demonstration, not a bug.
"""

from dotenv import load_dotenv
from openai import OpenAI
import tiktoken

load_dotenv()

MODEL = "gpt-4o-mini"
WINDOW_SIZE = 10

# Chat messages are flattened into one token stream before inference, roughly:
#   <|start|>user<|message|>Hello<|end|><|start|>assistant<|message|>
TOKENS_PER_MESSAGE = 3        # <|start|> / <|message|> / <|end|>, once per message
TOKENS_PER_REPLY_PRIMING = 3  # trailing <|start|>assistant<|message|>, once per request

INPUT_COST_PER_1M = 0.150   # USD, gpt-4o-mini
OUTPUT_COST_PER_1M = 0.600  # USD, gpt-4o-mini

MAX_TOKENS = 150  # keeps replies uniform so the token curve is signal, not noise

# Per-message content preview length in the wire-format dump. Set to None to print
# every message in full (verbose: ~11 messages of up to 150 tokens each, per turn).
WIRE_PREVIEW_CHARS = 64

client = OpenAI()  # reads OPENAI_API_KEY from env
encoding = tiktoken.encoding_for_model(MODEL)  # gpt-4o-mini -> o200k_base


class ConversationTurn:
    """One user message in the scripted conversation."""

    def __init__(self, user: str):
        self.user = user


def count_tokens(messages: list[dict]) -> int:
    """Estimate the billed prompt tokens for a request, before sending it."""
    total = 0
    for message in messages:
        total += TOKENS_PER_MESSAGE
        total += len(encoding.encode(message["role"]))  # 1 token, NOT free
        total += len(encoding.encode(message["content"]))
    return total + TOKENS_PER_REPLY_PRIMING


def _preview(text: str) -> str:
    """Collapse a message onto one line so the wire-format dump stays scannable."""
    flat = " ".join(text.split())
    if WIRE_PREVIEW_CHARS is None or len(flat) <= WIRE_PREVIEW_CHARS:
        return flat
    return flat[: WIRE_PREVIEW_CHARS - 1] + "\u2026"


def render_wire_format(messages: list[dict]) -> str:
    """Show the request as the tokenizer sees it, with the cost of each message.

    The `messages` list is not sent as JSON objects -- it is flattened into a single
    token stream. Printing that stream makes the two overheads in `count_tokens`
    visible: the <|start|>/<|message|>/<|end|> scaffolding paid once per message,
    and the trailing reply priming paid once per request.
    """
    lines = ["Wire format (what the tokenizer actually sees):"]
    for index, message in enumerate(messages):
        tokens = (
            TOKENS_PER_MESSAGE
            + len(encoding.encode(message["role"]))
            + len(encoding.encode(message["content"]))
        )
        marker = ">" if index == len(messages) - 1 else " "  # ">" = this turn's new message
        lines.append(
            f"  {marker} [{tokens:>4} tok] <|start|>{message['role']}<|message|>"
            f"{_preview(message['content'])}<|end|>"
        )
    lines.append(
        f"    [{TOKENS_PER_REPLY_PRIMING:>4} tok] <|start|>assistant<|message|>"
        "   <- reply priming; the model speaks next"
    )
    lines.append(f"    [{count_tokens(messages):>4} tok] = estimated prompt total")
    return "\n".join(lines)


def build_window(history: list[dict], window_size: int = WINDOW_SIZE) -> list[dict]:
    """The sliding window: the last `window_size` messages, never opening on an answer.

    A window that starts on an `assistant` message shows the model a reply whose
    question has been evicted -- a dangling non-sequitur, with nothing pinned above
    it to provide framing. Since history strictly alternates user/assistant, an
    even-sized slice lands on `assistant` from turn 6 onward, so the effective
    window oscillates 10 -> 9. That is expected.
    """
    window = history[-window_size:]
    if window and window[0]["role"] == "assistant":
        window = window[1:]  # 10 -> 9; never opens on a dangling answer

    assert len(window) <= window_size, "window exceeded its size cap"
    assert not window or window[0]["role"] != "assistant", "window opens on an answer"
    return window


def estimate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    """USD for one call at gpt-4o-mini rates."""
    return (
        prompt_tokens / 1_000_000 * INPUT_COST_PER_1M
        + completion_tokens / 1_000_000 * OUTPUT_COST_PER_1M
    )


def call_llm(messages: list[dict]):
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=MAX_TOKENS,
    )
    return response.choices[0].message.content, response.usage


def describe_oldest(history: list[dict], window: list[dict]) -> str:
    """Label the oldest surviving message, e.g. 'turn 10 user'."""
    if not window:
        return "nothing yet"
    index = len(history) - len(window)
    return f"turn {index // 2 + 1} {history[index]['role']}"


def run_conversation(conversation_turns: list[ConversationTurn]) -> None:
    history: list[dict] = []  # keeps ALL turns; only the *view* is windowed

    windowed_prompt_total = 0
    full_prompt_total = 0
    completion_total = 0
    running_cost = 0.0
    peak_windowed = 0
    peak_full = 0

    for turn_number, turn in enumerate(conversation_turns, start=1):
        user_message = {"role": "user", "content": turn.user}

        window = build_window(history)
        messages = window + [user_message]  # no prefix, nothing pinned
        assert all(m["role"] != "system" for m in messages), "nothing may be pinned"

        estimated = count_tokens(messages)
        full_transcript_tokens = count_tokens(history + [user_message])

        text, usage = call_llm(messages)

        windowed_prompt_total += usage.prompt_tokens
        full_prompt_total += full_transcript_tokens
        completion_total += usage.completion_tokens
        turn_cost = estimate_cost(usage.prompt_tokens, usage.completion_tokens)
        running_cost += turn_cost
        peak_windowed = max(peak_windowed, usage.prompt_tokens)
        peak_full = max(peak_full, full_transcript_tokens)

        ratio = full_transcript_tokens / estimated if estimated else 1.0
        drift = estimated - usage.prompt_tokens

        print(f"--- Turn {turn_number} ---")
        print(
            f"Window: {len(window)} messages (oldest kept: {describe_oldest(history, window)})"
            f" | evicted so far: {len(history) - len(window)}"
        )
        print(f"Sent:   {len(window)} window + 1 user = {len(messages)} messages")
        print(render_wire_format(messages))
        print(
            f"Tokens: est {estimated} | actual prompt {usage.prompt_tokens}"
            f" (drift {drift:+d}) | completion {usage.completion_tokens}"
            f" | total {usage.total_tokens}"
        )
        print(f"Cost:   ${turn_cost:.6f} this turn | ${running_cost:.6f} running")
        print(
            f"If full transcript had been sent: {full_transcript_tokens:,} prompt tokens"
            f" ({ratio:.1f}x)"
        )
        print(f"User: {turn.user}")
        print(f"Assistant: {text}\n")

        history.append(user_message)
        history.append({"role": "assistant", "content": text})

    print_summary(
        len(conversation_turns),
        windowed_prompt_total,
        full_prompt_total,
        completion_total,
        running_cost,
        peak_windowed,
        peak_full,
    )


def print_summary(
    turns: int,
    windowed_prompt_total: int,
    full_prompt_total: int,
    completion_total: int,
    running_cost: float,
    peak_windowed: int,
    peak_full: int,
) -> None:
    saved_tokens = full_prompt_total - windowed_prompt_total
    saved_pct = saved_tokens / full_prompt_total * 100 if full_prompt_total else 0.0
    windowed_input_cost = windowed_prompt_total / 1_000_000 * INPUT_COST_PER_1M
    full_input_cost = full_prompt_total / 1_000_000 * INPUT_COST_PER_1M

    print(f"=========== SUMMARY ({turns} turns) ===========")
    print(f"Windowed   : {windowed_prompt_total:>8,} prompt tokens   ${windowed_input_cost:.5f}")
    print(
        f"Full resend: {full_prompt_total:>8,} prompt tokens   ${full_input_cost:.5f}"
        "   <- what it would have cost"
    )
    print(f"Saved      : {saved_tokens:>8,} prompt tokens   ({saved_pct:.1f}%)")
    print(
        f"Peak prompt size, windowed: {peak_windowed:,} tokens (flat)"
        f"   full: {peak_full:,} tokens (growing)"
    )
    print(
        f"Actual spend this run: ${running_cost:.6f}"
        f" (input + {completion_total:,} completion tokens)"
    )


if __name__ == "__main__":
    conversation_turns = [
        # Turns 1-20 are reused verbatim from agent_in_context_memory.py so the
        # memory scripts stay directly comparable.
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
        # Turns 21-30 extend the run. The recall probes sit past turn 25, where the
        # answers have long fallen outside a 10-message window -- the model failing
        # them is the demonstration.
        ConversationTurn(user="Actually one more area -- what's the difference between a leader and a follower in Raft?"),
        ConversationTurn(user="How does log replication actually work, step by step?"),
        ConversationTurn(user="What happens if a follower's log diverges from the leader's?"),
        ConversationTurn(user="Is there an equivalent of that in Kafka's replication model?"),
        ConversationTurn(user="What's an ISR in Kafka?"),
        ConversationTurn(user="How does Kafka decide when to shrink the ISR?"),
        ConversationTurn(user="By the way, what was my name again?"),
        ConversationTurn(user="What was the very first thing I asked you today?"),
        ConversationTurn(user="Which book did you recommend at the start of our conversation?"),
        ConversationTurn(user="Alright, summarize everything we covered today."),
    ]
    run_conversation(conversation_turns)
