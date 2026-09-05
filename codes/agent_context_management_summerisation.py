"""
Summarisation-based context management: a 30-turn conversation where the history
is compressed instead of being allowed to grow.

Every turn, before any API call, the request that is about to be sent is measured:

    candidate = history + [this turn's user message]

If that crosses HISTORY_TOKEN_THRESHOLD, the history is split in half by message
count, the *older* half is sent to the model with an instruction to summarise it,
and those messages are replaced by the single summary that comes back. The recent
half is left untouched, word for word, because recent detail is the part the next
answer is most likely to depend on. Checking before the call is what makes the
threshold a real cap: no request in the run ever goes out over budget.

The contrast with `agent_context_management_sliding_window.py` is the whole point.
That script *discards* evicted messages, so anything outside the window is simply
gone. This script *compresses* them, so old information survives in lossy form.
The recall probes near the end -- "what was my name again?", "which book did you
recommend at the start?" -- are the test: the sliding window fails them, this
script should mostly pass them.

Summarisation is not free, and the script does not hide the price. Each compaction
is an extra API call, tracked in its own accumulator and reported in the final
summary alongside the prompt tokens it saved.

Each turn also dumps the request in the flattened wire format the tokenizer
actually sees -- <|start|>user<|message|>Hello<|end|> -- with every message printed
IN FULL rather than previewed, since the question here is precisely which messages
survived compaction and which were replaced by a summary. That is verbose -- roughly
2,500 lines -- so the whole run is also written to OUTPUT_PATH as it prints, leaving a
transcript to page through afterwards without having to redirect the shell.
"""

import contextlib
import sys
import textwrap
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
import tiktoken

load_dotenv()

MODEL = "gpt-4o-mini"

# Compaction fires when the request about to be sent crosses this line. At 1,000 the
# history holds roughly five turns before it has to be compressed, so compaction fires
# every 2-4 turns -- around 8 times across the 30. That is a deliberately tight budget:
# it makes the sawtooth obvious, at the price of summarising the same early turns over
# and over (see the note on SUMMARY_MAX_TOKENS below).
HISTORY_TOKEN_THRESHOLD = 1_000

# One compaction pass shrinks the history a long way but is not a mathematical
# guarantee: [summary] + kept half + this turn's question could in principle still
# land above the line. The slack absorbs a marginal overshoot; tripping it means
# SUMMARY_MAX_TOKENS is too generous for the threshold.
THRESHOLD_SLACK = 64

# Chat messages are flattened into one token stream before inference, roughly:
#   <|start|>user<|message|>Hello<|end|><|start|>assistant<|message|>
TOKENS_PER_MESSAGE = 3        # <|start|> / <|message|> / <|end|>, once per message
TOKENS_PER_REPLY_PRIMING = 3  # trailing <|start|>assistant<|message|>, once per request

INPUT_COST_PER_1M = 0.150   # USD, gpt-4o-mini
OUTPUT_COST_PER_1M = 0.600  # USD, gpt-4o-mini

MAX_TOKENS = 150  # keeps replies uniform so the sawtooth is readable, not noise

# ~200 words of English prose runs 260-280 tokens, so a 250-token cap would clip the
# summary mid-sentence -- and the tail is exactly where the prompt's lowest-priority
# material (the user's goal and constraints) lands. 320 leaves headroom for the prompt's
# "under 200 words" target.
#
# Against a 1,000-token threshold that ceiling is worth seeing clearly: the older half
# being compressed is only ~500 tokens, so a summary that actually ran to 320 would
# compress it barely 1.5x, and the recap alone would occupy a third of the next request.
# Real summaries come in nearer 200 tokens and the run stays comfortably net-positive,
# but this is the dial. If the compaction count climbs much past 8, or the "Xx
# compression" figure in the banners drops toward 1, turn this down before touching the
# threshold.
SUMMARY_MAX_TOKENS = 320

# The summary goes in as a plain `user` message, not a system message -- the main
# conversation pins nothing. A user-role recap is weaker framing than a system one,
# so the prefix is what tells the model to read it as background rather than as a
# fresh question. If the model ever visibly answers the recap instead of the actual
# question, that is a real property of this design, worth noticing rather than hiding.
SUMMARY_PREFIX = (
    "[Conversation summary so far - background context only, "
    "do not reply to this message directly]\n"
)

# Used verbatim as the summariser's system message. The transcript is substituted into
# the trailing placeholder, so the summariser call carries a system message and nothing
# else -- valid, if unusual. If gpt-4o-mini ever responds weakly to a system-only
# request, the fallback is to split this template at "CONVERSATION TO SUMMARISE:" and
# send the instructions as `system` and the transcript as `user`, without changing a
# word of the prompt.
SUMMARY_SYSTEM_PROMPT = """You are compressing the earlier part of a conversation so it can be
removed from an agent's context window while preserving everything that later turns
might depend on.
Write a summary that preserves, in this order of priority:
1. Every concrete figure, name, date, file path, ID and constraint the user stated.
Copy these EXACTLY -- never round, approximate or paraphrase a number.
2. Decisions that were made and the reasons behind them.
3. Open questions and anything explicitly ruled out (so it is not re-proposed).
4. The user's stated goal, preferences and constraints.
Discard: pleasantries, restated questions, and reasoning that led to a stated
conclusion (keep the conclusion). Write compact prose, not bullet points. Do not add
anything that was not in the conversation. Aim for under 200 words.
CONVERSATION TO SUMMARISE:
{transcript}"""

# Unlike the sliding-window sibling, which caps each message at 64 characters, this
# dump prints everything: the exercise here is reading which messages survived.
WIRE_PREVIEW_CHARS = None  # None = no truncation
WIRE_WRAP_WIDTH = 96       # wrap column for message content in the dump

# Where the run is saved. Resolved against this file rather than the working directory,
# so the transcript lands in the same place however the script is launched.
OUTPUT_PATH = (
    Path(__file__).resolve().parent
    / "output"
    / "summarisation"
    / "agent_context_management_summerisation_output.txt"
)

client = OpenAI()  # reads OPENAI_API_KEY from env
encoding = tiktoken.encoding_for_model(MODEL)  # gpt-4o-mini -> o200k_base


class ConversationTurn:
    """One user message in the scripted conversation."""

    def __init__(self, user: str):
        self.user = user


class Tee:
    """Send everything printed to the terminal and to the transcript file at once.

    A 30-turn run takes a couple of minutes and prints ~2,500 lines, so watching it
    happen and keeping the output both matter; a plain redirect would give up the first
    for the second.
    """

    def __init__(self, stream, handle):
        self.stream = stream
        self.handle = handle

    def write(self, text: str) -> int:
        self.stream.write(text)
        self.handle.write(text)
        return len(text)

    def flush(self) -> None:
        self.stream.flush()
        self.handle.flush()


class CompactionStats:
    """What one compaction did, for the banner printed above the turn."""

    def __init__(
        self,
        number: int,
        trigger_tokens: int,
        history_before: int,
        older_count: int,
        recent_count: int,
        older_tokens: int,
        summary_tokens: int,
        usage,
        cost: float,
    ):
        self.number = number
        self.trigger_tokens = trigger_tokens
        self.history_before = history_before
        self.older_count = older_count
        self.recent_count = recent_count
        self.older_tokens = older_tokens
        self.summary_tokens = summary_tokens
        self.usage = usage
        self.cost = cost


def count_tokens(messages: list[dict]) -> int:
    """Estimate the billed prompt tokens for a request, before sending it.

    Identical to the sliding-window sibling's version so the two runs are directly
    comparable -- and it is the same function that watches the history against the
    threshold, so the trigger and the reported cost use one definition, not two.
    """
    total = 0
    for message in messages:
        total += TOKENS_PER_MESSAGE
        total += len(encoding.encode(message["role"]))  # 1 token, NOT free
        total += len(encoding.encode(message["content"]))
    return total + TOKENS_PER_REPLY_PRIMING


def wrap_content(text: str, first_line_offset: int = 0) -> list[str]:
    """Wrap a message for the dump, keeping its own line breaks.

    The sibling collapses each message onto one line and elides it. Here nothing is
    elided: the tokenizer sees those newlines, so showing them keeps the dump faithful
    and makes multi-paragraph replies readable.

    `first_line_offset` is the width of the `<|start|>role<|message|>` opener that will
    be printed ahead of the first line. Reserving it keeps every line of the block
    inside the same right-hand margin instead of letting the opening line overshoot.
    """
    if WIRE_PREVIEW_CHARS is not None and len(text) > WIRE_PREVIEW_CHARS:
        text = text[: WIRE_PREVIEW_CHARS - 1] + "…"

    lines: list[str] = []
    for paragraph in text.split("\n"):
        if not paragraph.strip():
            lines.append("")
            continue
        # Budget the opener out of the first line by wrapping as though it were
        # indentation, then removing it again -- textwrap has no first-line-width knob.
        pad = " " * first_line_offset if not lines else ""
        wrapped = textwrap.wrap(paragraph, width=WIRE_WRAP_WIDTH, initial_indent=pad)
        if pad and wrapped:
            wrapped[0] = wrapped[0][first_line_offset:]
        lines.extend(wrapped)
    return lines or [""]


def render_wire_format(messages: list[dict]) -> str:
    """Show the request as the tokenizer sees it, with the cost of each message.

    The `messages` list is not sent as JSON objects -- it is flattened into a single
    token stream. Printing that stream makes the two overheads in `count_tokens`
    visible: the <|start|>/<|message|>/<|end|> scaffolding paid once per message, and
    the trailing reply priming paid once per request.

    The marker column flags what is not ordinary history:
        >  this turn's new user message
        S  the injected summary (a `user` message that is not a real user turn)
    """
    lines = ["Wire format (what the tokenizer actually sees):", ""]
    for index, message in enumerate(messages):
        tokens = (
            TOKENS_PER_MESSAGE
            + len(encoding.encode(message["role"]))
            + len(encoding.encode(message["content"]))
        )
        if index == len(messages) - 1:
            marker = ">"
        elif is_summary(message):
            marker = "S"
        else:
            marker = " "

        head = f"  {marker} [{tokens:>4} tok] "
        indent = " " * len(head)  # continuation lines align under the message text
        opener = f"<|start|>{message['role']}<|message|>"
        body = wrap_content(message["content"], first_line_offset=len(opener))

        lines.append(f"{head}{opener}{body[0]}")
        for continuation in body[1:]:
            # A blank line inside a message stays blank rather than carrying the indent.
            lines.append(indent + continuation if continuation else "")
        lines[-1] += "<|end|>" if lines[-1] else indent + "<|end|>"
        lines.append("")  # blank line between messages keeps the blocks scannable

    lines.append(
        f"    [{TOKENS_PER_REPLY_PRIMING:>4} tok] <|start|>assistant<|message|>"
        "   <- reply priming; the model speaks next"
    )
    lines.append(f"    [{count_tokens(messages):>4} tok] = estimated prompt total")
    return "\n".join(lines)


def is_summary(message: dict) -> bool:
    """True for the injected recap, which is a `user` message but not a user turn."""
    return message["role"] == "user" and message["content"].startswith(SUMMARY_PREFIX)


def split_history(history: list[dict]) -> tuple[list[dict], list[dict]]:
    """Older half (to be summarised), recent half (kept verbatim).

    The split is by message count, not by token mass. Assistant replies run ~150 tokens
    against ~15-token questions, so "half the messages" is deliberately not "half the
    tokens" -- but the message count is what the per-turn output makes legible, and a
    token-weighted cut would put the printed split point somewhere hard to reason about.

    The pair-alignment guard is the same one `build_window()` uses in the sliding-window
    sibling, for the same reason: a kept half that opens on an `assistant` message shows
    the model a reply whose question was just compressed away. Since history alternates
    user/assistant, an even-length history splits cleanly and the guard is a no-op; it
    fires only on an odd length, which is what a previous compaction leaves behind.
    """
    midpoint = len(history) // 2
    older, recent = history[:midpoint], history[midpoint:]
    if recent and recent[0]["role"] == "assistant":
        older.append(recent.pop(0))  # never open the kept half on a dangling answer
    return older, recent


def flatten_for_summary(messages: list[dict]) -> str:
    """The older half as plain text, for substitution into the summariser prompt."""
    return "\n".join(f"{message['role']}: {message['content']}" for message in messages)


def summarise(older: list[dict]):
    """One extra API call: compress the older half into a single recap message.

    `.format()` is safe here -- the template holds no braces other than {transcript},
    and braces inside the substituted transcript are not re-scanned.
    """
    prompt = SUMMARY_SYSTEM_PROMPT.format(transcript=flatten_for_summary(older))
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": prompt}],  # system-only, by design
        max_tokens=SUMMARY_MAX_TOKENS,
    )

    text = response.choices[0].message.content
    if not text or not text.strip():
        raise RuntimeError(
            "The summariser returned nothing. The request is system-only; if the model "
            "will not answer it, split SUMMARY_SYSTEM_PROMPT at 'CONVERSATION TO "
            "SUMMARISE:' and send the instructions as system, the transcript as user."
        )

    summary_message = {"role": "user", "content": SUMMARY_PREFIX + text.strip()}
    return summary_message, response.usage


def compact(history: list[dict], number: int):
    """Replace the older half of the history with a summary of it.

    On the second and every later compaction the existing summary is already history[0],
    so it lands in the older half and is folded into the new summary rather than sitting
    alongside it. The history is therefore always [at most one summary] + [recent
    verbatim messages]: the prefix never grows, and the token curve stays a genuine
    sawtooth instead of creeping upward under a chain of summaries.

    The cost of that is real, and is the honest lesson of summarisation: turn-1 detail
    is compressed again on every pass. Whether "Sourav" is still in the summary at turn
    27 is the demonstration.
    """
    older, recent = split_history(history)
    kept_before = list(recent)  # the recent half must come through byte-for-byte

    older_tokens = count_tokens(older)
    summary_message, usage = summarise(older)
    summary_tokens = count_tokens([summary_message])

    new_history = [summary_message] + recent

    assert summary_tokens < older_tokens, "compaction made the prompt bigger"
    assert new_history[1:] == kept_before, "the recent half was not preserved verbatim"
    assert sum(is_summary(m) for m in new_history) <= 1, "more than one summary in history"
    assert is_summary(new_history[0]), "the summary is not at index 0"
    assert new_history[0]["role"] == "user", "the summary is not a user message"

    stats = CompactionStats(
        number=number,
        trigger_tokens=0,  # filled in by the caller, which knows what tripped the line
        history_before=len(history),
        older_count=len(older),
        recent_count=len(recent),
        older_tokens=older_tokens,
        summary_tokens=summary_tokens,
        usage=usage,
        cost=estimate_cost(usage.prompt_tokens, usage.completion_tokens),
    )
    return new_history, stats


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


def print_compaction_banner(stats: CompactionStats, history_after: list[dict]) -> None:
    compression = stats.older_tokens / stats.summary_tokens if stats.summary_tokens else 0.0
    print(
        f"*** COMPACTION #{stats.number}: {stats.history_before} history + 1 new ="
        f" {stats.trigger_tokens:,} tokens exceeded threshold {HISTORY_TOKEN_THRESHOLD:,}"
    )
    print(
        f"    Split at message {stats.older_count} (pair-aligned):"
        f" {stats.older_count} older -> summarised,"
        f" {stats.recent_count} kept verbatim"
    )
    print(
        f"    Older half: {stats.older_tokens:,} tokens -> summary:"
        f" {stats.summary_tokens:,} tokens ({compression:.1f}x compression)"
    )
    print(
        f"    Summariser call: {stats.usage.prompt_tokens:,} prompt +"
        f" {stats.usage.completion_tokens:,} completion tokens = ${stats.cost:.6f}"
    )
    print(
        f"    History after: {len(history_after)} messages,"
        f" {count_tokens(history_after):,} tokens"
    )


def run_conversation(conversation_turns: list[ConversationTurn]) -> None:
    history: list[dict] = []          # compacted; this is what actually gets sent
    full_transcript: list[dict] = []  # never compacted, never sent -- accounting only

    compacted_prompt_total = 0
    full_prompt_total = 0
    completion_total = 0
    summary_prompt_total = 0
    summary_completion_total = 0
    running_cost = 0.0
    peak_compacted = 0
    peak_full = 0
    compactions = 0

    for turn_number, turn in enumerate(conversation_turns, start=1):
        user_message = {"role": "user", "content": turn.user}

        print(f"--- Turn {turn_number} ---")

        # The threshold is checked against the request about to be sent, not against
        # the history alone, and it is checked *before* the call. Checking afterwards
        # would let the turn that trips the line go out over budget, enforcing the cap
        # one turn late.
        candidate_tokens = count_tokens(history + [user_message])
        if candidate_tokens > HISTORY_TOKEN_THRESHOLD and len(history) >= 2:
            compactions += 1
            history, stats = compact(history, compactions)
            stats.trigger_tokens = candidate_tokens
            summary_prompt_total += stats.usage.prompt_tokens
            summary_completion_total += stats.usage.completion_tokens
            running_cost += stats.cost
            print_compaction_banner(stats, history)

        # The current user message is appended after compaction, so this turn's
        # question always reaches the model verbatim.
        messages = history + [user_message]
        estimated = count_tokens(messages)
        assert all(m["role"] != "system" for m in messages), "the main call pins nothing"
        assert estimated <= HISTORY_TOKEN_THRESHOLD + THRESHOLD_SLACK, (
            f"request of {estimated} tokens exceeded the cap -- lower SUMMARY_MAX_TOKENS"
        )

        full_transcript_tokens = count_tokens(full_transcript + [user_message])

        text, usage = call_llm(messages)

        compacted_prompt_total += usage.prompt_tokens
        full_prompt_total += full_transcript_tokens
        completion_total += usage.completion_tokens
        turn_cost = estimate_cost(usage.prompt_tokens, usage.completion_tokens)
        running_cost += turn_cost
        peak_compacted = max(peak_compacted, usage.prompt_tokens)
        peak_full = max(peak_full, full_transcript_tokens)

        ratio = full_transcript_tokens / estimated if estimated else 1.0
        drift = estimated - usage.prompt_tokens

        print(
            f"History: {len(history)} messages, {count_tokens(history):,} tokens"
            f" (threshold {HISTORY_TOKEN_THRESHOLD:,}) | compactions so far: {compactions}"
        )
        print(f"Sent:   {len(history)} history + 1 user = {len(messages)} messages")
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

        assistant_message = {"role": "assistant", "content": text}
        history.append(user_message)
        history.append(assistant_message)
        full_transcript.append(user_message)
        full_transcript.append(assistant_message)

    print_summary(
        len(conversation_turns),
        compactions,
        compacted_prompt_total,
        full_prompt_total,
        completion_total,
        summary_prompt_total,
        summary_completion_total,
        running_cost,
        peak_compacted,
        peak_full,
    )


def print_summary(
    turns: int,
    compactions: int,
    compacted_prompt_total: int,
    full_prompt_total: int,
    completion_total: int,
    summary_prompt_total: int,
    summary_completion_total: int,
    running_cost: float,
    peak_compacted: int,
    peak_full: int,
) -> None:
    saved_tokens = full_prompt_total - compacted_prompt_total
    saved_pct = saved_tokens / full_prompt_total * 100 if full_prompt_total else 0.0
    compacted_input_cost = compacted_prompt_total / 1_000_000 * INPUT_COST_PER_1M
    full_input_cost = full_prompt_total / 1_000_000 * INPUT_COST_PER_1M
    summariser_cost = estimate_cost(summary_prompt_total, summary_completion_total)
    net_saving = (full_input_cost - compacted_input_cost) - summariser_cost

    print(f"=========== SUMMARY ({turns} turns) ===========")
    print(f"Compactions    : {compactions:>8,} (summariser calls)")
    print(f"Compacted      : {compacted_prompt_total:>8,} prompt tokens   ${compacted_input_cost:.5f}")
    print(
        f"Full resend    : {full_prompt_total:>8,} prompt tokens   ${full_input_cost:.5f}"
        "   <- what it would have cost"
    )
    print(f"Saved          : {saved_tokens:>8,} prompt tokens   ({saved_pct:.1f}%)")
    # The point of honesty in this script: compaction is not free, so what it cost is
    # reported next to what it saved.
    print(
        f"Summariser cost: {summary_prompt_total:>8,} prompt + {summary_completion_total:>6,}"
        f" completion tokens   ${summariser_cost:.5f}   <- the overhead"
    )
    print(f"Net saving     : ${net_saving:.5f}")
    print(
        f"Peak prompt size, compacted: {peak_compacted:,} tokens (capped)"
        f"   full: {peak_full:,} tokens (growing)"
    )
    print(
        f"Actual spend this run: ${running_cost:.6f}"
        f" (input + {completion_total + summary_completion_total:,} completion tokens)"
    )


if __name__ == "__main__":
    conversation_turns = [
        # The same 30 turns as agent_context_management_sliding_window.py, copied
        # verbatim so the two strategies are comparable turn-for-turn. Turns 1-20
        # themselves come from agent_in_context_memory.py.
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
        # The recall probes. The sliding window fails these because the answers were
        # evicted; here they should survive inside the summary, in lossy form.
        ConversationTurn(user="Can you remind me what my name is?"),
        ConversationTurn(user="What was the first topic we discussed today?"),
        ConversationTurn(user="Summarize our whole conversation in three bullet points."),
        ConversationTurn(user="Thanks, that's all for now."),
        ConversationTurn(user="Actually one more area -- what's the difference between a leader and a follower in Raft?"),
        ConversationTurn(user="How does log replication actually work, step by step?"),
        ConversationTurn(user="What happens if a follower's log diverges from the leader's?"),
        ConversationTurn(user="Is there an equivalent of that in Kafka's replication model?"),
        ConversationTurn(user="What's an ISR in Kafka?"),
        ConversationTurn(user="How does Kafka decide when to shrink the ISR?"),
        ConversationTurn(user="By the way, what was my name again?"),
        # Turn 28 is the honest edge: a summary preserves facts, not phrasing. A
        # paraphrase of the first question here is summarisation working as designed.
        ConversationTurn(user="What was the very first thing I asked you today?"),
        ConversationTurn(user="Which book did you recommend at the start of our conversation?"),
        ConversationTurn(user="Alright, summarize everything we covered today."),
    ]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # The `with` closes the file even if an assertion trips mid-run, so a failed run
    # still leaves the turns that got that far on disk to look at.
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        with contextlib.redirect_stdout(Tee(sys.stdout, handle)):
            run_conversation(conversation_turns)

    print(f"\nTranscript written to {OUTPUT_PATH}")
