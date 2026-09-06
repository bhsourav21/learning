"""
Hybrid context management: pinned facts + a rolling summary + a verbatim window.

This is the three-layer configuration most production agents converge on, and each
layer exists because the other two cannot do its job:

    request = [pinned facts] + [rolling summary] + [recent window] + [this question]

    pinned facts  -- names, IDs, paths, numbers, constraints.  Must be EXACT.
    rolling summary -- the gist of the distant past.  Must be CHEAP, never grows.
    recent window -- the last few turns word for word.  Must be COMPLETE.

Both siblings evict; they differ only in what they do on the way out.
`agent_context_management_sliding_window.py` *discards* what falls out of the window.
`agent_context_management_summerisation.py` *compresses* it. This script does neither
in isolation: an evicted message passes two gates, and each may keep a piece of it.

    window overflows
          |
    GATE 1  extract_facts(evicted)   -> PINNED   (exact, copied verbatim)
          |  then, always
    GATE 2  summarise(summary + evicted) -> SUMMARY (lossy, capped)
          |
     messages dropped

Gate 1 runs before Gate 2, and that ordering is the whole design -- anything that must
be exact is lifted out *before* the lossy step ever sees it, so a name never depends on
the summariser choosing to keep it. The pure-summarisation run is why: after eight
re-folds "Designing Data-Intensive Applications" survived as a title but its
attribution inverted, and the turn-29 probe answered accordingly. Facts survived;
provenance did not. Gate 1 is the fix, and `compact()` asserts the ordering rather than
leaving it to a comment.

Pinned facts are stored in a dict keyed by a short lowercase key, so re-emitting a key
*replaces* it. That is the answer to staleness: "we're using Postgres" can later be
superseded by "actually, SQLite" without the block filling with contradictions.

What this buys, stated up front so the run can be checked against it rather than
rationalised afterwards: recall should improve markedly (the name and book probes
answer from the pinned block, exactly, however many times the summary has been
re-folded), and cost should improve barely at all -- the hybrid pays two gate calls per
compaction where the summarisation sibling pays one, so at this length it may well come
out level with it or dearer.

Measured, over two runs: recall held, exactly and identically both times. Cost came out
better than predicted but still small -- 68.9% off the prompt tokens, $0.00234 of that
handed back to the two gates, $0.0037 net over 30 turns. The prediction was pessimistic
because REQUEST_TOKEN_BUDGET came down to 1,000 (see below) and Gate 1's replies
collapsed to 38-74 completion tokens once its prompt stopped inviting prose. Four
tenths of a cent is still not why anyone runs this configuration, which is the point:
it earns its keep when the conversation is long enough for the quadratic baseline to
dominate, or when a wrong answer costs more than the tokens saved. The script's job is
to make that trade visible, not to win a benchmark.

A note on Gate 1 being an LLM call at all: for this scripted conversation a regex over
"my name is X" would pin the name for free, and most of the overhead here is that call.
The LLM call is the general solution -- it also caught the book title, its attribution
and the user's stated language, none of which a regex would know to look for -- but on a
narrow, known domain the cheap version is the right one.

The per-turn wire dump prints every message IN FULL, as in the summarisation sibling,
since the question is precisely which layer each surviving token is in. That is verbose,
so the whole run is also written to OUTPUT_PATH as it prints.
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

# The same budget as the summarisation sibling's HISTORY_TOKEN_THRESHOLD, so the two runs
# are directly comparable turn for turn.
#
# This was planned at 1,400 on the reasoning that the hybrid carries a fixed prefix that
# script does not -- pinned (<=120) + summary (<=200) ~= 320 tokens -- and that a 1,000
# budget would leave the window ~660 tokens, halving to four messages: two turns, which
# would break the layer whose entire job is to be complete. Measuring it changed the
# arithmetic. Summaries come back at ~140 tokens rather than the 200 they are allowed,
# so the real prefix is ~245, not 320:
#
#   1,000 - 245 (prefix) - 16 (question) - 3 (priming) ~= 736 tokens
#   measured: 10 messages = ~727 tok, 6 messages = ~503 tok
#                                                      -> fills to ~10, halves to 6
#
# Three verbatim turns, which is what the window already keeps -- the layer the 1,400
# budget was protecting is not the thing that was at risk. What 1,400 cost instead was
# the budget itself: peak requests measured 1,079-1,084, so the clause never once fired
# and every compaction came from WINDOW_MAX_MESSAGES. A ceiling that cannot be reached
# is not a ceiling, and this script exists to make the mechanism visible.
REQUEST_TOKEN_BUDGET = 1_000

# One compaction pass shrinks the request a long way but is not a mathematical
# guarantee. The slack absorbs a marginal overshoot; tripping it means the two block
# caps are too generous for the budget.
BUDGET_SLACK = 64

# The sliding window's rule surviving into the hybrid: even under budget, the verbatim
# window never exceeds a fixed message count -- the window stays bounded by *structure*
# and not only by tokens.
#
# 10 is the sliding-window sibling's WINDOW_SIZE, so the verbatim layer here is the same
# size as that script's entire context and the two are directly comparable. The window
# grows two messages per turn, so this clause trips at 12 and halves to 6.
#
# The plan sized this at 16 expecting the token budget to bite first, but measured turns
# cost ~130 tokens rather than the ~168 the arithmetic assumed, so 18 messages still
# fitted under 1,400 and the cap was the *only* trigger that ever fired. At 10 against a
# 1,000 budget the two clauses share the work -- measured, 5 of 8 compactions fire on the
# budget and 3 on this cap -- which is the arrangement worth demonstrating: neither
# clause is decoration.
WINDOW_MAX_MESSAGES = 10

# If the pinned block outgrows this, the largest fact is dropped (see enforce_cap). If
# that eviction fires often, the extractor is pinning things that are not really facts
# -- that is a signal about the prompt, not a reason to raise the cap.
PINNED_MAX_TOKENS = 120

# Down from 320 in the pure-summarisation script. This is the layer paying for the
# pinned block: exactness is now Gate 1's job, so the summary is told not to repeat
# known facts, its target drops from "under 200 words" to "under 120", and its cap
# drops with it. The three layers together should cost about what the summary alone
# cost before.
SUMMARY_MAX_TOKENS = 200
FACT_MAX_TOKENS = 200

# A single fact's value may not exceed this. Measured, every legitimate fact ran 1-22
# tokens (`name: Sourav` is 3, a book title with two authors 22) while the repository
# *descriptions* that caused every observed cap overflow ran 33-39. The two populations
# barely overlap, so a length guard separates them deterministically, and the guard is
# what makes the block's contents predictable rather than a matter of how the extractor
# felt that call.
FACT_MAX_VALUE_TOKENS = 30

# Neither block is a system message: the main conversation pins nothing, so the token
# curve has no fixed system prefix inflating it and stays comparable with both siblings.
# The prefixes are what tell the model how to read them. The accepted consequence is
# that a request can open with THREE consecutive `user` messages (pinned, summary, then
# the first surviving real turn). Chat Completions permits this. If the model ever
# visibly answers a block instead of the question, that is worth reporting, not hiding.
PINNED_PREFIX = (
    "[Known facts - background context only, "
    "do not reply to this message directly]\n"
)
SUMMARY_PREFIX = (
    "[Conversation summary so far - background context only, "
    "do not reply to this message directly]\n"
)

# Gate 1. Three things this prompt is doing deliberately:
#   - `key: value` lines, because facts are merged into a dict by key, so re-emitting a
#     key supersedes it rather than appending a contradiction;
#   - {known} is fed back in, so the extractor emits a line only to add or update;
#   - "Record WHO stated or recommended something" is aimed straight at the attribution
#     drift the pure-summarisation run exhibited on the turn-29 probe.
#
# The "reuse a key ONLY to supersede" rule was added after the first measured run, which
# failed the turn-29 probe: turn 2 recommended three books and turn 4 a fourth, the
# extractor keyed all four as `book recommended`, and merge-by-key kept the last one --
# so the model confidently named a book it had suggested much later. Merging by key
# handles supersession ACROSS batches; it silently destroys distinct facts that collide
# WITHIN one, and the extractor has to be told which case it is looking at. (It already
# did this unprompted for repositories, keying them `repository 1` and `repository 2`.)
FACT_SYSTEM_PROMPT = """You are extracting durable facts from part of a conversation that is about
to be deleted from an agent's context window.
Output ONLY facts a later turn could not be answered correctly without:
- what the user told you about themselves: name, language, role, stated preferences,
  hard constraints;
- decisions the user made or explicitly accepted;
- when the user asked for a recommendation, the single top thing the assistant
  recommended, and WHO recommended it;
- identifiers that must be exact: file paths, URLs, versions, numbers, dates the USER
  stated.
NEVER output any of the following, however concrete it looks:
- explanations, definitions or any sentence describing how something works;
- quotes or sentences copied out of the assistant's answers;
- things the assistant merely listed, mentioned in passing, or offered as alternatives;
- your own knowledge cutoff or training date;
- topics that were discussed.
Rules:
- One fact per line, formatted exactly as  key: value
- Use a short lowercase key (for example: name, language, book recommended).
- Copy values EXACTLY as stated. Never round, paraphrase or infer.
- If a fact supersedes something already known, emit it again with the SAME key.
- Reuse a key ONLY to supersede a fact listed under KNOWN FACTS. Every key in that
  list is already taken: to record a NEW fact, pick a key that is not in it.
- If several DISTINCT items of the same kind must be kept, give each its own numbered
  key (for example: book recommended 1, book recommended 2).
- Prefer emitting nothing to emitting something that is merely interesting. If nothing
  here must be exact, output the single word NONE.
KNOWN FACTS ALREADY PINNED:
{known}
CONVERSATION EXCERPT:
{transcript}"""

# Gate 2. Adapted, not verbatim, from the summarisation sibling: its job has changed.
# The pinned block now owns every exact detail, so asking the summary to also preserve
# "every concrete figure, name, date, file path, ID and constraint ... copy these
# EXACTLY" would duplicate the pinned block inside the summary and pay for it twice.
SUMMARY_SYSTEM_PROMPT = """You are compressing the earlier part of a conversation so it can be
removed from an agent's context window.
Exact details (names, numbers, identifiers, paths, constraints) are ALREADY preserved
separately and are listed below as KNOWN FACTS -- do not repeat them.
Write a summary that captures, in this order of priority:
1. Decisions that were made and the reasons behind them.
2. Open questions and anything explicitly ruled out (so it is not re-proposed).
3. The topics covered, in the order they came up.
Discard: pleasantries, restated questions, and reasoning that led to a stated conclusion
(keep the conclusion). Write compact prose, not bullet points. Do not add anything that
was not in the conversation. Aim for under 120 words.
KNOWN FACTS (already preserved -- do not repeat):
{known}
CONVERSATION TO SUMMARISE:
{transcript}"""

# Chat messages are flattened into one token stream before inference, roughly:
#   <|start|>user<|message|>Hello<|end|><|start|>assistant<|message|>
TOKENS_PER_MESSAGE = 3        # <|start|> / <|message|> / <|end|>, once per message
TOKENS_PER_REPLY_PRIMING = 3  # trailing <|start|>assistant<|message|>, once per request

INPUT_COST_PER_1M = 0.150   # USD, gpt-4o-mini
OUTPUT_COST_PER_1M = 0.600  # USD, gpt-4o-mini

MAX_TOKENS = 150  # keeps replies uniform so the sawtooth is readable, not noise

# As in the summarisation sibling: nothing is elided, because the exercise is reading
# which messages survived and in which layer.
WIRE_PREVIEW_CHARS = None  # None = no truncation
WIRE_WRAP_WIDTH = 96       # wrap column for message content in the dump

# Resolved against this file rather than the working directory, so the transcript lands
# in the same place however the script is launched.
OUTPUT_PATH = (
    Path(__file__).resolve().parent
    / "output"
    / "hybrid"
    / "agent_context_management_hybrid_output.txt"
)

client = OpenAI()  # reads OPENAI_API_KEY from env
encoding = tiktoken.encoding_for_model(MODEL)  # gpt-4o-mini -> o200k_base


class ConversationTurn:
    """One user message in the scripted conversation."""

    def __init__(self, user: str):
        self.user = user


class Tee:
    """Send everything printed to the terminal and to the transcript file at once.

    A 30-turn run takes a couple of minutes and prints thousands of lines, so watching
    it happen and keeping the output both matter; a plain redirect would give up the
    first for the second.
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


class PinnedFacts:
    """Exact facts, merged by key so a later statement supersedes an earlier one.

    The merge is the point. An append-only pinned block would accumulate "database:
    Postgres" and "database: SQLite" side by side and leave the model to pick between
    them; keying by `database` means the second line simply replaces the first. The cost
    is that the extractor has to reuse keys, which is why the prompt says so twice and
    feeds the current keys back in as {known}.
    """

    def __init__(self):
        self.facts: dict[str, str] = {}      # key -> value, insertion-ordered
        self.last_seen: dict[str, int] = {}  # key -> turn number, for cap eviction

    def update(self, pairs: list[tuple[str, str]], turn_number: int) -> tuple[int, int]:
        """Merge extracted pairs in. Returns (added, replaced).

        Both counts are measured against the block as it stood *before* this batch, and
        each key is counted at most once. That matters because the two ways a key can
        repeat are not the same event: one batch superseding an earlier batch's value is
        the staleness mechanism working, while one batch emitting the same key twice is
        the extractor failing to distinguish two facts -- and the first run counted the
        second as five "replaced", which read as the mechanism working when it was not.
        """
        before = dict(self.facts)
        added: set[str] = set()
        replaced: set[str] = set()
        for key, value in pairs:
            # A numbered key is an enumeration slot, not an attribute: `book recommended 1`
            # names one particular book, so a later batch arriving with a different value
            # for it is not superseding that book, it is a fresh item whose numbering
            # restarted at 1. Bare keys (`name`, `database`) keep supersede semantics --
            # they are the case the merge-by-key rule was designed for. Without this,
            # eviction can DESTROY a correct pinned fact, which is the one thing the
            # pinned layer exists to prevent; a measured run overwrote the turn-2 book
            # recommendation with a system named twenty turns later.
            if key in before and before[key] != value and key[-1].isdigit():
                key = self._next_free_slot(key)
            if key in before:
                # A re-emitted key with an identical value is the extractor restating
                # rather than superseding; it still counts as a touch for the LRU cap,
                # but it is not a replacement and should not be reported as one.
                if before[key] != value:
                    replaced.add(key)
            else:
                added.add(key)
            self.facts[key] = value
            self.last_seen[key] = turn_number
        return len(added), len(replaced)

    def _next_free_slot(self, key: str) -> str:
        """`book recommended 1` -> the lowest `book recommended N` not already taken."""
        stem = key.rstrip("0123456789").rstrip()
        number = 2
        while f"{stem} {number}" in self.facts:
            number += 1
        return f"{stem} {number}"

    def enforce_cap(self) -> list[str]:
        """Drop the largest facts until the block fits. Returns what went.

        The plan specified least-recently-updated, and the first runs implemented it.
        Measurement said that is backwards. The block only ever overflows because one or
        two bloated entries crowd out many small ones, so evicting by write time throws
        away short, stable facts to make room for long, junk ones. Worse, a fact that
        never needs updating is permanently the oldest thing in the block: `name:
        Sourav`, written once at the first compaction, was first out every time the cap
        fired. At compaction #3 of a measured run that policy discarded `name` and three
        book titles to keep one paragraph about etcd, and the turn-17, 27 and 29 probes
        all failed as a result.

        Size-first eviction inverts that: what goes is whatever was bloated enough to
        cause the overflow. Ties fall back to the least recently updated, so the old rule
        still breaks a draw between two equally sized facts.
        """
        dropped: list[str] = []
        while self.facts and self.token_count() > PINNED_MAX_TOKENS:
            biggest = max(
                self.facts,
                key=lambda key: (len(encoding.encode(self.facts[key])), -self.last_seen[key]),
            )
            del self.facts[biggest]
            del self.last_seen[biggest]
            dropped.append(biggest)
        return dropped

    def as_text(self) -> str:
        """The {known} block, shared by both gate prompts."""
        return "\n".join(f"{key}: {value}" for key, value in self.facts.items())

    def as_message(self) -> dict | None:
        """The pinned block as a request message, or None while there is nothing to pin."""
        if not self.facts:
            return None
        return {"role": "user", "content": PINNED_PREFIX + self.as_text()}

    def token_count(self) -> int:
        """What the pinned block costs in a request, scaffolding included."""
        message = self.as_message()
        return block_tokens([message]) if message else 0


class CompactionStats:
    """What one compaction did, for the banner printed above the turn."""

    def __init__(
        self,
        number: int,
        window_before: int,
        older_count: int,
        kept_count: int,
        older_tokens: int,
        summary_tokens: int,
        facts_added: int,
        facts_replaced: int,
        facts_dropped: list[str],
        facts_rejected: int,
        pinned_count: int,
        pinned_tokens: int,
        window_tokens: int,
        fact_usage,
        summary_usage,
        gate_order: list[str],
    ):
        self.number = number
        self.window_before = window_before
        self.older_count = older_count
        self.kept_count = kept_count
        self.older_tokens = older_tokens
        self.summary_tokens = summary_tokens
        self.facts_added = facts_added
        self.facts_replaced = facts_replaced
        self.facts_dropped = facts_dropped
        self.facts_rejected = facts_rejected
        self.pinned_count = pinned_count
        self.pinned_tokens = pinned_tokens
        self.window_tokens = window_tokens
        self.fact_usage = fact_usage
        self.summary_usage = summary_usage
        self.gate_order = gate_order
        self.fact_cost = estimate_cost(fact_usage.prompt_tokens, fact_usage.completion_tokens)
        self.summary_cost = estimate_cost(
            summary_usage.prompt_tokens, summary_usage.completion_tokens
        )
        # Filled in by the caller, which is what knows what tripped the line.
        self.trigger_tokens = 0
        self.trigger_reason = ""


def count_tokens(messages: list[dict]) -> int:
    """Estimate the billed prompt tokens for a request, before sending it.

    Identical to both siblings' version so the three runs are directly comparable -- and
    it is the same function that watches the request against the budget, so the trigger
    and the reported cost use one definition, not two.
    """
    total = 0
    for message in messages:
        total += TOKENS_PER_MESSAGE
        total += len(encoding.encode(message["role"]))  # 1 token, NOT free
        total += len(encoding.encode(message["content"]))
    return total + TOKENS_PER_REPLY_PRIMING


def block_tokens(messages: list[dict]) -> int:
    """What one layer costs inside a request: `count_tokens` without the reply priming.

    The priming is paid once per request, not once per layer, so charging it to each
    block would make the three layer figures fail to add up to the request total. With
    it removed, `pinned + summary + window + question + priming` is exactly the estimate.
    """
    return count_tokens(messages) - TOKENS_PER_REPLY_PRIMING if messages else 0


def wrap_content(text: str, first_line_offset: int = 0) -> list[str]:
    """Wrap a message for the dump, keeping its own line breaks.

    Nothing is elided: the tokenizer sees those newlines, so showing them keeps the dump
    faithful and makes the pinned block's one-fact-per-line shape readable.

    `first_line_offset` is the width of the `<|start|>role<|message|>` opener printed
    ahead of the first line. Reserving it keeps every line of the block inside the same
    right-hand margin instead of letting the opening line overshoot.
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

    The marker column is what separates the three layers, all of which are `user`
    messages and none of which the wire format itself distinguishes:
        P  the pinned facts block
        S  the rolling summary
        >  this turn's new user message
    Everything unmarked is the verbatim window.
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
        elif is_pinned(message):
            marker = "P"
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


def split_history(window: list[dict]) -> tuple[list[dict], list[dict]]:
    """Older half (to be evicted through the gates), recent half (kept verbatim).

    Reused from the summarisation sibling, guard and all. The split is by message count,
    not token mass: assistant replies run ~150 tokens against ~15-token questions, so
    "half the messages" is deliberately not "half the tokens", but the message count is
    what the per-turn output makes legible.

    The pair-alignment guard is the same one `build_window()` uses in the sliding-window
    sibling, for the same reason: a kept half that opens on an `assistant` message shows
    the model a reply whose question was just evicted. Since the window alternates
    user/assistant, an even-length window splits cleanly and the guard is a no-op; it
    fires only on an odd length, which is what a previous compaction can leave behind.

    Why half, and not one pair at a time: once the window is full it overflows every
    single turn, so evicting one pair per turn would fire both gates 20+ times -- two
    extra API calls per turn, costing more than they save and burying the demonstration.
    Halving drops the window to a level that takes ~3 turns to refill.
    """
    midpoint = len(window) // 2
    older, recent = window[:midpoint], window[midpoint:]
    if recent and recent[0]["role"] == "assistant":
        older.append(recent.pop(0))  # never open the kept half on a dangling answer
    return older, recent


def is_pinned(message: dict) -> bool:
    """True for the pinned block, which is a `user` message but not a user turn."""
    return message["role"] == "user" and message["content"].startswith(PINNED_PREFIX)


def is_summary(message: dict) -> bool:
    """True for the injected recap, which is a `user` message but not a user turn."""
    return message["role"] == "user" and message["content"].startswith(SUMMARY_PREFIX)


def flatten_for_gate(messages: list[dict]) -> str:
    """Messages as plain text, for substitution into either gate's prompt.

    The block prefixes are stripped on the way in. When the previous summary is folded
    into a new one it arrives carrying "[Conversation summary so far - ... do not reply
    to this message directly]", an instruction aimed at the conversation model that
    would only confuse the summariser being asked to rewrite it.
    """
    lines = []
    for message in messages:
        content = message["content"]
        for prefix in (PINNED_PREFIX, SUMMARY_PREFIX):
            if content.startswith(prefix):
                content = content[len(prefix):]
        lines.append(f"{message['role']}: {content}")
    return "\n".join(lines)


def extract_facts(older: list[dict], pinned: PinnedFacts):
    """GATE 1: lift anything that must be exact out of the batch about to be deleted.

    Runs BEFORE the summariser sees the same messages, which is the entire point of the
    design: a name or a title is copied out verbatim rather than left to survive a lossy
    rewrite. Returns the parsed `key: value` pairs, the call's usage, and how many values
    the length guard threw away -- worth watching, since a guard that fires at every
    compaction means the prompt is still being ignored.

    `.format()` is safe here -- the template holds no braces other than the two
    placeholders, and braces inside the substituted text are not re-scanned.
    """
    prompt = FACT_SYSTEM_PROMPT.format(
        known=pinned.as_text() or "(none yet)",
        transcript=flatten_for_gate(older),
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": prompt}],  # system-only, by design
        max_tokens=FACT_MAX_TOKENS,
    )

    pairs: list[tuple[str, str]] = []
    rejected = 0
    for line in (response.choices[0].message.content or "").splitlines():
        line = line.strip().lstrip("-*•").strip()
        if not line or line.upper() == "NONE":
            continue
        if ":" not in line:
            continue  # not a key: value line -- the extractor editorialising
        key, _, value = line.partition(":")  # first colon only; values contain colons
        key, value = key.strip().lower(), value.strip()
        if not key or not value:
            continue
        if len(encoding.encode(value)) > FACT_MAX_VALUE_TOKENS:
            rejected += 1  # a "fact" this long is a description, and descriptions are
            continue       # what fill the block and evict the facts that matter
        pairs.append((key, value))

    return pairs, response.usage, rejected


def summarise(older: list[dict], summary: dict | None, pinned: PinnedFacts):
    """GATE 2: fold the evicted batch into the rolling summary.

    The existing summary is prepended to the batch and re-summarised, so the history
    stays [<=1 pinned] + [<=1 summary] + [window]. Neither prefix grows and the token
    curve is a genuine sawtooth rather than creeping upward under a chain of summaries.

    `pinned` is passed in *after* Gate 1 has updated it, so the {known} block tells the
    summariser exactly what it does not need to carry -- that is what pays for the
    reduced budget here.
    """
    to_summarise = ([summary] if summary else []) + older
    prompt = SUMMARY_SYSTEM_PROMPT.format(
        known=pinned.as_text() or "(none yet)",
        transcript=flatten_for_gate(to_summarise),
    )
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


def compact(
    pinned: PinnedFacts,
    summary: dict | None,
    window: list[dict],
    number: int,
    turn_number: int,
):
    """Evict the older half of the window through both gates, in that order."""
    older, window = split_history(window)
    kept_before = list(window)  # the kept half must come through byte-for-byte
    older_tokens = block_tokens(([summary] if summary else []) + older)

    gate_order: list[str] = []  # the ordering is the design, so it is recorded and tested

    gate_order.append("facts")
    pairs, fact_usage, rejected = extract_facts(older, pinned)
    added, replaced = pinned.update(pairs, turn_number)

    gate_order.append("summary")
    summary, summary_usage = summarise(older, summary, pinned)

    dropped = pinned.enforce_cap()

    assert gate_order == ["facts", "summary"], "Gate 1 must run before Gate 2"
    assert window == kept_before, "the kept half of the window was not preserved verbatim"
    assert len(pinned.facts) == len(set(pinned.facts)), "duplicate key in the pinned block"
    assert pinned.token_count() <= PINNED_MAX_TOKENS, "the pinned block broke its cap"
    assert block_tokens([summary]) < older_tokens, "compaction made the prompt bigger"

    stats = CompactionStats(
        number=number,
        window_before=len(kept_before) + len(older),
        older_count=len(older),
        kept_count=len(window),
        older_tokens=older_tokens,
        summary_tokens=block_tokens([summary]),
        facts_added=added,
        facts_replaced=replaced,
        facts_dropped=dropped,
        facts_rejected=rejected,
        pinned_count=len(pinned.facts),
        pinned_tokens=pinned.token_count(),
        window_tokens=block_tokens(window),
        fact_usage=fact_usage,
        summary_usage=summary_usage,
        gate_order=gate_order,
    )
    return pinned, summary, window, stats


def build_request(
    pinned: PinnedFacts,
    summary: dict | None,
    window: list[dict],
    user_message: dict,
) -> list[dict]:
    """The three blocks, then the question -- and the layer invariants, asserted.

    Both blocks are omitted entirely until they have content, so early turns are pure
    window and the token curve starts as low as the sliding-window script's. The current
    user message goes on last, after any compaction, so this turn's question always
    reaches the model verbatim.
    """
    pinned_message = pinned.as_message()
    messages = (
        ([pinned_message] if pinned_message else [])
        + ([summary] if summary else [])
        + window
        + [user_message]
    )

    assert all(m["role"] != "system" for m in messages), "the main call pins nothing"
    assert sum(is_pinned(m) for m in messages) <= 1, "more than one pinned block"
    assert sum(is_summary(m) for m in messages) <= 1, "more than one summary"
    if pinned_message:
        assert is_pinned(messages[0]), "the pinned block is not at index 0"
        assert messages[0]["role"] == "user", "the pinned block is not a user message"
    if summary:
        index = 1 if pinned_message else 0
        assert is_summary(messages[index]), "the summary is not directly after the pinned block"
        assert messages[index]["role"] == "user", "the summary is not a user message"
    return messages


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


def print_compaction_banner(stats: CompactionStats) -> None:
    """Both gates, separately, because both were separately paid for.

    The `GATE 1 ... N new, N replaced` line is the one to watch: `replaced` is the
    staleness mechanism working, and a `replaced` count that never moves means facts are
    only ever accreting.
    """
    compression = stats.older_tokens / stats.summary_tokens if stats.summary_tokens else 0.0
    print(
        f"*** COMPACTION #{stats.number}: request would be {stats.trigger_tokens:,} tokens,"
        f" {stats.trigger_reason}"
    )
    print(
        f"    Window split at message {stats.older_count} (pair-aligned):"
        f" {stats.older_count} evicted -> gates, {stats.kept_count} kept verbatim"
    )
    print(
        f"    GATE 1 (facts):   {stats.facts_added} new, {stats.facts_replaced} replaced"
        f" -> pinned now {stats.pinned_count} facts / {stats.pinned_tokens} tok"
    )
    print(
        f"                      call: {stats.fact_usage.prompt_tokens:,} prompt +"
        f" {stats.fact_usage.completion_tokens:,} completion = ${stats.fact_cost:.6f}"
    )
    if stats.facts_rejected:
        print(
            f"                      guard rejected {stats.facts_rejected} value(s) over"
            f" {FACT_MAX_VALUE_TOKENS} tokens -- descriptions, not facts"
        )
    if stats.facts_dropped:
        print(f"                      cap evicted (largest first): {', '.join(stats.facts_dropped)}")
    print(
        f"    GATE 2 (summary): {stats.older_tokens:,} tokens of history ->"
        f" {stats.summary_tokens:,} token summary ({compression:.1f}x)"
    )
    print(
        f"                      call: {stats.summary_usage.prompt_tokens:,} prompt +"
        f" {stats.summary_usage.completion_tokens:,} completion = ${stats.summary_cost:.6f}"
    )
    print(
        f"    Layers after: pinned {stats.pinned_tokens} + summary {stats.summary_tokens}"
        f" + window {stats.window_tokens} ="
        f" {stats.pinned_tokens + stats.summary_tokens + stats.window_tokens} tokens"
    )


def run_conversation(conversation_turns: list[ConversationTurn]) -> None:
    pinned = PinnedFacts()          # layer 1: exact, merged by key
    summary: dict | None = None     # layer 2: lossy, capped, re-folded
    window: list[dict] = []         # layer 3: verbatim, bounded
    full_transcript: list[dict] = []  # never compacted, never sent -- accounting only

    # Three kinds of call, three accumulators, because there are three things to be
    # honest about: the conversation itself and each gate's overhead.
    main_prompt_total = 0
    main_completion_total = 0
    fact_prompt_total = 0
    fact_completion_total = 0
    summary_prompt_total = 0
    summary_completion_total = 0
    full_prompt_total = 0

    running_cost = 0.0
    peak_hybrid = 0
    peak_full = 0
    compactions = 0

    for turn_number, turn in enumerate(conversation_turns, start=1):
        user_message = {"role": "user", "content": turn.user}

        print(f"--- Turn {turn_number} ---")

        # The trigger is the full request about to be sent -- all three layers plus this
        # turn's question -- and it is checked BEFORE the call. Checking afterwards would
        # let the turn that trips the line go out over budget, enforcing the cap one turn
        # late. Both clauses fire in practice (measured: 5 of 8 compactions on the budget,
        # 3 on the message count), so the window stays bounded by structure as well as by
        # tokens -- see WINDOW_MAX_MESSAGES.
        candidate_tokens = count_tokens(build_request(pinned, summary, window, user_message))
        over_budget = candidate_tokens > REQUEST_TOKEN_BUDGET
        over_messages = len(window) > WINDOW_MAX_MESSAGES
        if (over_budget or over_messages) and len(window) >= 2:
            compactions += 1
            pinned, summary, window, stats = compact(
                pinned, summary, window, compactions, turn_number
            )
            stats.trigger_tokens = candidate_tokens
            stats.trigger_reason = (
                f"over budget {REQUEST_TOKEN_BUDGET:,}"
                if over_budget
                else f"window of {stats.window_before} messages over cap"
                f" {WINDOW_MAX_MESSAGES}"
            )
            fact_prompt_total += stats.fact_usage.prompt_tokens
            fact_completion_total += stats.fact_usage.completion_tokens
            summary_prompt_total += stats.summary_usage.prompt_tokens
            summary_completion_total += stats.summary_usage.completion_tokens
            running_cost += stats.fact_cost + stats.summary_cost
            print_compaction_banner(stats)

        messages = build_request(pinned, summary, window, user_message)
        estimated = count_tokens(messages)
        assert estimated <= REQUEST_TOKEN_BUDGET + BUDGET_SLACK, (
            f"request of {estimated} tokens exceeded the budget -- lower"
            " SUMMARY_MAX_TOKENS or PINNED_MAX_TOKENS"
        )

        pinned_tokens = pinned.token_count()
        summary_tokens = block_tokens([summary]) if summary else 0
        window_tokens = block_tokens(window)
        full_transcript_tokens = count_tokens(full_transcript + [user_message])

        text, usage = call_llm(messages)

        main_prompt_total += usage.prompt_tokens
        main_completion_total += usage.completion_tokens
        full_prompt_total += full_transcript_tokens
        turn_cost = estimate_cost(usage.prompt_tokens, usage.completion_tokens)
        running_cost += turn_cost
        peak_hybrid = max(peak_hybrid, usage.prompt_tokens)
        peak_full = max(peak_full, full_transcript_tokens)

        ratio = full_transcript_tokens / estimated if estimated else 1.0
        drift = estimated - usage.prompt_tokens
        blocks = (1 if pinned_tokens else 0) + (1 if summary else 0)

        print(
            f"Layers: pinned {len(pinned.facts)} facts / {pinned_tokens} tok"
            f" | summary {summary_tokens} tok"
            f" | window {len(window)} msgs / {window_tokens} tok"
        )
        print(
            f"        = {estimated:,} of {REQUEST_TOKEN_BUDGET:,} budget"
            f" | compactions so far: {compactions}"
        )
        print(
            f"Sent:   {1 if pinned_tokens else 0} pinned + {1 if summary else 0} summary"
            f" + {len(window)} window + 1 user = {len(messages)} messages"
        )
        assert len(messages) == blocks + len(window) + 1, "the request is not three layers + a question"
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
        window.append(user_message)
        window.append(assistant_message)
        full_transcript.append(user_message)
        full_transcript.append(assistant_message)

    print_summary(
        len(conversation_turns),
        compactions,
        main_prompt_total,
        main_completion_total,
        fact_prompt_total,
        fact_completion_total,
        summary_prompt_total,
        summary_completion_total,
        full_prompt_total,
        running_cost,
        peak_hybrid,
        peak_full,
        pinned,
    )


def print_summary(
    turns: int,
    compactions: int,
    main_prompt_total: int,
    main_completion_total: int,
    fact_prompt_total: int,
    fact_completion_total: int,
    summary_prompt_total: int,
    summary_completion_total: int,
    full_prompt_total: int,
    running_cost: float,
    peak_hybrid: int,
    peak_full: int,
    pinned: PinnedFacts,
) -> None:
    saved_tokens = full_prompt_total - main_prompt_total
    saved_pct = saved_tokens / full_prompt_total * 100 if full_prompt_total else 0.0
    hybrid_input_cost = main_prompt_total / 1_000_000 * INPUT_COST_PER_1M
    full_input_cost = full_prompt_total / 1_000_000 * INPUT_COST_PER_1M
    fact_cost = estimate_cost(fact_prompt_total, fact_completion_total)
    summary_cost = estimate_cost(summary_prompt_total, summary_completion_total)
    gate_overhead = fact_cost + summary_cost
    completion_total = main_completion_total + fact_completion_total + summary_completion_total

    print(f"=========== SUMMARY ({turns} turns) ===========")
    print(
        f"Compactions     : {compactions:>8,}"
        f" ({compactions * 2} gate calls: {compactions} extract + {compactions} summarise)"
    )
    print(f"Hybrid          : {main_prompt_total:>8,} prompt tokens   ${hybrid_input_cost:.5f}")
    print(
        f"Full resend     : {full_prompt_total:>8,} prompt tokens   ${full_input_cost:.5f}"
        "   <- what it would have cost"
    )
    print(f"Saved           : {saved_tokens:>8,} prompt tokens   ({saved_pct:.1f}%)")
    # The honesty this run owes: two gates were paid for, so both are reported next to
    # what they saved. A net saving near zero at 30 turns is the expected result, not a
    # failure -- see the module docstring.
    print(
        f"Gate 1 (facts)  : {fact_prompt_total:>8,} prompt + {fact_completion_total:>6,}"
        f" completion tokens   ${fact_cost:.5f}"
    )
    print(
        f"Gate 2 (summary): {summary_prompt_total:>8,} prompt + {summary_completion_total:>6,}"
        f" completion tokens   ${summary_cost:.5f}"
    )
    print(f"Gate overhead   : ${gate_overhead:.5f}   <- the price of the two gates")
    print(f"Net saving      : ${full_input_cost - hybrid_input_cost - gate_overhead:.5f}")
    print(
        f"Peak prompt size, hybrid: {peak_hybrid:,} tokens (capped)"
        f"   full: {peak_full:,} tokens (growing)"
    )
    print(f"Pinned at end   : {len(pinned.facts)} facts / {pinned.token_count()} tokens")
    for key, value in pinned.facts.items():
        print(f"                  {key}: {value}")
    print(
        f"Actual spend this run: ${running_cost:.6f}"
        f" (input + {completion_total:,} completion tokens)"
    )


if __name__ == "__main__":
    conversation_turns = [
        # The same 30 turns as both siblings, copied verbatim so the three strategies are
        # comparable turn-for-turn. Turns 1-20 themselves come from
        # agent_in_context_memory.py.
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
        # evicted; pure summarisation passed them but lost the book's *attribution*.
        # Here they should come off the pinned block, exactly, however many times the
        # summary has been re-folded.
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
        # Turn 28 stays partial BY DESIGN and should be read that way: no layer here
        # stores the *wording* of turn 1. Pinned facts store values, the summary stores
        # gist, and turn 1 left the window long ago. A paraphrase is the correct outcome.
        ConversationTurn(user="What was the very first thing I asked you today?"),
        # Turn 29 is the probe that justifies the whole design -- the one pure
        # summarisation answered differently on different runs.
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
