"""Multi-granular video memory: clips -> events -> storyline.

The long-video agents that lead LVBench (Deep Video Discovery, VideoSeek) do not
search one flat list of fixed windows. They keep the video at several
granularities and move coarse-to-fine: read the storyline, pick the relevant
part, search clips inside it, and only then spend frames. This module builds the
coarse levels on top of the fixed-length clips:

  L0 storyline  one paragraph about the whole video
  L1 events     runs of consecutive clips that belong together (a scene, a topic),
                cut where adjacent clips are unusually dissimilar; one summary each
  L2 clips      the fixed-length segments: subtitles, captions, embeddings
  L3 frames     read on demand by inspect_clip

L0-L2 are text, so the agent can browse the whole video for a few hundred tokens
and keep its frame budget for questions that need pixels.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from ..video import fmt_ts
from .store import Event, Segment

Writer = Callable[[str], str]  # prompt -> text (local captioner or an LLM)

MAX_LINES_PER_PROMPT = 30
SUBTITLE_CHARS = 800
STORYLINE_GROUP = 20


def cut_events(
    segments: Sequence[Segment],
    clip_emb: np.ndarray | None,
    min_seconds: float = 20.0,
    max_seconds: float = 180.0,
    z: float = 1.0,
) -> list[Event]:
    """Shot/scene-style boundaries from clip embeddings.

    A boundary goes between clips i and i+1 when their cosine similarity is below
    mean - z * std of all adjacent similarities (an adaptive threshold: a static
    lecture and an action film have very different baselines), as long as the
    current event is at least `min_seconds`; events are also closed at `max_seconds`.
    Without embeddings the video is cut into equal windows of `max_seconds / 2`.
    """
    n = len(segments)
    if n == 0:
        return []
    if clip_emb is None or len(clip_emb) != n or n < 2:
        sims, threshold = None, -np.inf
        max_seconds = max_seconds / 2
    else:
        emb = clip_emb.astype(np.float32)
        sims = np.sum(emb[:-1] * emb[1:], axis=1)
        threshold = float(sims.mean() - z * sims.std())

    spans: list[tuple[int, int]] = []
    start = 0
    for i in range(n - 1):
        length = segments[i].t_end - segments[start].t_start
        too_long = segments[i + 1].t_end - segments[start].t_start > max_seconds
        scene_change = sims is not None and sims[i] < threshold and length >= min_seconds
        if scene_change or too_long:
            spans.append((start, i))
            start = i + 1
    spans.append((start, n - 1))
    # A short tail is merged into the previous event.
    if len(spans) > 1 and segments[spans[-1][1]].t_end - segments[spans[-1][0]].t_start < min_seconds:
        last = spans.pop()
        spans[-1] = (spans[-1][0], last[1])
    return [
        Event(k, segments[a].t_start, segments[b].t_end, segments[a].seg_id, segments[b].seg_id)
        for k, (a, b) in enumerate(spans)
    ]


EVENT_PROMPT = """These notes describe consecutive clips from one part of a video ({start}-{end}), in order.
{notes}
Write one or two sentences (at most 50 words) summarising what happens in this part: who or what is \
shown, actions, setting, and any key on-screen text or dialogue. Use only the notes; do not speculate."""

STORYLINE_PROMPT = """Below are summaries of consecutive parts of a {duration} video, in order.
{parts}
Write a summary of the whole video in at most {words} words: the setting, the main people or subjects, \
and how things unfold. Use only these summaries."""


def _notes(segments: Sequence[Segment]) -> str:
    lines = [f"- {fmt_ts(s.t_start)}: {s.caption}" for s in segments if s.caption][:MAX_LINES_PER_PROMPT]
    subs = " ".join(s.subtitle for s in segments if s.subtitle)[:SUBTITLE_CHARS]
    out = "Clip captions:\n" + "\n".join(lines) if lines else ""
    if subs:
        out += ("\n" if out else "") + f"Speech/subtitles: {subs}"
    return out


def _fallback_summary(segments: Sequence[Segment], limit: int = 220) -> str:
    """Without a writer model: the first caption plus the start of the speech."""
    caption = next((s.caption for s in segments if s.caption), "")
    subs = " ".join(s.subtitle for s in segments if s.subtitle)
    text = " ".join(p for p in (caption, f'Speech: "{subs}"' if subs else "") if p)
    return text if len(text) <= limit else text[: limit - 3] + "..."


BatchWriter = Callable[[list[str]], list[str]]  # several prompts in one batched call


def _writer_many(write: Writer, write_many: BatchWriter | None) -> BatchWriter:
    return write_many or (lambda prompts: [write(p) for p in prompts])


def summarize_events(events: list[Event], segments: Sequence[Segment], write: Writer | None,
                     write_many: BatchWriter | None = None) -> None:
    by_id = {s.seg_id: s for s in segments}
    pending: list[tuple[Event, str]] = []
    for ev in events:
        members = [by_id[i] for i in range(ev.seg_first, ev.seg_last + 1) if i in by_id]
        notes = _notes(members)
        if write is None or not notes:
            ev.summary = _fallback_summary(members)
        else:
            pending.append((ev, EVENT_PROMPT.format(start=fmt_ts(ev.t_start), end=fmt_ts(ev.t_end), notes=notes)))
    if pending:
        summaries = _writer_many(write, write_many)([prompt for _, prompt in pending])
        for (ev, _), summary in zip(pending, summaries):
            ev.summary = summary


def write_storyline(events: Sequence[Event], duration: float, write: Writer | None,
                    write_many: BatchWriter | None = None) -> str:
    parts = [f"[{fmt_ts(e.t_start)}-{fmt_ts(e.t_end)}] {e.summary}" for e in events if e.summary]
    if not parts:
        return ""
    if write is None:
        return " ".join(e.summary for e in events[:3] if e.summary)[:400]
    # Hierarchical for long videos: summarise groups of parts, then the group summaries.
    while len(parts) > STORYLINE_GROUP * 2:
        parts = _writer_many(write, write_many)([
            STORYLINE_PROMPT.format(duration=fmt_ts(duration), parts="\n".join(parts[i:i + STORYLINE_GROUP]), words=80)
            for i in range(0, len(parts), STORYLINE_GROUP)
        ])
    return write(STORYLINE_PROMPT.format(duration=fmt_ts(duration), parts="\n".join(parts), words=120))
