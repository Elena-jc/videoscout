"""Minimal SRT parser and subtitle-to-segment assignment."""

from __future__ import annotations

import re
from pathlib import Path

from .store import Segment

_TIME = re.compile(r"(\d+):(\d{2}):(\d{2})[,.](\d{1,3})")
_TAG = re.compile(r"<[^>]+>|\{[^}]+\}")


def _seconds(match: re.Match) -> float:
    h, m, s, ms = match.groups()
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000


def parse_srt(path: str | Path) -> list[tuple[float, float, str]]:
    text = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    cues = []
    for block in re.split(r"\r?\n\s*\r?\n", text.strip()):
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        for i, line in enumerate(lines):
            times = list(_TIME.finditer(line))
            if "-->" in line and len(times) == 2:
                body = " ".join(_TAG.sub("", ln) for ln in lines[i + 1 :]).strip()
                if body:
                    cues.append((_seconds(times[0]), _seconds(times[1]), body))
                break
    return cues


def assign_to_segments(cues: list[tuple[float, float, str]], segments: list[Segment]) -> None:
    """Attach each cue to every segment it overlaps."""
    for seg in segments:
        texts = [body for start, end, body in cues if start < seg.t_end and end > seg.t_start]
        seg.subtitle = " ".join(texts)
