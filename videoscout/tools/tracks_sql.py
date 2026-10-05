"""query_tracks: read-only SQL over the object-tracking memory.

Letting a model write SQL is powerful (counting, durations, ordering, motion) and
risky. Guardrails, from outermost to innermost:
  1. the database file is opened with mode=ro, so the OS-level handle cannot write;
  2. an SQLite authorizer allows only SELECT / READ / FUNCTION operations, so
     PRAGMA, ATTACH, INSERT, DROP... are rejected at compile time;
  3. the sqlite3 module executes a single statement per call (no "; DROP TABLE");
  4. a progress handler aborts queries that run past a time limit;
  5. results are capped at MAX_ROWS rows so one query cannot flood the context.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Callable

from pydantic import BaseModel, ConfigDict, Field

from .registry import Tool, ToolError

MAX_ROWS = 50
TIMEOUT_SECONDS = 2.0

_ALLOWED_ACTIONS = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
    getattr(sqlite3, "SQLITE_RECURSIVE", 33),  # WITH RECURSIVE; bounded by the timeout
}


def _authorizer(action: int, arg1, arg2, db_name, trigger) -> int:
    return sqlite3.SQLITE_OK if action in _ALLOWED_ACTIONS else sqlite3.SQLITE_DENY


def run_readonly_query(
    connect: Callable[[], sqlite3.Connection],
    sql: str,
    max_rows: int = MAX_ROWS,
    timeout: float = TIMEOUT_SECONDS,
) -> tuple[list[str], list[tuple], bool]:
    conn = connect()
    conn.set_authorizer(_authorizer)
    deadline = time.monotonic() + timeout
    conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
    try:
        cursor = conn.execute(sql)
        rows = cursor.fetchmany(max_rows + 1)
        columns = [d[0] for d in cursor.description or []]
    except (sqlite3.Error, sqlite3.Warning) as err:
        if "interrupted" in str(err):
            raise ToolError(f"Query exceeded the {timeout:g}s time limit; simplify it or add filters.") from err
        if "not authorized" in str(err):
            raise ToolError("Only read-only SELECT queries are allowed.") from err
        raise ToolError(f"SQL error: {err}") from err
    finally:
        conn.close()
    return columns, rows[:max_rows], len(rows) > max_rows


def _fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def format_table(columns: list[str], rows: list[tuple], truncated: bool) -> str:
    if not rows:
        return "Query returned no rows."
    lines = [" | ".join(columns)]
    lines += [" | ".join(_fmt(v) for v in row) for row in rows]
    footer = f"(first {len(rows)} rows shown; aggregate or add LIMIT)" if truncated else f"({len(rows)} rows)"
    return "\n".join(lines + [footer])


DESCRIPTION_TEMPLATE = """Run one read-only SQLite SELECT over the object-tracking memory of the video \
(YOLO detections linked into tracks by ByteTrack). Best for counting, durations, first/last appearance, \
temporal order and motion questions. Times are seconds; coordinates are normalised to [0, 1] (x to the right, y down).

Tables:
  tracks(track_id, label, t_first, t_last, duration, n_obs, mean_conf, cx_first, cy_first, cx_last, cy_last,
         path_length, net_displacement, mean_area)
      one row per tracked object; label is one of the 80 COCO class names ('person', 'car', 'bicycle', 'dog', ...)
      (for anything else, use find_objects);
      path_length = total distance travelled, net_displacement = start-to-end distance, mean_area = box size
  detections(track_id, t, x1, y1, x2, y2, conf)
      one row per object per sampled frame (about {track_fps:g} frames per second)
  segments(seg_id, t_start, t_end, subtitle, objects, caption)
  events(event_id, t_start, t_end, seg_first, seg_last, summary)

Caveats: track IDs can switch or fragment under occlusion, so COUNT(DISTINCT track_id) over-counts objects. \
Short, low-confidence tracks (mean_conf < 0.5) are often duplicate boxes on an object that is already tracked. \
For "how many X" prefer the peak number visible at once among confident tracks, then verify visually with \
inspect_clip. Small or distant objects are often missed.

Examples:
  SELECT MAX(n) FROM (SELECT t, COUNT(*) AS n FROM detections JOIN tracks USING(track_id)
                      WHERE label='person' AND mean_conf >= 0.5 GROUP BY t)
  SELECT track_id, t_first, t_last, duration FROM tracks WHERE label='car' ORDER BY t_first LIMIT 10
  SELECT track_id, duration, net_displacement FROM tracks WHERE label='person' AND duration > 30 ORDER BY net_displacement
Results are capped at {max_rows} rows."""


class QueryTracksArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sql: str = Field(description="A single SQLite SELECT statement.")


def make_query_tracks_tool(connect: Callable[[], sqlite3.Connection], track_fps: float) -> Tool:
    def run(args: QueryTracksArgs) -> str:
        return format_table(*run_readonly_query(connect, args.sql))

    description = DESCRIPTION_TEMPLATE.format(track_fps=track_fps, max_rows=MAX_ROWS)
    return Tool("query_tracks", description, QueryTracksArgs, run)
