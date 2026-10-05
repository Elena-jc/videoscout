from __future__ import annotations

import numpy as np
import pytest

from videoscout.index.store import Segment, VideoIndex, write_index
from videoscout.index.tracker import summarize_tracks, tag_segments
from videoscout.tools.registry import ToolError
from videoscout.tools.tracks_sql import format_table, make_query_tracks_tool, run_readonly_query

ROWS = [
    # track 1: a person standing still for 30 s
    *[(1, float(t), 0.40, 0.40, 0.50, 0.80, 0.9) for t in range(0, 31, 1)],
    # track 2: a person walking left to right
    *[(2, float(t), 0.05 * t, 0.50, 0.05 * t + 0.1, 0.9, 0.8) for t in range(0, 11)],
    # track 3: a single false positive (dropped: fewer than 2 observations)
    (3, 5.0, 0.1, 0.1, 0.2, 0.2, 0.3),
]


@pytest.fixture
def index(tmp_path):
    tracks = summarize_tracks(ROWS, {1: "person", 2: "person", 3: "dog"})
    segments = [Segment(i, i * 10.0, (i + 1) * 10.0) for i in range(4)]
    tag_segments(segments, ROWS, tracks)
    meta = {"video_path": "none.mp4", "duration": 40.0, "segment_seconds": 10.0}
    kept = [r for r in ROWS if r[0] != 3]
    write_index(tmp_path, meta, segments, [], np.zeros((0, 0), np.float32), tracks, kept)
    return VideoIndex(tmp_path)


def test_track_summary_statistics(index):
    rows = run_readonly_query(index.connect, "SELECT track_id, duration, path_length, net_displacement FROM tracks ORDER BY track_id")[1]
    (t1, d1, p1, n1), (t2, d2, p2, n2) = rows
    assert (t1, d1, p1, n1) == (1, 30.0, 0.0, 0.0)  # loiterer: long stay, no movement
    assert d2 == 10.0 and p2 == pytest.approx(0.5) and n2 == pytest.approx(0.5)
    assert index.segments[0].objects == "objects: person x2"
    assert index.label_counts == {"person": 2}


def test_select_and_aggregate(index):
    cols, rows, truncated = run_readonly_query(
        index.connect,
        "SELECT MAX(n) FROM (SELECT t, COUNT(*) AS n FROM detections JOIN tracks USING(track_id) "
        "WHERE label='person' GROUP BY t)",
    )
    assert rows == [(2,)] and not truncated


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM tracks",
        "INSERT INTO tracks(track_id, label) VALUES (9, 'x')",
        "DROP TABLE detections",
        "UPDATE tracks SET label='cat'",
        "PRAGMA table_info(tracks)",
        "ATTACH DATABASE 'other.db' AS other",
        "CREATE TABLE evil(x)",
    ],
)
def test_writes_and_side_effects_are_rejected(index, sql):
    with pytest.raises(ToolError):
        run_readonly_query(index.connect, sql)
    assert run_readonly_query(index.connect, "SELECT COUNT(*) FROM tracks")[1] == [(2,)]


def test_stacked_statements_are_rejected(index):
    with pytest.raises(ToolError):
        run_readonly_query(index.connect, "SELECT 1; DELETE FROM tracks")


def test_runaway_query_hits_timeout(index):
    sql = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT MAX(x) FROM c"
    with pytest.raises(ToolError, match="time limit"):
        run_readonly_query(index.connect, sql, timeout=0.3)


def test_rows_are_capped(index):
    cols, rows, truncated = run_readonly_query(index.connect, "SELECT * FROM detections", max_rows=5)
    assert len(rows) == 5 and truncated
    assert "first 5 rows" in format_table(cols, rows, truncated)


def test_tool_returns_errors_to_the_model(index):
    tool = make_query_tracks_tool(index.connect, track_fps=2.0)
    ok = tool.run({"sql": "SELECT label, COUNT(*) FROM tracks GROUP BY label"})
    assert not ok.is_error and "person | 2" in ok.text
    bad = tool.run({"sql": "DROP TABLE tracks"})
    assert bad.is_error and "read-only" in bad.text
    invalid = tool.run({"query": "SELECT 1"})
    assert invalid.is_error and "Invalid arguments" in invalid.text
