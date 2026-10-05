"""End-to-end indexing with real models (YOLO11n + ByteTrack, SigLIP) on the demo
video. Excluded by default; run with `pytest -m integration`. No API calls."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration
pytest.importorskip("ultralytics")
pytest.importorskip("transformers")

from videoscout.config import load_config  # noqa: E402
from videoscout.index.build import build_index  # noqa: E402
from videoscout.index.store import VideoIndex  # noqa: E402
from videoscout.retrieval import HybridRetriever  # noqa: E402
from videoscout.tools import default_embedder_factory  # noqa: E402
from videoscout.tools.tracks_sql import run_readonly_query  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def demo_index(tmp_path_factory):
    out = tmp_path_factory.mktemp("demo")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_demo_video.py"), "--out", str(out)], check=True)
    cfg = load_config()
    build_index(out / "demo.mp4", out / "index", cfg, srt_path=out / "demo.srt")
    return VideoIndex(out / "index"), cfg


def test_tracking_finds_people_and_the_bus(demo_index):
    index, _ = demo_index
    assert "person" in index.label_counts and "bus" in index.label_counts
    # When the camera zooms in, YOLO11n also puts a second, low-confidence box on one of
    # the men (a duplicate track), so the raw peak is 3. Filtering weak tracks, as the
    # query_tracks description tells the agent to do, gives the true count.
    sql = (
        "SELECT MAX(n) FROM (SELECT t, COUNT(*) AS n FROM detections JOIN tracks USING(track_id) "
        "WHERE label='person' AND mean_conf >= 0.5 AND t BETWEEN 40 AND 80 GROUP BY t)"
    )
    peak = run_readonly_query(index.connect, sql)[1][0][0]
    assert peak == 2  # the office scene shows two men


def test_open_vocabulary_detection_from_text(demo_index):
    from videoscout.tools import default_detector
    from videoscout.video import VideoReader

    index, cfg = demo_index
    detector = default_detector(cfg)
    reader = VideoReader(index.video_path)
    street = detector.detect([f for _, f in reader.sample(5, 35, 4)], ["bus", "necktie"])
    gate = detector.detect([f for _, f in reader.sample(85, 115, 4)], ["bus", "necktie"])
    assert sum(any(d[0] == "bus" for d in frame) for frame in street) >= 3
    assert not any(d[0] == "bus" for frame in gate for d in frame)


@pytest.mark.parametrize(
    "visual_query, window",
    [("a large bus on a city street", (0, 40)), ("two men in suits", (40, 80)), ("a white sign board with black text", (80, 120))],
)
def test_dense_search_localises_scenes(demo_index, visual_query, window):
    index, cfg = demo_index
    cfg.retrieval.use_bm25 = False  # test the visual retriever on its own
    retriever = HybridRetriever(index, cfg.retrieval, default_embedder_factory(index, cfg))
    hits = retriever.search(visual_query, visual_query, top_k=3)
    assert any(window[0] <= h.t_start < window[1] for h in hits), [(h.t_start, h.score) for h in hits]
