"""Generate a 2-minute demo video, subtitles and a small QA set, so the whole
pipeline can be exercised without downloading a benchmark.

Scenes (the photos ship inside the ultralytics package):
  0-40 s   street: camera pans over a bus with people (ultralytics bus.jpg)
  40-80 s  office: slow zoom on two men talking (ultralytics zidane.jpg)
  80-120 s gate:   a sign reading "GATE B CLOSED" and a red square moving across

    python scripts/make_demo_video.py --out data/demo
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from videoscout.video import open_browser_writer

W, H, FPS, SCENE_SECONDS = 640, 360, 10, 40


def _asset(name: str) -> np.ndarray | None:
    try:
        from ultralytics.utils import ASSETS
    except ImportError:
        return None
    img = cv2.imread(str(Path(ASSETS) / name))
    return img


def _placeholder(label: str, color: tuple[int, int, int]) -> np.ndarray:
    img = np.full((720, 1280, 3), color, np.uint8)
    cv2.putText(img, label, (80, 380), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 6)
    return img


def street_frames(img: np.ndarray):
    """Pan a 16:9 window down the (portrait) photo."""
    h, w = img.shape[:2]
    win_h = min(h, int(w * 9 / 16))
    n = SCENE_SECONDS * FPS
    for i in range(n):
        y = int((h - win_h) * i / (n - 1))
        yield cv2.resize(img[y : y + win_h, :], (W, H), interpolation=cv2.INTER_AREA)


def office_frames(img: np.ndarray):
    """Slow zoom from the full frame to an 85% centre crop."""
    h, w = img.shape[:2]
    n = SCENE_SECONDS * FPS
    for i in range(n):
        scale = 1.0 - 0.15 * i / (n - 1)
        cw, ch = int(w * scale), int(h * scale)
        x, y = (w - cw) // 2, (h - ch) // 2
        yield cv2.resize(img[y : y + ch, x : x + cw], (W, H), interpolation=cv2.INTER_AREA)


def gate_frames():
    n = SCENE_SECONDS * FPS
    for i in range(n):
        frame = np.full((H, W, 3), (70, 70, 70), np.uint8)
        cv2.rectangle(frame, (0, 260), (W, H), (40, 90, 40), -1)  # grass
        cv2.rectangle(frame, (170, 60), (470, 150), (245, 245, 245), -1)  # sign board
        cv2.rectangle(frame, (170, 60), (470, 150), (20, 20, 20), 3)
        cv2.putText(frame, "GATE B CLOSED", (188, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (20, 20, 20), 3)
        cv2.rectangle(frame, (310, 150), (330, 260), (60, 60, 60), -1)  # post
        x = int(20 + (W - 80) * i / (n - 1))
        cv2.rectangle(frame, (x, 280), (x + 50, 330), (0, 0, 255), -1)  # red square (BGR)
        yield frame


SRT = """1
00:00:02,000 --> 00:00:06,000
Our city tour starts at the main bus stop.

2
00:00:20,000 --> 00:00:24,000
Several passengers are waiting on the street.

3
00:00:44,000 --> 00:00:48,000
Now we are inside, where two colleagues are talking.

4
00:01:24,000 --> 00:01:28,000
Finally, notice the sign at the entrance.
"""

QA = [
    {"qid": "demo-1", "question": "What kind of vehicle appears in the video?",
     "options": ["A. A bus", "B. A bicycle", "C. An airplane", "D. A boat"], "answer": "A",
     "task_type": "object recognition", "gt_windows": [[0, 40]]},
    {"qid": "demo-2", "question": "What does the sign at the entrance say?",
     "options": ["A. GATE A OPEN", "B. GATE B CLOSED", "C. EXIT ONLY", "D. NO PARKING"], "answer": "B",
     "task_type": "OCR", "gt_windows": [[80, 120]]},
    {"qid": "demo-3", "question": "What color is the square that moves across the gate scene?",
     "options": ["A. Green", "B. Blue", "C. Red", "D. Yellow"], "answer": "C",
     "task_type": "attribute", "gt_windows": [[80, 120]]},
    {"qid": "demo-4", "question": "When in the video do the two men talk to each other?",
     "options": ["A. In the first third", "B. In the middle third", "C. In the last third", "D. They never appear"],
     "answer": "B", "task_type": "temporal localization", "gt_windows": [[40, 80]]},
    {"qid": "demo-5", "question": "How many people are visible at the same time in the office scene?",
     "options": ["A. One", "B. Two", "C. Three", "D. Five"], "answer": "B",
     "task_type": "counting", "gt_windows": [[40, 80]]},
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/demo")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    bus = _asset("bus.jpg")
    office = _asset("zidane.jpg")
    if bus is None or office is None:
        print("ultralytics assets not found; using placeholder scenes (detection results will be empty)")
        bus = bus if bus is not None else _placeholder("BUS STOP", (120, 80, 30))
        office = office if office is not None else _placeholder("OFFICE", (30, 80, 120))

    # H.264 (or VP9 WebM) so the same file plays in the browser UI.
    writer, video_path = open_browser_writer(out / "demo", FPS, (W, H))
    for frames in (street_frames(bus), office_frames(office), gate_frames()):
        for frame in frames:
            writer.write(frame)
    writer.release()

    (out / "demo.srt").write_text(SRT, encoding="utf-8")
    with open(out / "qa.jsonl", "w", encoding="utf-8") as f:
        for item in QA:
            row = {**item, "video_id": "demo", "video_path": video_path.name, "subtitle_path": "demo.srt"}
            f.write(json.dumps(row) + "\n")
    print(f"wrote {video_path}, {out / 'demo.srt'}, {out / 'qa.jsonl'}")


if __name__ == "__main__":
    main()
