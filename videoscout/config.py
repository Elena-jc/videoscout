"""Typed configuration loaded from YAML, with dotted CLI overrides."""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"
WEIGHTS_DIR = Path(os.environ.get("VIDEOSCOUT_WEIGHTS_DIR") or PROJECT_ROOT / "weights")


def weights_path(name: str) -> str:
    """Bare checkpoint names (e.g. 'yolo26s.pt') live in the project's weights/
    folder; Ultralytics downloads a missing release asset to exactly this path."""
    path = Path(name)
    if path.is_absolute() or path.parent != Path("."):
        return str(path)
    WEIGHTS_DIR.mkdir(exist_ok=True)
    return str(WEIGHTS_DIR / path)


def load_dotenv(path: Path | None = None) -> None:
    """Read KEY=VALUE lines from the project's .env into the environment.
    Variables that are already set win, and .env is git-ignored."""
    path = path or PROJECT_ROOT / ".env"
    if os.environ.get("VIDEOSCOUT_NO_DOTENV") or not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        if key and value:
            os.environ.setdefault(key, value)


@dataclass
class ModelConfig:
    provider: str = "anthropic"  # anthropic | openai_compat (Gemini, Ollama, OpenRouter, ...)
    base_url: str = ""  # openai_compat only, e.g. http://localhost:11434/v1
    api_key_env: str = ""  # openai_compat only: name of the env var holding the key
    json_mode: str = "schema"  # openai_compat only: schema | object
    min_request_interval: float = 0.0  # seconds between requests (free-tier rate limits)
    send_reasoning_effort: bool = False  # openai_compat only: pass <role>_effort as reasoning_effort
    planner: str = "claude-opus-5-5"
    verifier: str = "claude-opus-5-5"
    vision: str = "claude-opus-5-5"
    captioner: str = "claude-opus-5-5"
    planner_effort: str = "medium"
    verifier_effort: str = "medium"
    vision_effort: str = "low"
    captioner_effort: str = "low"
    max_tokens: int = 16000
    use_fallbacks: bool = True
    daily_request_limit: int = 0  # local safety caps (0 = none), see videoscout/spend.py
    daily_cost_limit_usd: float = 0.0


@dataclass
class IndexConfig:
    segment_seconds: float = 10.0
    keyframe_every: float = 2.0
    track_fps: float = 2.0
    yolo_weights: str = "yolo26s.pt"
    yolo_conf: float = 0.3
    tracker: str = "bytetrack.yaml"  # or botsort.yaml (camera-motion compensation, fewer ID switches)
    siglip_model: str = "google/siglip2-so400m-patch14-384"
    device: str = "auto"
    caption: bool = False
    caption_mode: str = "batch"


@dataclass
class RetrievalConfig:
    use_dense: bool = True
    use_bm25: bool = True
    rrf_k: int = 60
    candidate_pool: int = 40
    mmr_lambda: float = 0.7
    mmr_tau_seconds: float = 30.0


@dataclass
class GroundingConfig:
    """Open-vocabulary detector behind the find_objects tool (query time)."""

    weights: str = "yoloe-26s-seg.pt"  # empty string disables the tool
    conf: float = 0.25
    max_frames: int = 16
    default_frames: int = 8


@dataclass
class AgentConfig:
    tools: str = "inprocess"
    disabled_tools: str = ""  # comma-separated, for ablations, e.g. "query_tracks"
    strict_tools: bool = True
    max_tool_calls: int = 12
    max_frames: int = 48
    max_frames_per_inspect: int = 8
    default_frames_per_inspect: int = 6
    verify: bool = True
    max_verify_rounds: int = 2
    confidence_threshold: float = 0.6
    max_llm_turns: int = 20


@dataclass
class Config:
    models: ModelConfig = field(default_factory=ModelConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    grounding: GroundingConfig = field(default_factory=GroundingConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    pricing: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _coerce(value: str, current: Any) -> Any:
    """Parse a CLI override string using the type of the current value."""
    if isinstance(current, bool):
        if value.lower() in ("1", "true", "yes", "on"):
            return True
        if value.lower() in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"expected a boolean, got {value!r}")
    if isinstance(current, int):
        return int(value)
    if isinstance(current, float):
        return float(value)
    return value


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    load_dotenv()
    raw: dict[str, Any] = {}
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    if cfg_path.exists():
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

    cfg = Config(
        models=ModelConfig(**raw.get("models", {})),
        index=IndexConfig(**raw.get("index", {})),
        retrieval=RetrievalConfig(**raw.get("retrieval", {})),
        grounding=GroundingConfig(**raw.get("grounding", {})),
        agent=AgentConfig(**raw.get("agent", {})),
        pricing=raw.get("pricing", {}),
    )

    for item in overrides or []:
        key, sep, value = item.partition("=")
        section, dot, name = key.strip().partition(".")
        if not sep or not dot:
            raise ValueError(f"override must look like section.field=value, got {item!r}")
        target = getattr(cfg, section, None)
        if target is None or not dataclasses.is_dataclass(target) or not hasattr(target, name):
            raise ValueError(f"unknown config field {key!r}")
        setattr(target, name, _coerce(value.strip(), getattr(target, name)))
    return cfg
