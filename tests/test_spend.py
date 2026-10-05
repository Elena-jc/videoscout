from __future__ import annotations

import pytest
from conftest import FakeAnthropic, text

from videoscout.llm import ClaudeLLM
from videoscout.spend import SpendGuard, SpendLimitError


@pytest.fixture(autouse=True)
def isolated_ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("VIDEOSCOUT_USAGE_DIR", str(tmp_path))


def test_request_cap_blocks_the_next_request(cfg):
    cfg.models.daily_request_limit = 2
    client = FakeAnthropic(planner=[[text("a")], [text("b")], [text("c")]])
    llm = ClaudeLLM(cfg.models, cfg.pricing, client=client)
    for _ in range(2):
        llm.create("planner", [{"role": "user", "content": "hi"}], tools=[{"name": "x"}])
    with pytest.raises(SpendLimitError, match="2 requests"):
        llm.create("planner", [{"role": "user", "content": "hi"}], tools=[{"name": "x"}])
    assert len(client.requests) == 2  # the blocked request was never sent


def test_cost_cap_and_shared_ledger(cfg):
    cfg.models.daily_cost_limit_usd = 0.01  # each fake call costs 0.0081
    llm = ClaudeLLM(cfg.models, cfg.pricing, client=FakeAnthropic(planner=[[text("a")], [text("b")], [text("c")]]))
    llm.create("planner", [{"role": "user", "content": "hi"}], tools=[{"name": "x"}])
    # A second client (another question, another process) sees the same ledger.
    other = llm.fresh()
    other.create("planner", [{"role": "user", "content": "hi"}], tools=[{"name": "x"}])
    with pytest.raises(SpendLimitError, match=r"\$0\.02"):
        other.create("planner", [{"role": "user", "content": "hi"}], tools=[{"name": "x"}])
    assert SpendGuard("anthropic").today()["requests"] == 2


def test_no_cap_means_unlimited_but_still_counted():
    guard = SpendGuard("Some Host.example")
    for _ in range(5):
        guard.check()
        guard.record(0.0)
    assert guard.name == "some-host-example" and guard.today()["requests"] == 5
