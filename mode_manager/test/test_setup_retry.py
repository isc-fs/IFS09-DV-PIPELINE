"""The ~/setup call asks again when a reply is lost (#25).

A node can answer ~/setup before it has discovered a brand-new client's
reply reader; DDS then drops the reply. mode_manager used to wait the whole
_SETUP_TIMEOUT_S (60 s) for it and fail the mission. It now asks again on
the same client after _SETUP_REPLY_TIMEOUT_S.

Runs _call_setup_with_client against a fake client: no DDS, no spinning.
"""
from __future__ import annotations

import os
import sys
import time
from concurrent.futures import Future
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

# A plain import: importorskip at module level aborts the whole collection
# under the pytest 6 that ROS Humble ships.
import mode_manager.mode_manager_node as mmn  # noqa: E402


class FakeClient:
    """Answers each call_async only if `answers` says so, in order."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = 0
        self.removed = 0

    def wait_for_service(self, timeout_sec):
        return True

    def call_async(self, req):
        self.calls += 1
        fut = Future()
        answer = self.answers.pop(0) if self.answers else False
        if answer:
            fut.set_result(SimpleNamespace(success=True, message="Ready"))
        return fut

    def remove_pending_request(self, fut):
        self.removed += 1


def fake_node():
    log = SimpleNamespace(warnings=[], info=lambda msg: None)
    log.warning = log.warnings.append
    progress = []
    node = SimpleNamespace(
        _wait_for_future=mmn.ModeManagerNode._wait_for_future,
        _publish_progress=lambda *a: progress.append(a),
        get_logger=lambda: log,
    )
    return node, log, progress


@pytest.fixture
def short_timeouts(monkeypatch):
    monkeypatch.setattr(mmn, "_SETUP_REPLY_TIMEOUT_S", 0.05)
    monkeypatch.setattr(mmn, "_SETUP_TIMEOUT_S", 0.5)


def call(node, cli):
    return mmn.ModeManagerNode._call_setup_with_client(
        node, cli, "slam_node", "accel", "accel",
    )


def test_a_lost_reply_is_asked_for_again_on_the_same_client(short_timeouts):
    node, log, progress = fake_node()
    cli = FakeClient([False, True])

    t0 = time.monotonic()
    ok, err = call(node, cli)

    assert (ok, err) == (True, "")
    assert cli.calls == 2
    assert cli.removed == 1
    assert len(log.warnings) == 1
    assert progress[-1][2] == "ok"
    assert time.monotonic() - t0 < 0.4


def test_a_first_reply_is_not_asked_for_twice(short_timeouts):
    node, log, _ = fake_node()
    cli = FakeClient([True])

    assert call(node, cli) == (True, "")
    assert cli.calls == 1
    assert log.warnings == []


def test_a_node_that_never_answers_still_fails_at_the_setup_timeout(short_timeouts):
    node, _, progress = fake_node()
    cli = FakeClient([])

    t0 = time.monotonic()
    ok, err = call(node, cli)
    took = time.monotonic() - t0

    assert not ok
    assert err.startswith("timeout waiting for setup response")
    assert cli.calls >= 2
    assert progress[-1][2] == "timeout"
    assert 0.45 <= took < 1.0
