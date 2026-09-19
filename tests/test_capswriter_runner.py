from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

import mlx_qwen3_asr.capswriter_runner as runner_module
from mlx_qwen3_asr.capswriter_runner import (
    AudioFeedPatch,
    CapsWriterRunnerConfig,
    QwenASRRunner,
)


@dataclass
class _FakeTranscription:
    text: str = "测试通过"
    language: str = "Chinese"
    finish_reason: str = "eos"
    truncated: bool = False
    chunks: list[dict] | None = None
    segments: list[dict] | None = None


class _FakeSession:
    """避免单测加载真实模型，只记录 Runner 传给 Session 的参数。"""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.model_info = {"model_id": "fake"}

    def transcribe(self, audio, **kwargs):
        self.calls.append({"audio": audio, "kwargs": kwargs})
        return _FakeTranscription(chunks=[{"text": "测试通过"}])


def test_capswriter_runner_buffers_until_final() -> None:
    session = _FakeSession()
    runner = QwenASRRunner(
        model="fake-model",
        session=session,
        config=CapsWriterRunnerConfig(
            return_chunks=True,
            max_new_tokens=123,
            enable_startup_prewarm=False,
            enable_wired_memory=False,
        ),
    )

    first = np.ones(1600, dtype=np.float32)
    second = np.ones(3200, dtype=np.float32) * 2

    assert runner.feed_audio(
        AudioFeedPatch(task_id="task-1", audio=first, is_final=False)
    ) is None
    result = runner.feed_audio(
        AudioFeedPatch(
            task_id="task-1",
            audio=second,
            is_final=True,
            context="技术词",
            language="Chinese",
        )
    )

    assert result is not None
    assert result.text == "测试通过"
    assert result.duration == 0.3
    assert len(session.calls) == 1
    np.testing.assert_array_equal(
        session.calls[0]["audio"],
        np.concatenate([first, second]),
    )
    assert session.calls[0]["kwargs"]["context"] == "技术词"
    assert session.calls[0]["kwargs"]["language"] == "Chinese"
    assert session.calls[0]["kwargs"]["max_new_tokens"] == 123
    assert session.calls[0]["kwargs"]["return_chunks"] is True


def test_capswriter_runner_resamples_in_memory_audio_without_ffmpeg() -> None:
    session = _FakeSession()
    runner = QwenASRRunner(
        model="fake-model",
        session=session,
        config=CapsWriterRunnerConfig(
            enable_startup_prewarm=False,
            enable_wired_memory=False,
        ),
    )

    audio_48k = np.ones(4800, dtype=np.float32)
    result = runner.feed_audio(
        AudioFeedPatch(
            task_id="task-2",
            audio=audio_48k,
            sample_rate=48000,
            is_final=True,
        )
    )

    assert result is not None
    assert result.duration == 0.1
    assert len(session.calls) == 1
    assert session.calls[0]["audio"].shape == (1600,)


@pytest.fixture
def memory_runtime(monkeypatch):
    """隔离进程级Metal额度和真实锁页，检查Runner是否选择正确的生命周期。"""
    events = []
    monkeypatch.setattr(mx, "set_wired_limit", lambda n: events.append(("metal", n)) or 0)
    monkeypatch.setattr(mx, "get_active_memory", lambda: 10000)
    monkeypatch.setattr(mx, "device_info", lambda: {
        "memory_size": 100000, "max_recommended_working_set_size": 80000})

    class FakeLock:
        """仅模拟成功持有的锁页资源，底层页范围与错误回滚由专项测试验证。"""
        locked_bytes = 8192
        tensor_count = 1
        range_count = 1

        def __init__(self, arrays, *, limit_bytes):
            events.append(("lock", limit_bytes, len(list(arrays))))

        def close(self):
            events.append(("unlock",))

    monkeypatch.setattr(runner_module, "LockedModelWeights", FakeLock, raising=False)
    session = _FakeSession()
    session.model = SimpleNamespace(parameters=lambda: {"weight": np.ones(1024)})
    return session, events


def test_resident_mode_requires_actual_weight_lock(memory_runtime):
    """回归旧实现仅设置Metal预算便宣称常驻成功的问题。"""
    session, events = memory_runtime
    runner = QwenASRRunner("fake", session=session)
    assert events == [("lock", 12000, 1)]
    assert runner.wired_memory_info["method"] == "mlock"
    assert runner.wired_memory_info["locked_bytes"] == 8192
    assert runner.wired_memory_info["ok"] is True
    runner.cleanup()
    runner.cleanup()
    assert events[-1] == ("unlock",)
    assert events.count(("unlock",)) == 1
    assert runner.wired_memory_info["ok"] is False


def test_pageable_mode_never_touches_memory_locking(memory_runtime):
    """关闭开关允许系统管理内存，预热仍执行，但不调用任何额度或锁页接口。"""
    session, events = memory_runtime
    runner = QwenASRRunner("fake", session=session,
        config=CapsWriterRunnerConfig(enable_wired_memory=False))
    runner.cleanup()
    assert events == []
    assert len(session.calls) == 1
    assert runner.wired_memory_info == {"enabled": False}


def test_resident_mode_fails_startup_when_lock_fails(memory_runtime, monkeypatch):
    """常驻承诺不能在原生锁页失败后静默降级为可换出模式。"""
    session, events = memory_runtime
    def fail_lock(*args, **kwargs):
        raise OSError(12, "injected lock failure")
    monkeypatch.setattr(runner_module, "LockedModelWeights", fail_lock)
    with pytest.raises(RuntimeError, match="权重锁页失败"):
        QwenASRRunner("fake", session=session)
    assert events == []


def test_resident_mode_without_prewarm_still_locks(memory_runtime):
    """预热和常驻是独立配置，关闭预热不能使常驻开关失效。"""
    session, events = memory_runtime
    runner = QwenASRRunner("fake", session=session,
        config=CapsWriterRunnerConfig(enable_startup_prewarm=False, wired_memory_limit="16k"))
    assert events == [("lock", 16384, 1)]
    assert session.calls == []
    runner.cleanup()
