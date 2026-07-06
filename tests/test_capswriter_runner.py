from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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
