"""锁住 CapsWriter 正文前缀的真实调用链，避免诊断钩子掩盖参数漏传。

只替换昂贵的声学编码和生成；Runner、Session、提示词组装和结果解析均走生产代码。
"""

import asyncio
import importlib
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mlx_qwen3_asr import CapsWriterRunnerConfig, QwenASRRunner, Session
from mlx_qwen3_asr.tokenizer import Tokenizer


class _CharacterTokenizer(Tokenizer):
    """用字符编号替代 BPE，但复用生产 build_prompt_tokens 验证模板边界。"""

    def __init__(self):
        self._system_prefix_tokens = self.encode("system\n")
        self._user_tokens = self.encode("user\n")
        self._assistant_tokens = self.encode("assistant\n")
        self._newline_id = ord("\n")
        self.AUDIO_START_TOKEN_ID = 151669
        self.AUDIO_TOKEN_ID = 151676
        self.AUDIO_END_TOKEN_ID = 151670

    def encode(self, text, add_special_tokens=False):
        return list(map(ord, text))

    def decode(self, ids, skip_special_tokens=False):
        return "".join(map(chr, ids))


@pytest.fixture
def inference(monkeypatch):
    """创建真实 Session，生成器保存最终 prompt，允许测试逐次换模式。"""
    tmod = importlib.import_module("mlx_qwen3_asr.transcribe")
    smod = importlib.import_module("mlx_qwen3_asr.session")
    session = Session.__new__(Session)
    session.dtype = mx.float16
    session.tokenizer = _CharacterTokenizer()
    session.model = SimpleNamespace(
        config=SimpleNamespace(support_languages=[]),
        audio_tower=lambda *args: (mx.zeros((1, 2, 4)), None),
    )
    captured = []
    monkeypatch.setattr(tmod, "compute_features", lambda audio: (mx.zeros((1, 2, 4)), None))

    def generate(**kwargs):
        captured.append(kwargs["input_ids"].tolist()[0])
        return session.tokenizer.encode("Mailbox 是什么？")

    monkeypatch.setattr(tmod, "generate", generate)
    # 对齐路径用空结果替身，测试重点是保留语言识别所需的旧 prompt。
    monkeypatch.setattr(smod, "_resolve_aligner", lambda *args: None)
    return session, captured


def test_runner_defaults_to_text_without_changing_bare_session(inference):
    """同一个 Session 先裸调用、再经 Runner；配置不得污染 tokenizer 缓存。"""
    session, captured = inference
    audio = np.ones(3200, dtype=np.float32)
    original = session.tokenizer.build_prompt_tokens(2)
    session.transcribe(audio)
    runner = QwenASRRunner("unused", session=session, config=CapsWriterRunnerConfig(
        enable_startup_prewarm=False, enable_wired_memory=False))
    result = runner.transcribe_audio(audio, task_id="mixed")
    session.transcribe(audio)
    assert captured == [original, original + session.tokenizer.encode("<asr_text>"), original]
    assert result.text == "Mailbox 是什么？"
    assert result.language == "unknown"


@pytest.mark.parametrize("language", [None, "zh", "en"])
@pytest.mark.parametrize("enabled", [False, True])
def test_switch_and_explicit_language(inference, language, enabled):
    """开关能恢复原路径，显式语言优先且不得重复追加正文分隔符。"""
    session, captured = inference
    result = session.transcribe(np.ones(3200, dtype=np.float32), language=language,
                                auto_language_text_only=enabled, context="术语")
    expected = session.tokenizer.build_prompt_tokens(2, language=language, context="术语")
    if enabled and language is None:
        expected += session.tokenizer.encode("<asr_text>")
    assert captured == [expected]
    assert result.language == {None: "unknown", "zh": "Chinese", "en": "English"}[language]


@pytest.mark.parametrize("alignment", [{"return_timestamps": True}, {"diarize": True}])
def test_alignment_retains_language_detection(inference, monkeypatch, alignment):
    """时间戳/说话人分段需要语言标签，不能因正文模式静默丢失对齐结果。"""
    session, captured = inference
    tmod = importlib.import_module("mlx_qwen3_asr.transcribe")
    monkeypatch.setattr(tmod, "infer_speaker_turns", lambda *args, **kwargs: [])
    monkeypatch.setattr(tmod, "diarize_word_segments", lambda *args, **kwargs: ([], []))
    session.transcribe(np.ones(3200, dtype=np.float32), auto_language_text_only=True, **alignment)
    assert captured == [session.tokenizer.build_prompt_tokens(2)]


def test_async_and_every_chunk_receive_text_prefix(inference, monkeypatch):
    """异步包装和内部切段都应透传开关，不能仅在第一个 chunk 上修复。"""
    session, captured = inference
    tmod = importlib.import_module("mlx_qwen3_asr.transcribe")
    monkeypatch.setattr(tmod, "split_audio_into_chunks", lambda audio, sr: [(audio, 0), (audio, 1)])
    result = asyncio.run(session.transcribe_async(np.ones(3200, dtype=np.float32),
                                                 auto_language_text_only=True, return_chunks=True))
    expected = session.tokenizer.build_prompt_tokens(2) + session.tokenizer.encode("<asr_text>")
    assert captured == [expected, expected]
    assert len(result.chunks) == 2
    assert all(chunk["language"] == "unknown" for chunk in result.chunks)


@pytest.mark.parametrize("chunks, expected", [
    (["这是中文", "下一段"], "这是中文下一段"),
    (["这是中文。", "下一段"], "这是中文。下一段"),
    (["Hello", "world"], "Hello world"),
    (["中文 hello", "world"], "中文 hello world"),
    (["Hello.", "Next sentence."], "Hello. Next sentence."),
    (["中文", "", "后续"], "中文后续"),
])
def test_text_only_chunk_join_preserves_word_boundaries(chunks, expected):
    """无语言标签时只处理拼接边界，不能把汉字拆开或把相邻英文词黏在一起。"""
    tmod = importlib.import_module("mlx_qwen3_asr.transcribe")
    assert tmod._join_chunk_texts(chunks, "unknown", text_only=True) == expected
