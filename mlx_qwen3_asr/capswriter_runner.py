"""CapsWriter 专用的 Qwen3-ASR 中层 Runner。

这个模块刻意放在 ``mlx_qwen3_asr`` package 内，而不是放在
CapsWriter server 适配层里。原因是 prompt、language、generation、
chunking、后续预热和 wired memory 都属于推理管线策略；这些策略如果散落在
server 外层，评测 driver 和真实产品路径就很容易再次分叉。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np

from .audio import SAMPLE_RATE
from .session import Session
from .transcribe import TranscriptionResult


@dataclass(frozen=True)
class CapsWriterRunnerConfig:
    """CapsWriter 场景下集中管理的推理配置。

    这些字段会影响 ASR 输出或输出观测元数据，因此统一由 package 内 Runner
    持有。CapsWriter server 外层只负责把模型路径、音频、语言选择和用户 context
    传进来，不再在外层散落 generation / timestamps / chunks 等推理级参数。
    """

    return_timestamps: bool = False
    return_chunks: bool = True
    max_new_tokens: Optional[int] = None
    num_draft_tokens: int = 4
    verbose: bool = False
    min_audio_seconds: float = 0.1
    enable_startup_prewarm: bool = True
    enable_wired_memory: bool = True
    wired_memory_limit: Union[str, int, None] = "auto"
    wired_memory_auto_ratio: float = 1.2
    wired_memory_auto_max_ratio: float = 0.60
    prewarm_audio_seconds: float = 1.0


@dataclass(frozen=True)
class AudioFeedPatch:
    """同一个 ``task_id`` 下按时间顺序送入 Runner 的音频增量。

    它不是 WebSocket 协议包，也不是模型内部推理 chunk；它只表达“这次完整录音
    又新增了一段连续音频”。真正的 30 秒级 ``InferenceChunk`` 切分继续交给
    package 内现有 ``Session.transcribe()`` / ``split_audio_into_chunks()`` 路径。
    """

    task_id: str
    audio: np.ndarray
    sample_rate: int = SAMPLE_RATE
    is_final: bool = False
    context: str = ""
    language: Optional[str] = None
    source: str = ""


@dataclass(frozen=True)
class QwenASRRunnerResult:
    """Runner 完成一个完整 ``task_id`` 后返回给 CapsWriter 的稳定结果。"""

    task_id: str
    text: str
    language: str
    duration: float
    is_final: bool
    finish_reason: Optional[str] = None
    truncated: bool = False
    segments: Optional[list[dict]] = None
    chunks: Optional[list[dict]] = None


@dataclass
class _TaskAudioState:
    """Runner 内部为单个 ``task_id`` 保存的音频缓冲和请求元信息。"""

    task_id: str
    chunks: list[np.ndarray] = field(default_factory=list)
    sample_count: int = 0
    context: str = ""
    language: Optional[str] = None
    source: str = ""


class QwenASRRunner:
    """CapsWriter 与评测 driver 共用的 Qwen3-ASR 推理入口。

    当前 P0 先实现“流式喂音频、final 后返回完整结果”的可运行闭环。这样 server
    不再拥有 Qwen3-ASR 的语义分段和拼接策略；后续若要做录音过程中提前处理稳定
    ``InferenceChunk``，也可以继续在这个类内部演进，而不需要再改 server 队列层。
    """

    def __init__(
        self,
        model: str,
        *,
        config: Optional[CapsWriterRunnerConfig] = None,
        session: Optional[Session] = None,
    ) -> None:
        self.config = config or CapsWriterRunnerConfig()
        # Session 负责模型和 tokenizer 生命周期；Runner 负责 CapsWriter 的任务生命周期。
        self.session = session or Session(model=model)
        self._states: dict[str, _TaskAudioState] = {}
        self.prewarm_info: dict[str, object] = {}
        self.wired_memory_info: dict[str, object] = {}
        self._previous_wired_limit: Optional[int] = None
        self._startup_initialize_runtime()

    def feed_audio(self, patch: AudioFeedPatch) -> Optional[QwenASRRunnerResult]:
        """接收一段音频增量；只有 final 到达时才返回完整识别结果。"""
        state = self._states.get(patch.task_id)
        if state is None:
            state = _TaskAudioState(task_id=patch.task_id)
            self._states[patch.task_id] = state

        # context / language 以最后一次非空值为准，兼容客户端每包都携带同一元信息的现状。
        if patch.context:
            state.context = patch.context
        if patch.language:
            state.language = patch.language
        if patch.source:
            state.source = patch.source

        audio = self._prepare_audio(patch.audio, patch.sample_rate)
        if audio.size:
            state.chunks.append(audio)
            state.sample_count += int(audio.size)

        if not patch.is_final:
            return None
        return self._finalize_task(patch.task_id)

    def transcribe_audio(
        self,
        audio: np.ndarray,
        *,
        task_id: str,
        sample_rate: int = SAMPLE_RATE,
        context: str = "",
        language: Optional[str] = None,
        source: str = "",
    ) -> QwenASRRunnerResult:
        """一次性识别整段音频，供旧兼容接口和评测短音频路径复用。"""
        result = self.feed_audio(
            AudioFeedPatch(
                task_id=task_id,
                audio=audio,
                sample_rate=sample_rate,
                is_final=True,
                context=context,
                language=language,
                source=source,
            )
        )
        if result is None:
            raise RuntimeError("final 音频提交后 Runner 未返回结果")
        return result

    def cancel_task(self, task_id: str) -> None:
        """客户端断连或任务废弃时释放该 ``task_id`` 的音频缓冲。"""
        self._states.pop(task_id, None)

    def runtime_info(self) -> dict[str, object]:
        """返回最小运行时信息，便于 CapsWriter 记录实际加载路径。"""
        return {
            "runner_module": __file__,
            "model_info": self.session.model_info,
            "config": {
                "return_timestamps": self.config.return_timestamps,
                "return_chunks": self.config.return_chunks,
                "max_new_tokens": self.config.max_new_tokens,
                "num_draft_tokens": self.config.num_draft_tokens,
                "verbose": self.config.verbose,
                "enable_startup_prewarm": self.config.enable_startup_prewarm,
                "enable_wired_memory": self.config.enable_wired_memory,
                "wired_memory_limit": self.config.wired_memory_limit,
            },
            "prewarm_info": dict(self.prewarm_info),
            "wired_memory_info": dict(self.wired_memory_info),
        }

    def cleanup(self) -> None:
        """释放 Runner 持有的任务缓冲，并尽量恢复 MLX wired limit。

        wired limit 是 MLX/Metal 进程级预算。服务端进程通常会直接退出，但显式
        cleanup 可以让开发期模型重载或单测结束时恢复旧额度，避免同一进程内状态残留。
        """
        self._states.clear()
        self._restore_wired_limit_safely()

    def _finalize_task(self, task_id: str) -> QwenASRRunnerResult:
        """拼接完整音频并调用现有 Session 管线生成最终结果。"""
        state = self._states.pop(task_id, None)
        if state is None:
            return QwenASRRunnerResult(
                task_id=task_id,
                text="",
                language="unknown",
                duration=0.0,
                is_final=True,
                finish_reason="missing_task",
                truncated=False,
            )

        audio = self._concat_audio(state)
        duration = float(audio.size / SAMPLE_RATE) if audio.size else 0.0
        if duration < self.config.min_audio_seconds:
            return QwenASRRunnerResult(
                task_id=task_id,
                text="",
                language="unknown",
                duration=duration,
                is_final=True,
                finish_reason="empty_audio",
                truncated=False,
            )

        transcription = self._transcribe_prepared_audio(
            audio,
            context=state.context,
            language=state.language,
        )
        return QwenASRRunnerResult(
            task_id=task_id,
            text=(transcription.text or "").strip(),
            language=getattr(transcription, "language", "unknown") or "unknown",
            duration=duration,
            is_final=True,
            finish_reason=getattr(transcription, "finish_reason", None),
            truncated=bool(getattr(transcription, "truncated", False)),
            segments=getattr(transcription, "segments", None),
            chunks=getattr(transcription, "chunks", None),
        )

    def _transcribe_prepared_audio(
        self,
        audio: np.ndarray,
        *,
        context: str,
        language: Optional[str],
    ) -> TranscriptionResult:
        """把集中配置一次性应用到 package 现有高层推理管线。"""
        return self.session.transcribe(
            audio,
            context=context or "",
            language=language,
            return_timestamps=self.config.return_timestamps,
            return_chunks=self.config.return_chunks,
            max_new_tokens=self.config.max_new_tokens,
            num_draft_tokens=self.config.num_draft_tokens,
            verbose=self.config.verbose,
        )

    def _startup_initialize_runtime(self) -> None:
        """按“先预热、后 wired”的顺序初始化运行态。

        预热会让 MLX 完成首次 kernel 编译和关键缓存初始化；随后读取 active memory
        再设置 wired limit，auto 策略才有接近当前模型规格的基准值。
        """
        if self.config.enable_startup_prewarm:
            self._prewarm_safely()
        else:
            self.prewarm_info = {"enabled": False}

        if self.config.enable_wired_memory:
            self._configure_wired_memory_safely()
        else:
            self.wired_memory_info = {"enabled": False}

    def _prewarm_safely(self) -> None:
        """用极短静音走一遍真实 transcribe 路径；失败只记录状态，不阻断启动。"""
        t0 = time.time()
        try:
            seconds = max(float(self.config.prewarm_audio_seconds), self.config.min_audio_seconds)
            sample_count = max(int(round(seconds * SAMPLE_RATE)), 1)
            # 使用很低幅度的静音数组，不引入外部音频文件，也不改变用户输入链路。
            warmup_audio = np.zeros(sample_count, dtype=np.float32)
            result = self._transcribe_prepared_audio(
                warmup_audio,
                context="",
                language=None,
            )
            self.prewarm_info = {
                "enabled": True,
                "ok": True,
                "seconds": seconds,
                "cost_sec": time.time() - t0,
                "finish_reason": getattr(result, "finish_reason", None),
                "truncated": bool(getattr(result, "truncated", False)),
            }
        except Exception as exc:
            self.prewarm_info = {
                "enabled": True,
                "ok": False,
                "cost_sec": time.time() - t0,
                "error": repr(exc),
            }

    def _configure_wired_memory_safely(self) -> None:
        """设置 MLX wired limit；不支持或失败时只记录状态，不让 server 启动失败。"""
        try:
            import mlx.core as mx

            set_wired_limit = getattr(mx, "set_wired_limit", None)
            if not callable(set_wired_limit):
                self.wired_memory_info = {
                    "enabled": True,
                    "ok": False,
                    "reason": "mlx.core.set_wired_limit unavailable",
                }
                return

            active = int(getattr(mx, "get_active_memory")())
            device_info_fn = getattr(mx, "device_info", None)
            device_info = device_info_fn() if callable(device_info_fn) else {}
            memory_size = int(device_info.get("memory_size") or 0)
            recommended = int(device_info.get("max_recommended_working_set_size") or 0)
            limit = self._resolve_wired_limit(
                active_bytes=active,
                memory_size_bytes=memory_size,
                recommended_bytes=recommended,
            )
            if limit <= 0:
                self.wired_memory_info = {
                    "enabled": True,
                    "ok": False,
                    "reason": "resolved limit <= 0",
                    "active_bytes": active,
                }
                return

            previous = int(set_wired_limit(limit))
            self._previous_wired_limit = previous
            self.wired_memory_info = {
                "enabled": True,
                "ok": True,
                "limit_bytes": limit,
                "previous_limit_bytes": previous,
                "active_bytes": active,
                "memory_size_bytes": memory_size,
                "recommended_bytes": recommended,
            }
        except Exception as exc:
            self.wired_memory_info = {
                "enabled": True,
                "ok": False,
                "error": repr(exc),
            }

    def _resolve_wired_limit(
        self,
        *,
        active_bytes: int,
        memory_size_bytes: int,
        recommended_bytes: int,
    ) -> int:
        """解析 wired limit。

        `auto` 不是固定占用内存，而是根据预热后的 MLX active memory 给一个常驻预算。
        预算会低于系统推荐 working set，也会按总内存比例收口，避免把额度开得过大。
        """
        configured = self.config.wired_memory_limit
        if isinstance(configured, int):
            return int(configured)
        if isinstance(configured, str) and configured.strip().lower() != "auto":
            return self._parse_size_to_bytes(configured)

        base = int(active_bytes * float(self.config.wired_memory_auto_ratio))
        caps = [value for value in (memory_size_bytes, recommended_bytes) if value > 0]
        if memory_size_bytes > 0:
            caps.append(int(memory_size_bytes * float(self.config.wired_memory_auto_max_ratio)))
        if recommended_bytes > 0:
            # 官方建议 wired limit 不应超过 max_recommended_working_set_size；
            # 这里再留一点余量，避免用户机器状态变化时顶到系统错误。
            caps.append(int(recommended_bytes * 0.90))
        if caps:
            base = min(base, *caps)
        return max(base, 0)

    @staticmethod
    def _parse_size_to_bytes(value: str) -> int:
        """解析用户可读大小字符串，例如 ``3g``、``3072m`` 或纯字节数。"""
        text = value.strip().lower()
        if not text:
            return 0
        units = {
            "k": 1024,
            "kb": 1024,
            "m": 1024 ** 2,
            "mb": 1024 ** 2,
            "g": 1024 ** 3,
            "gb": 1024 ** 3,
        }
        for suffix, multiplier in sorted(units.items(), key=lambda item: len(item[0]), reverse=True):
            if text.endswith(suffix):
                number = float(text[: -len(suffix)].strip())
                return int(number * multiplier)
        return int(text)

    def _restore_wired_limit_safely(self) -> None:
        """尽量恢复进入 Runner 前的 wired limit；失败不影响主流程退出。"""
        if self._previous_wired_limit is None:
            return
        try:
            import mlx.core as mx

            set_wired_limit = getattr(mx, "set_wired_limit", None)
            if callable(set_wired_limit):
                set_wired_limit(int(self._previous_wired_limit))
        except Exception:
            return
        finally:
            self._previous_wired_limit = None

    @staticmethod
    def _concat_audio(state: _TaskAudioState) -> np.ndarray:
        """合并同一 ``task_id`` 下已经按时序进入 Runner 的音频块。"""
        if not state.chunks:
            return np.array([], dtype=np.float32)
        if len(state.chunks) == 1:
            return state.chunks[0].astype(np.float32, copy=False)
        return np.concatenate(state.chunks).astype(np.float32, copy=False)

    @staticmethod
    def _prepare_audio(audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """整理输入音频到 16kHz mono float32，避免依赖 ffmpeg 做内存数组重采样。"""
        normalized = np.asarray(audio, dtype=np.float32)
        if normalized.size == 0:
            return np.array([], dtype=np.float32)
        if normalized.ndim > 1:
            normalized = normalized.mean(axis=1)
        if sample_rate == SAMPLE_RATE:
            return normalized.astype(np.float32, copy=False)
        return QwenASRRunner._resample_audio_linear(normalized, sample_rate, SAMPLE_RATE)

    @staticmethod
    def _resample_audio_linear(
        audio: np.ndarray,
        source_sample_rate: int,
        target_sample_rate: int,
    ) -> np.ndarray:
        """使用线性插值兜底重采样，保证 server 无 ffmpeg 时也能跑通。"""
        if source_sample_rate <= 0 or target_sample_rate <= 0:
            raise ValueError(
                f"非法采样率: source={source_sample_rate}, target={target_sample_rate}"
            )
        if audio.size == 0:
            return np.array([], dtype=np.float32)

        target_size = int(round(audio.size * target_sample_rate / source_sample_rate))
        if target_size <= 0:
            return np.array([], dtype=np.float32)

        source_positions = np.linspace(0.0, 1.0, num=audio.size, endpoint=False)
        target_positions = np.linspace(0.0, 1.0, num=target_size, endpoint=False)
        return np.interp(target_positions, source_positions, audio).astype(np.float32)
