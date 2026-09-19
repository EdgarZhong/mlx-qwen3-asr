"""macOS 模型权重的物理锁页：直接锁原始buffer，避免仅有Metal预算而空闲可换出。"""

from __future__ import annotations

import ctypes
import os
import sys
import weakref
from collections.abc import Iterable


def _unlock_pages(lib, ranges: list[tuple[int, int]], views: list[memoryview]) -> None:
    """逐段解除本对象持有的锁；失败保留范围与引用，允许显式close再次尝试。"""
    errors = []
    for start, end in reversed(ranges.copy()):
        if lib.munlock(start, end - start) == 0:
            ranges.remove((start, end))
        else:
            code = ctypes.get_errno()
            errors.append(OSError(code, os.strerror(code)))
    if not ranges:
        views.clear()
    if errors:
        raise errors[0]


class LockedModelWeights:
    """持有权重原始页的mlock及buffer引用，生命周期与Runner一致。

    Metal驻留集合在本机GPU空闲时未能保持系统wired计数；mlock直接对物理页
    建立不可分页约束。仅锁模型参数，不锁推理临时张量，也不建立保活线程。
    """

    def __init__(self, arrays: Iterable[object], *, limit_bytes: int) -> None:
        if sys.platform != "darwin":
            raise RuntimeError("权重物理锁页目前仅支持macOS")
        page_size = os.sysconf("SC_PAGE_SIZE")
        self._views: list[memoryview] = []
        self._ranges: list[tuple[int, int]] = []
        candidates = []
        for array in arrays:
            # memoryview使用MLX的原始buffer协议；不经过numpy数值转换或复制。
            # bfloat16导出的格式为B、itemsize却是2，必须先cast为原始字节。
            view = memoryview(array)
            if not view.nbytes:
                continue
            if not view.c_contiguous:
                raise ValueError("模型参数不是C连续buffer，拒绝锁定可能的转换副本")
            raw = view.cast("B")
            address = ctypes.addressof(ctypes.c_char.from_buffer(raw))
            start = address // page_size * page_size
            end = (address + raw.nbytes + page_size - 1) // page_size * page_size
            candidates.append((start, end))
            # 直到munlock成功都保留引用，防止锁页地址已被释放或分配给其它张量。
            self._views.append(raw)
        self.tensor_count = len(self._views)
        if not candidates:
            raise ValueError("模型没有可锁定的权重buffer")

        merged: list[tuple[int, int]] = []
        for start, end in sorted(candidates):
            # 共享权重、同页小张量及相邻页合并，避免重复锁同一页和重复计费。
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
            else:
                merged.append((start, end))
        required = sum(end - start for start, end in merged)
        if limit_bytes <= 0 or required > limit_bytes:
            raise ValueError(f"完整权重锁页需要{required}字节，超过预算{limit_bytes}字节")

        self._lib = ctypes.CDLL(None, use_errno=True)
        for name in ("mlock", "munlock"):
            function = getattr(self._lib, name)
            function.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            function.restype = ctypes.c_int
        # finalizer不捕获self；异常构造或忘记cleanup时仍有成对解锁的兜底。
        self._finalizer = weakref.finalize(
            self, _unlock_pages, self._lib, self._ranges, self._views)
        try:
            for start, end in merged:
                if self._lib.mlock(start, end - start) != 0:
                    code = ctypes.get_errno()
                    raise OSError(code, os.strerror(code))
                self._ranges.append((start, end))
        except BaseException:
            # 不保留“只锁了一部分”的伪成功，失败即回滚已成功的范围并向上传播。
            self.close()
            raise

    @property
    def locked_bytes(self) -> int:
        """返回去重、按页对齐且当前仍由本对象锁定的真实字节数。"""
        return sum(end - start for start, end in self._ranges)

    @property
    def range_count(self) -> int:
        """返回当前持有的原生锁页范围数。"""
        return len(self._ranges)

    def close(self) -> None:
        """成对释放锁页，允许重复调用；失败时保留引用和finalizer供重试。"""
        _unlock_pages(self._lib, self._ranges, self._views)
        self._finalizer.detach()
