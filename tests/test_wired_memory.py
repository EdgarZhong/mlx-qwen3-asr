"""验证真实buffer范围、物理锁页生命周期及原生失败回滚；不锁大块测试内存。"""

import ctypes
import gc
import mmap
import weakref
from unittest.mock import Mock

import mlx.core as mx
import numpy as np
import pytest

import mlx_qwen3_asr.wired_memory as wiring


@pytest.fixture
def native(monkeypatch):
    """替换系统调用，保留真实地址及页对齐计算，避免测试影响机器内存。"""
    lib = Mock(mlock=Mock(return_value=0), munlock=Mock(return_value=0))
    monkeypatch.setattr(wiring.sys, "platform", "darwin")
    monkeypatch.setattr(wiring.ctypes, "CDLL", lambda *args, **kwargs: lib)
    return lib


def test_aliases_and_adjacent_pages_are_locked_once(native):
    """同一物理页中的别名及相邻页只能锁一次，预算按页去重而非按tensor相加。"""
    page = mmap.PAGESIZE
    storage = mmap.mmap(-1, page * 3)
    a = np.frombuffer(storage, dtype=np.uint8)
    addr = a.ctypes.data
    lock = wiring.LockedModelWeights([a[3:30], a[8:50], a[page:page * 2]], limit_bytes=page * 2)
    assert native.mlock.call_args_list == [((addr, page * 2),)]
    assert lock.locked_bytes == page * 2
    assert lock.tensor_count == 3
    lock.close()
    lock.close()
    native.munlock.assert_called_once_with(addr, page * 2)
    assert lock.locked_bytes == 0
    del a
    storage.close()


def test_bfloat16_locks_original_bytes_without_conversion(native):
    """MLX的bf16格式字段与itemsize不一致，原始字节buffer必须仍指向原权重。"""
    a = mx.ones((8192,), dtype=mx.bfloat16)
    raw = memoryview(a).cast("B")
    address = ctypes.addressof(ctypes.c_char.from_buffer(raw))
    page = mmap.PAGESIZE
    start = address // page * page
    end = (address + raw.nbytes + page - 1) // page * page
    lock = wiring.LockedModelWeights([a], limit_bytes=page * 4)
    native.mlock.assert_called_once_with(start, end - start)
    assert lock.locked_bytes >= 8192 * 2
    lock.close()


@pytest.mark.parametrize("limit", [0, -1, mmap.PAGESIZE - 1])
def test_insufficient_budget_never_partially_locks(native, limit):
    """不足以覆盖全部页时，在第一个原生调用前拒绝，不能假报部分常驻。"""
    with pytest.raises(ValueError, match="超过预算"):
        wiring.LockedModelWeights([np.ones(1)], limit_bytes=limit)
    native.mlock.assert_not_called()


def test_partial_native_failure_rolls_back(native):
    """第二个不相邻范围锁页失败时，撤销第一个范围并保留原始errno。"""
    page = mmap.PAGESIZE
    storage = mmap.mmap(-1, page * 3)
    a = np.frombuffer(storage, dtype=np.uint8)
    def lock_page(*args):
        if native.mlock.call_count == 2:
            ctypes.set_errno(12)
            return -1
        return 0
    native.mlock.side_effect = lock_page
    with pytest.raises(OSError) as caught:
        wiring.LockedModelWeights([a[:page], a[page * 2:]], limit_bytes=page * 2)
    assert caught.value.errno == 12
    native.munlock.assert_called_once_with(a.ctypes.data, page)
    del caught, a
    gc.collect()
    storage.close()


def test_unlock_failure_preserves_reference_for_retry(native):
    """解锁失败不能丢掉范围或原buffer；下一次close必须能够重试。"""
    a = np.ones(1)
    ref = weakref.ref(a)
    lock = wiring.LockedModelWeights([a], limit_bytes=mmap.PAGESIZE * 2)
    del a
    def fail_unlock(*args):
        ctypes.set_errno(12)
        return -1
    native.munlock.side_effect = fail_unlock
    with pytest.raises(OSError):
        lock.close()
    assert ref() is not None and lock.locked_bytes > 0
    native.munlock.side_effect = None
    lock.close()
    assert ref() is None and lock.locked_bytes == 0


def test_finalizer_releases_abandoned_lock(native):
    """忘记显式cleanup时，锁对象回收仍会对称解锁。"""
    lock = wiring.LockedModelWeights([np.ones(1)], limit_bytes=mmap.PAGESIZE * 2)
    ref = weakref.ref(lock)
    del lock
    gc.collect()
    assert ref() is None
    native.munlock.assert_called_once()


def test_strided_weights_fail_instead_of_locking_a_copy(native):
    """未来若模型改为非连续权重，必须显式报错，不能悄悄复制后锁错对象。"""
    with pytest.raises(ValueError, match="不是C连续"):
        wiring.LockedModelWeights([np.ones(8)[::2]], limit_bytes=mmap.PAGESIZE * 2)
    native.mlock.assert_not_called()


def test_empty_model_is_not_success(native):
    """没有锁住任何buffer不能汇报常驻成功。"""
    with pytest.raises(ValueError, match="没有可锁定"):
        wiring.LockedModelWeights([], limit_bytes=mmap.PAGESIZE)
    native.mlock.assert_not_called()


def test_unsupported_platform_fails_explicitly(native, monkeypatch):
    """不支持的平台不能声称常驻成功。"""
    monkeypatch.setattr(wiring.sys, "platform", "linux")
    with pytest.raises(RuntimeError, match="仅支持macOS"):
        wiring.LockedModelWeights([np.ones(1)], limit_bytes=mmap.PAGESIZE * 2)
    native.mlock.assert_not_called()
