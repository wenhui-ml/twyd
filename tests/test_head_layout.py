"""Attention-layout unit tests: LRC keeps teacher heads, shrinks only residual.

Verified against the released LRC configs:
  LRC-1.5B-Base: hidden 1536, 24 heads / 8 kv, head_dim 128 (Llama-3.2-3B)
  LRC-1.7B-Base: hidden 1200, 16 heads / 2 kv, head_dim 128 (Qwen2.5-3B)
Head count is unchanged; q/k/v output width stays at the teacher's H.
"""
import types

import pytest

from twyd.tuner import _student_head_layout


def _tc(hidden, heads, kv, head_dim):
    return types.SimpleNamespace(hidden_size=hidden, num_attention_heads=heads,
                                 num_key_value_heads=kv, head_dim=head_dim)


@pytest.mark.parametrize("r", [1536, 1024, 512, 768])
def test_llama3b_keeps_teacher_heads(r):
    heads, kv, hd, nh, nkv, kv_map = _student_head_layout(_tc(3072, 24, 8, 128), r)
    assert (heads, kv, hd) == (24, 8, 128)
    assert (nh, nkv) == (24, 8)
    assert kv_map == list(range(8))
    assert nh * hd == 3072  # q/k/v/o stay at teacher width


def test_llama1b_keeps_teacher_heads():
    heads, kv, hd, nh, nkv, kv_map = _student_head_layout(_tc(2048, 32, 8, 64), 512)
    assert (heads, kv, hd) == (32, 8, 64)
    assert (nh, nkv) == (32, 8)
    assert nh * hd == 2048


def test_qwen25_3b_keeps_teacher_heads():
    heads, kv, hd, nh, nkv, kv_map = _student_head_layout(_tc(2048, 16, 2, 128), 1200)
    assert (heads, kv, hd) == (16, 2, 128)
    assert (nh, nkv) == (16, 2)
    assert nh * hd == 2048


def test_bad_head_dim_raises():
    with pytest.raises(ValueError):
        _student_head_layout(_tc(0, 0, 0, 0), 512)
    with pytest.raises(ValueError):
        _student_head_layout(_tc(3072, 24, 8, 128), 0)
