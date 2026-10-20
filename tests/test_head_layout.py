"""GQA layout unit tests: width shrink must not scramble q/kv grouping."""
import types

import pytest

from twyd.tuner import _student_head_layout


def _tc(hidden, heads, kv, head_dim):
    return types.SimpleNamespace(hidden_size=hidden, num_attention_heads=heads,
                                 num_key_value_heads=kv, head_dim=head_dim)


def test_llama3b_r1536_preserves_group_of_3():
    # 24q/8kv (group 3) -> 12q/4kv, each q head keeps its teacher kv head.
    _, _, _, nh, nkv, kv_map = _student_head_layout(_tc(3072, 24, 8, 128), 1536)
    assert (nh, nkv) == (12, 4)
    assert kv_map == [0, 1, 2, 3]
    for q in range(nh):
        assert kv_map[q // 3] == q // 3


def test_llama3b_r1024_duplicates_kv_per_q_head():
    # 8 q heads cannot be split into groups of 3; ratio-1 kv heads are
    # duplicated so q head i still reads teacher kv head floor(i/3).
    _, _, _, nh, nkv, kv_map = _student_head_layout(_tc(3072, 24, 8, 128), 1024)
    assert (nh, nkv) == (8, 8)
    assert kv_map == [0, 0, 0, 1, 1, 1, 2, 2]
    assert [kv_map[q] for q in range(nh)] == [q * 8 // 24 for q in range(nh)]


def test_llama3b_r512():
    _, _, _, nh, nkv, kv_map = _student_head_layout(_tc(3072, 24, 8, 128), 512)
    assert (nh, nkv) == (4, 4)
    assert kv_map == [0, 0, 0, 1]


def test_llama1b_r512_preserves_group_of_4():
    # 32q/8kv -> 8q/2kv; q head i reads teacher kv i//4.
    _, _, _, nh, nkv, kv_map = _student_head_layout(_tc(2048, 32, 8, 64), 512)
    assert (nh, nkv) == (8, 2)
    assert kv_map == [0, 1]
    assert nh % nkv == 0


def test_r_not_multiple_of_head_dim_raises():
    with pytest.raises(ValueError):
        _student_head_layout(_tc(3072, 24, 8, 128), 1000)
