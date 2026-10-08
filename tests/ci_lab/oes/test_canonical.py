import math

import pytest

from ci_lab.oes import canonical

DOC = {"b": [1, 2.5, None, True], "a": {"z": "é", "y": 0.1}, "contentHash": "sha256:stale"}
# Golden value: changing canonicalisation invalidates every sealed envelope on disk.
GOLDEN = "sha256:790907390a4006400ef8729b2299f3be5adbcbe473b9719f536fab85a692d04c"


def test_canonical_json_is_sorted_compact_utf8():
    assert canonical.canonical_json(DOC) == ('{"a":{"y":0.1,"z":"é"},"b":[1,2.5,null,true],'
                                             '"contentHash":"sha256:stale"}').encode()


def test_content_hash_golden_value():
    assert canonical.content_hash(DOC) == GOLDEN


def test_content_hash_ignores_key_order_and_hash_field():
    reordered = {"contentHash": "sha256:other", "a": {"y": 0.1, "z": "é"}, "b": [1, 2.5, None, True]}
    assert canonical.content_hash(reordered) == canonical.content_hash(DOC)
    assert canonical.content_hash(DOC) == canonical.digest({k: v for k, v in DOC.items() if k != "contentHash"})


def test_content_hash_distinguishes_int_and_float():
    assert canonical.digest({"x": 1}) != canonical.digest({"x": 1.0})


def test_seal_and_verify():
    sealed = canonical.seal(DOC)
    assert canonical.verify(sealed) and not canonical.verify(DOC)
    assert canonical.seal(sealed) == sealed
    sealed["a"]["y"] = 0.2
    assert not canonical.verify(sealed)


def test_nan_is_rejected():
    with pytest.raises(ValueError):
        canonical.digest({"x": math.nan})
