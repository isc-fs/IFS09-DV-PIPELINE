"""Both detector backends must reproduce the golden outputs on real car scans.

See golden/golden.py for what the fixtures are and how to regenerate them. The
test skips when scans.npz is not pulled from Git LFS.
"""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden"))
import golden  # noqa: E402

pytestmark = pytest.mark.skipif(
    not golden.scans_available(), reason="golden scans.npz not pulled (git lfs pull)"
)


@pytest.fixture(scope="module")
def reference():
    with open(golden.REFERENCE) as fh:
        return json.load(fh)["scans"]


@pytest.fixture(scope="module")
def scans():
    return golden.load_scans()


@pytest.mark.parametrize("backend", ["python", "fused"])
def test_matches_golden(backend, scans, reference):
    out = golden.run(golden.production_config(backend), scans)
    assert len(out) == len(reference)
    for k, (got, ref) in enumerate(zip(out, reference)):
        assert got["counters"] == ref["counters"], f"scan {k}"
        np.testing.assert_allclose(got["plane"], ref["plane"], rtol=0, atol=1e-9, err_msg=f"scan {k}")
        assert len(got["cones"]) == len(ref["cones"]), f"scan {k}"
        np.testing.assert_allclose(
            np.asarray(got["cones"]).reshape(-1, 4),
            np.asarray(ref["cones"]).reshape(-1, 4),
            rtol=0,
            atol=1e-6,
            err_msg=f"scan {k}",
        )
