"""deploy/dv_warm_numba.py must bake the config cone_detection_node will run (SW-08).

A bake with any other config compiles other signatures, and bring-up then JITs
inside mode_manager's 120 s budget (measured 128 s cold on the RK3588).
"""
import importlib.util
import os

import yaml

from cone_detection.config import ConeDetectionConfig

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
PARAMS = os.path.join(REPO, "bringup", "config", "params.yaml")


def _bake_module():
    spec = importlib.util.spec_from_file_location(
        "dv_warm_numba", os.path.join(REPO, "deploy", "dv_warm_numba.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_bake_uses_params_yaml_and_both_detector_backends(tmp_path):
    with open(PARAMS) as fh:
        data = yaml.safe_load(fh)
    data["cone_detection_node"]["ros__parameters"]["cluster_backend"] = "grid"
    data["cone_detection_node"]["ros__parameters"]["dbscan_eps"] = 0.25
    edited = tmp_path / "params.yaml"
    edited.write_text(yaml.safe_dump(data))

    cfgs = _bake_module().cone_detection_configs(str(edited))
    assert sorted(c.detector_backend for c in cfgs) == ["fused", "python"]
    for c in cfgs:
        assert c.cluster_backend == "grid"
        assert c.dbscan_eps == 0.25


def test_bake_also_warms_the_grid_clusterer():
    cfgs = _bake_module().cone_detection_configs(PARAMS)
    pairs = {(c.detector_backend, c.cluster_backend) for c in cfgs}
    shipped = ConeDetectionConfig().cluster_backend
    for b in ("python", "fused"):
        assert (b, "grid") in pairs
        assert (b, shipped) in pairs


def test_bake_matches_defaults_with_the_shipped_params():
    cfgs = _bake_module().cone_detection_configs(PARAMS)
    py = next(
        c for c in cfgs
        if c.detector_backend == "python" and c.cluster_backend == ConeDetectionConfig().cluster_backend
    )
    # params.yaml mirrors the code defaults (bringup/test/test_params_file.py)
    assert py == ConeDetectionConfig()
