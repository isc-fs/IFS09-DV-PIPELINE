"""control_node comes from control_cpp unless DV_CONTROL=python."""
import pytest

from bringup.launch_common import control_package


def test_default_is_cpp():
    assert control_package({}) == "control_cpp"
    assert control_package({"DV_CONTROL": "cpp"}) == "control_cpp"


def test_python_fallback():
    assert control_package({"DV_CONTROL": "python"}) == "control"


def test_typo_stops_the_launch():
    with pytest.raises(ValueError):
        control_package({"DV_CONTROL": "pyhton"})
