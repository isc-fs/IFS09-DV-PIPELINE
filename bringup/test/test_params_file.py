"""bringup/config/params.yaml must match the defaults the nodes declare.

Every launch loads params.yaml into the nodes that declare parameters, so the
YAML is what actually runs. The defaults in the code (declare_parameter) are
still needed: ROS requires every parameter to be declared, and a node started
without the file (unit tests, `ros2 run`) falls back to them. Two copies of
the same numbers drift, so this test pins them together:

  * every parameter a node declares is in params.yaml, and nothing else is;
  * each value equals the code default, with the same type. A float param
    written as `3` (an int) would make ROS refuse the file at launch
    (InvalidParameterTypeException), so types are checked, not just values.

To change a default, change it in BOTH places; the failure message says which
value is where. Two kinds of parameters must stay OUT of the YAML, or the file
would override the thing that sets them:

  * ENV_DRIVEN: defaults computed at runtime (read from the environment);
  * LAUNCH_SWITCHES: mission_control's free_run / hard_stop_on_finish, set by
    launch arguments. hard_stop_on_finish is safety-critical (docs/HARD_STOP.md):
    it must never be one careless edit of a tuning file away from true.

Source-level (reads the .py / .cpp files, no rclpy), so it runs in plain pytest.
"""
from __future__ import annotations

import ast
import os
import re

import yaml

_HERE = os.path.dirname(__file__)
REPO = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
PARAMS_FILE = os.path.join(REPO, "bringup", "config", "params.yaml")

# node -> source file declaring its parameters (the files with declare_parameter calls)
SOURCES = {
    "control_node": "control/control/control_node.py",
    "slam_node": "cone_slam/cone_slam/cone_graph_slam_node.py",
    "odometry_filter_node": "odometry_filter_node/src/odometry_filter_node.cpp",
    "mission_control_node": "mission_control/mission_control/mission_control_node.py",
    "pipeline_watchdog_node": "pipeline_watchdog/pipeline_watchdog/pipeline_watchdog_node.py",
}
# nodes that declare one parameter per field of a dataclass: node -> (file, class)
DATACLASS_SOURCES = {
    "cone_detection_node": ("cone_detection/cone_detection/config.py", "ConeDetectionConfig"),
}
# C++ headers holding the named constants some C++ defaults refer to
CPP_HEADERS = ["odometry_filter/include/odometry_filter/odometry_filter.hpp"]
# kept out of params.yaml on purpose (see the module docstring)
ENV_DRIVEN = {"slam_node": {"debug_gt_cones"}}
LAUNCH_SWITCHES = {"mission_control_node": {"free_run", "hard_stop_on_finish"}}


def _excluded(node: str) -> set:
    return ENV_DRIVEN.get(node, set()) | LAUNCH_SWITCHES.get(node, set())

_DECL = re.compile(
    r"declare_parameter(?:<\s*([\w:]+)\s*>)?\(\s*[\"']([\w.]+)[\"']\s*,\s*([^()]*?)\s*\)", re.S
)
_DECL_ANY = re.compile(r"declare_parameter(?:<[^>]*>)?\(\s*[\"']([\w.]+)[\"']")
_PY_CONST = re.compile(r"^([A-Z][A-Z0-9_]*)\s*(?::[^=\n]+)?=\s*([^\n#]+)", re.M)
_CPP_CONST = re.compile(r"\bconst(?:expr)?\s+[\w:<>]+\s+(k\w+|[A-Z][A-Z0-9_]+)\s*=\s*([^;]+);")


def _read(rel: str) -> str:
    with open(os.path.join(REPO, rel)) as fh:
        return fh.read()


def _value(expr: str, consts: dict, cpp_type: str | None, depth: int = 0):
    """A default as written in the code -> its value (literals and named constants)."""
    e = expr.strip()
    if re.fullmatch(r"[\w:]+", e) and not re.fullmatch(r"-?\d+", e):
        name = e.rsplit("::", 1)[-1]
        if name in ("true", "false"):
            return name == "true"
        if name in consts and depth < 3:
            return _value(consts[name], consts, cpp_type, depth + 1)
    e = re.sub(r"(?<=\d)[fFlL]\b", "", e)  # C++ 0.5f
    v = ast.literal_eval(e)  # anything else (a call, an expression) raises
    if cpp_type == "double" and isinstance(v, int) and not isinstance(v, bool):
        v = float(v)  # declare_parameter<double>("x", 1) is a double param
    return list(v) if isinstance(v, tuple) else v


_FIELD = re.compile(r"^    ([a-z_][a-z0-9_]*): *([\w\[\], |]+?) *= *(.+?)\s*(?:#.*)?$", re.M)


def _dataclass_defaults(rel: str, cls: str) -> tuple[dict, list[str]]:
    """``name: type = default`` fields of ``cls`` (the node declares a parameter per field, typed
    by the field: a ``float`` field is a double parameter even with an int default)."""
    text = _read(rel)
    body = text.split(f"class {cls}", 1)[1]
    body = body.split("\n\n\n", 1)[0]  # up to the next top-level definition
    out, unreadable = {}, []
    for name, typ, expr in _FIELD.findall(body):
        try:
            v = _value(expr, {}, None)
        except (ValueError, SyntaxError):
            unreadable.append(name)
            continue
        if typ.strip() == "float" and isinstance(v, int) and not isinstance(v, bool):
            v = float(v)
        out[name] = v
    return out, unreadable


def declared(node: str) -> tuple[dict, list[str]]:
    """``({"dotted.name": default}, [names whose default isn't a fixed value])`` for one node."""
    if node in DATACLASS_SOURCES:
        return _dataclass_defaults(*DATACLASS_SOURCES[node])
    rel = SOURCES[node]
    text = _read(rel)
    if rel.endswith(".py"):
        consts = dict(_PY_CONST.findall(text))
    else:
        consts = {}
        for h in CPP_HEADERS + [rel]:
            consts.update(_CPP_CONST.findall(_read(h)))
    out, unreadable = {}, []
    for cpp_type, name, expr in _DECL.findall(text):
        try:
            out[name] = _value(expr, consts, cpp_type or None)
        except (ValueError, SyntaxError):
            unreadable.append(name)
    unreadable += [n for n in dict.fromkeys(_DECL_ANY.findall(text)) if n not in out and n not in unreadable]
    return out, unreadable


def _flat(d: dict, pre: str = "") -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{pre}{k}."))
        else:
            out[f"{pre}{k}"] = v
    return out


def _load() -> dict:
    with open(PARAMS_FILE) as fh:
        return yaml.safe_load(fh)


def test_params_file_is_a_ros_params_file():
    data = _load()
    assert isinstance(data, dict) and data
    for node, body in data.items():
        assert list(body) == ["ros__parameters"], f"{node}: expected only a ros__parameters mapping"


def test_every_node_with_parameters_is_covered():
    for node, rel in {**SOURCES, **{n: f for n, (f, _) in DATACLASS_SOURCES.items()}}.items():
        assert os.path.isfile(os.path.join(REPO, rel)), f"{node}: {rel} moved? update SOURCES"
    nodes = [*SOURCES, *DATACLASS_SOURCES]
    tunable = {n for n in nodes if set(declared(n)[0]) - _excluded(n)}
    assert set(_load()) == tunable, "params.yaml nodes differ from the nodes with tunable parameters"


def test_params_file_matches_the_declared_defaults():
    data = _load()
    problems = []
    for node in [*SOURCES, *DATACLASS_SOURCES]:
        code, unreadable = declared(node)
        skip = _excluded(node)
        unexpected = set(unreadable) - skip
        assert not unexpected, (
            f"{node}: can't read the default of {sorted(unexpected)} from the code. Use a literal or a "
            f"named constant, or add it to ENV_DRIVEN if it is computed at runtime."
        )
        yml = _flat(data.get(node, {}).get("ros__parameters", {}))
        for name in sorted(set(code) | set(yml)):
            if name in skip:
                if name in yml:
                    problems.append(f"{node}.{name}: set at runtime or by a launch argument; keep it out of params.yaml")
            elif name not in yml:
                problems.append(f"{node}.{name}: declared in the code (default {code[name]!r}) but missing in params.yaml")
            elif name not in code:
                problems.append(f"{node}.{name}: in params.yaml but not declared by the node (typo, or removed?)")
            elif yml[name] != code[name] or type(yml[name]) is not type(code[name]):
                problems.append(
                    f"{node}.{name}: code default {code[name]!r} ({type(code[name]).__name__}) but params.yaml "
                    f"has {yml[name]!r} ({type(yml[name]).__name__}); change both"
                )
    assert not problems, "params.yaml and the code disagree:\n  " + "\n  ".join(problems)


def test_launch_loads_the_params_file():
    """The file is installed with the package and handed to the nodes by launch_common."""
    setup = _read("bringup/setup.py")
    assert '"config"' in setup and "*.yaml" in setup, "bringup/setup.py must install config/*.yaml"
    common = _read("bringup/bringup/launch_common.py")
    # the three node builders (autonomy lifecycle nodes, the auto-active trio, the watchdog)
    assert common.count("parameters=node_parameters(") == 3, "every node builder must pass node_parameters()"
