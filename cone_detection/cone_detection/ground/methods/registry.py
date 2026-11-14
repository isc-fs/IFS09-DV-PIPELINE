"""The ground-method registry and the ``GroundRemover`` contract.

A method is registered with ``@register("name")`` on either

* a plain function ``fn(data, cfg) -> (non_ground_mask, plane)`` for a method that
  keeps nothing between scans (wrapped in :class:`StatelessGround`), or
* a class ``cls(cfg)`` implementing :class:`GroundRemover`, for a method that
  carries state from one scan to the next (a warm-start plane, a learning
  estimator). The state lives on the instance, never in module globals, so two
  detectors never share it and ``reset()`` clears it.

:func:`make_ground_remover` builds the remover the detector owns.
"""

from __future__ import annotations

import importlib
from typing import Callable, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class GroundRemover(Protocol):
    """One ground-removal method bound to a config, owning any per-scan state."""

    def remove(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(non_ground_mask, plane)`` for one sensor-frame scan; see ``ground.methods``."""
        ...

    def reset(self) -> None:
        """Forget everything learned from previous scans (bag seek, new run)."""
        ...


class StatelessGround:
    """Adapts a plain ``fn(data, cfg)`` method to :class:`GroundRemover`."""

    __slots__ = ("_fn", "_cfg")

    def __init__(self, fn: Callable, cfg) -> None:
        self._fn = fn
        self._cfg = cfg

    def remove(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self._fn(data, self._cfg)

    def reset(self) -> None:
        pass


# name -> plain function or GroundRemover class
METHODS: dict[str, Callable] = {}


def register(name: str):
    def deco(obj):
        METHODS[name] = obj
        return obj

    return deco


def build_remover(method: str, cfg) -> GroundRemover:
    try:
        obj = METHODS[method]
    except KeyError:
        raise ValueError(
            f"unknown ground_method {method!r}; choose from {sorted(METHODS)}"
        ) from None
    if isinstance(obj, type):
        return obj(cfg)
    return StatelessGround(obj, cfg)


def _optional(module: str, pip_name: str):
    try:
        return importlib.import_module(module)
    except ImportError as e:
        raise ImportError(
            f"this ground_method needs the optional package {pip_name!r} (pip install {pip_name})"
        ) from e
