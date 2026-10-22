"""Solver-sufficient symbols for piecewise-affine (PWA) systems, Section 2 of the paper.

* :func:`cpr` invents a symbol basis ``W`` (a relative affine-TU certificate) for a task's discrete interface.
* :class:`E2EController` solves the native MILP at every step; :class:`SymbolGuidedController` searches symbol
  plans and solves conditioned LPs instead.
* :func:`make_task` builds one of the six tasks (``mpusht``, ``hook``, ``shelf``, ``block``, ``cuboid``, ``pen``).
"""
from .control import E2EController, rollout
from .cpr import Certificate, cpr
from .guided import SymbolGuidedController, SymbolPlanLibrary
from .symbols import SymbolGraph, symbol
from .tasks import TASKS, make_task
from .tu import is_totally_unimodular

__all__ = ["Certificate", "E2EController", "SymbolGraph", "SymbolGuidedController", "SymbolPlanLibrary", "TASKS",
           "cpr", "is_totally_unimodular", "make_task", "rollout", "symbol"]
