"""The six PWA tasks of the paper: three planar manipulation tasks and three in-hand reorientation tasks."""
from __future__ import annotations

from importlib import import_module

from .base import DiscreteInterface, Task

TASKS = {
    "mpusht": "mpusht.task:MPushT",
    "hook": "hook.task:Hook",
    "shelf": "shelf.task:Shelf",
    "block": "block.task:Block",
    "cuboid": "cuboid.task:Cuboid",
    "pen": "pen.task:Pen",
}


def task_class(name: str) -> type[Task]:
    if name not in TASKS:
        raise ValueError(f"unknown task {name!r}; choose from {', '.join(TASKS)}")
    module, cls = TASKS[name].split(":")
    return getattr(import_module(f"{__name__}.{module}"), cls)


def make_task(name: str, setting: int = 1, **overrides) -> Task:
    """Instantiate a task; ``overrides`` replace fields of its ``Config`` (e.g. ``time_limit=5.0``)."""
    return task_class(name)(setting, **overrides)


__all__ = ["TASKS", "DiscreteInterface", "Task", "make_task", "task_class"]
