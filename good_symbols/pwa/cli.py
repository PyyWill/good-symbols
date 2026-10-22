"""Command line: ``python -m good_symbols.pwa {cpr,run,collect,build-graph} ...``."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from .control import E2EController, rollout
from .cpr import cpr
from .demonstrations import Demonstration, load_demonstrations, save_demonstrations
from .guided import SymbolGuidedController
from .symbols import SymbolGraph, build_symbol_graph, file_digest
from .tasks import TASKS, make_task, task_class
from .tasks.base import DATA_ROOT
from .tasks.hand.task import HandTask


def seeds(text: str) -> list[int]:
    """``"0:16"``, ``"3,5,8"`` or a mix such as ``"0:4,10"``."""
    out = []
    for part in text.split(","):
        start, _, stop = part.partition(":")
        out += list(range(int(start), int(stop))) if stop else [int(start)]
    return out


def command_cpr(args) -> None:
    """Invent symbols for each task and compare them with the released certificate."""
    for name in args.tasks:
        interface = task_class(name).discrete_interface()
        certificate = cpr(interface.E, interface.P, interface.B)
        checks = certificate.check(interface.E, interface.P, interface.B)
        released = DATA_ROOT / name / "symbol_graph.json"
        same = np.array_equal(SymbolGraph.load(released).W, certificate.W) if released.exists() else None
        print(f"{name:7s} nz={certificate.W.shape[1]:3d}  k={certificate.k:3d}  "
              f"rank bound={certificate.rank_bound:3d}  "
              f"splits={list(certificate.splits)}  certified={all(checks.values())}  "
              f"matches released W={same}")


def _make(args):
    overrides = {"time_limit": args.time_limit} if args.time_limit is not None else {}
    if issubclass(task_class(args.task), HandTask):
        overrides["perturbation"] = args.perturbation
    return make_task(args.task, args.setting, **overrides)


def command_run(args) -> None:
    """Closed-loop evaluation of the E2E or the symbol-guided controller."""
    task = _make(args)
    controller = E2EController(task) if args.controller == "e2e" else SymbolGuidedController.from_data(task)
    output = Path(args.output or f"results/{args.task}_{args.controller}_setting{args.setting}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"task": args.task, "controller": args.controller, "setting": args.setting,
              "config": asdict(task.config), "episodes": []}
    task.launch(args.device)
    for seed in seeds(args.seeds):
        episode = rollout(task, controller, seed, max_steps=args.max_steps)
        report["episodes"].append(episode.summary())
        report["success_rate"] = float(np.mean([e["success"] for e in report["episodes"]]))
        output.write_text(json.dumps(report, indent=2) + "\n")         # before close(): IsaacLab exits on close
        print(json.dumps(episode.summary()), flush=True)
    print(f"{args.task} {args.controller}: {sum(e['success'] for e in report['episodes'])}/{len(report['episodes'])} "
          f"successful -> {output}", flush=True)
    task.close()


def command_collect(args) -> None:
    """Record E2E demonstrations (successful episodes only)."""
    task = _make(args)
    output = Path(args.output)
    demonstrations = load_demonstrations([output]) if args.append and output.exists() else []
    controller = E2EController(task)
    task.launch(args.device)
    for seed in seeds(args.seeds):
        episode = rollout(task, controller, seed, max_steps=args.max_steps)
        print(json.dumps(episode.summary()), flush=True)
        if episode.success and episode.actions:
            demonstrations.append(Demonstration(
                np.asarray(episode.states), np.asarray(episode.modes), np.asarray(episode.actions),
                np.asarray(episode.plans), np.asarray(episode.goal),
                {"setting": task.setting, "seed": seed, "success": True, "steps": episode.steps}))
        save_demonstrations(output, demonstrations)
    print(f"{len(demonstrations)} demonstrations -> {output}", flush=True)
    task.close()


def command_build_graph(args) -> None:
    """Invent symbols with CPR and build the symbol graph from demonstrations."""
    cls = task_class(args.task)
    interface = cls.discrete_interface()
    W = cpr(interface.E, interface.P, interface.B).W
    paths = [Path(p) for p in args.demonstrations or [DATA_ROOT / args.task / "demonstrations.npz"]]
    demonstrations = load_demonstrations(paths)
    tasks = {s: make_task(args.task, s) for s in sorted({d.setting for d in demonstrations})}
    graph, audit = build_symbol_graph(W, demonstrations, lambda d, t: tasks[d.setting].milp(d.states[t], d.goal),
                                      horizon=next(iter(tasks.values())).horizon)
    if len(paths) == 1:
        graph.metadata["demonstrations_sha256"] = file_digest(paths[0])
    output = Path(args.output or DATA_ROOT / args.task / "symbol_graph.json")
    graph.save(output)
    print(f"{args.task}: {len(graph.nodes)} symbols, {len(graph.edges)} transitions "
          f"({audit['plans']} plans validated, {len(audit['rejected_plans'])} rejected) -> {output}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="python -m good_symbols.pwa", description=(
        "Solver-sufficient symbols for PWA systems: symbol invention (CPR), symbol-guided control and the "
        "end-to-end MILP controller."))
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("cpr", help=command_cpr.__doc__)
    p.add_argument("tasks", nargs="*", default=list(TASKS), choices=list(TASKS), metavar="task")
    p.set_defaults(run=command_cpr)

    def episode_options(p):
        p.add_argument("--task", required=True, choices=list(TASKS))
        p.add_argument("--seeds", default="0:16", help='e.g. "0:16" or "0,3,7"')
        p.add_argument("--setting", type=int, default=1, help="fitted setting of the Shadow Hand tasks (1-3)")
        p.add_argument("--perturbation", type=float, default=1.0, help="scale of the Shadow Hand reset perturbation")
        p.add_argument("--time-limit", type=float, help="per-decision solver time limit (s)")
        p.add_argument("--max-steps", type=int)
        p.add_argument("--device", default="cuda:0")

    p = commands.add_parser("run", help=command_run.__doc__)
    episode_options(p)
    p.add_argument("--controller", choices=("symbol-guided", "e2e"), default="symbol-guided")
    p.add_argument("--output")
    p.set_defaults(run=command_run)

    p = commands.add_parser("collect", help=command_collect.__doc__)
    episode_options(p)
    p.add_argument("--output", required=True)
    p.add_argument("--append", action="store_true", help="add to an existing demonstration file")
    p.set_defaults(run=command_collect)

    p = commands.add_parser("build-graph", help=command_build_graph.__doc__)
    p.add_argument("--task", required=True, choices=list(TASKS))
    p.add_argument("--demonstrations", nargs="*")
    p.add_argument("--output")
    p.set_defaults(run=command_build_graph)

    args = parser.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()
