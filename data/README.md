# Data

One directory per task (`mpusht`, `hook`, `shelf`, `block`, `cuboid`, `pen`).

| File | Content |
| --- | --- |
| `symbol_graph.json` | The symbol basis `W` invented by CPR, the symbols `psi = W z` and the symbol transitions seen in demonstrations: every executed transition of a successful demonstration, and every transition of a demonstrated H-step plan whose conditioned LP is feasible. Built from `demonstrations.npz` with `python -m good_symbols.pwa build-graph`. |
| `demonstrations.npz` | Episodes of the end-to-end MILP controller: states, observed modes, actions, the H-step mode plan of every step and the goal, with JSON metadata (`setting`, `seed`, `success`, ...). |
| `demonstrations_extra.npz` | (Shelf only) additional dense demonstrations. They extend the symbol graph with their executed transitions and the plan library of the symbol-guided controller, but were not used to build `symbol_graph.json`. |
| `setting_{1,2,3}.npz` | (Shadow Hand tasks) the fitted local model of one grasp setting: actuated joints, joint limits, servo gains, contact-arc rotation gains, fingertip and contact-gap models, the reference trajectory used for resets, the goal and the feasible contact flows. |

The symbol-guided controller reads `symbol_graph.json` and all `demonstrations*.npz` files of a task; no
demonstrated action is executed.  Set `GOOD_SYMBOLS_DATA` to use a different data directory.

The data in this directory and the figures in `assets/` are licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
