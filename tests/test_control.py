import numpy as np
import pytest

from good_symbols.pwa import E2EController, SymbolGuidedController, make_task
from good_symbols.pwa.demonstrations import load_demonstrations
from good_symbols.pwa.tasks import TASKS


@pytest.mark.parametrize("name", TASKS)
def test_milp_puts_modes_first(name):
    task = make_task(name)
    demo = load_demonstrations([task.data_dir / "demonstrations.npz"])[0]
    problem = task.milp(demo.states[0], demo.goal)
    n = task.horizon * task.nz
    assert problem.integrality[:n].all()
    assert np.all(problem.bounds.lb[:n] == 0) and np.all(problem.bounds.ub[:n] == 1)
    assert problem.constraints.A.shape[1] == len(problem.cost)


@pytest.mark.parametrize("name", ["mpusht", "hook"])
def test_controllers_agree_on_a_feasible_first_step(name):
    task = make_task(name)
    state, goal = task.reset(0)
    e2e = E2EController(task).act(state, goal)
    guided = SymbolGuidedController.from_data(task)
    guided.reset(state)
    decision = guided.act(state, goal)
    assert decision.action.shape == e2e.action.shape
    # A conditioned problem restricts the native one; the E2E optimum is exact up to its relative MIP gap.
    assert decision.objective >= e2e.objective - 1e-3 * max(1.0, abs(e2e.objective))
