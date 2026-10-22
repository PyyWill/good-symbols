import numpy as np
import pytest

from good_symbols.pwa import SymbolGraph, cpr, is_totally_unimodular, make_task
from good_symbols.pwa.tasks import TASKS, task_class
from good_symbols.pwa.tu import TUOracleError


def test_tu_oracle_on_signed_graphs():
    incidence = np.array([[1, 1, 0], [-1, 0, 1], [0, -1, -1]])     # directed graph incidence matrix
    assert is_totally_unimodular(incidence)
    odd_cycle = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]])         # determinant 2
    assert not is_totally_unimodular(odd_cycle)
    assert not is_totally_unimodular([[2]])


@pytest.mark.parametrize("name", TASKS)
def test_cpr_reproduces_released_symbols(name):
    interface = task_class(name).discrete_interface()
    try:
        certificate = cpr(interface.E, interface.P, interface.B)
    except TUOracleError:
        pytest.skip("cmr-tu is not installed")
    assert all(certificate.check(interface.E, interface.P, interface.B).values())
    released = SymbolGraph.load(make_task(name).data_dir / "symbol_graph.json")
    assert np.array_equal(certificate.W, released.W)
