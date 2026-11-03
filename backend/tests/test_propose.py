import numpy as np

from mixpilot.propose import pareto


def test_pareto_keeps_only_non_dominated_points():
    gwp = np.array([100, 120, 110, 200, 150])
    psi = np.array([4000, 5000, 3900, 5500, 4800])
    assert list(pareto(gwp, psi)) == [0, 1, 3]
