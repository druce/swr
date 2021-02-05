# test_crra.py

# pip install pytest
# pytest -q test_crra.py
import pytest
import numpy as np
from SWRsimulation.SWRsimulation import crra_ce, crra_ce_deathrate


def test_zero():
    testseries = np.random.uniform(1, 100, 100)
    testseries[50] = 0.0
    assert crra_ce(testseries, 0.0) == pytest.approx(np.sum(testseries), 0.0000001), \
        "if gamma = 0 should return np.sum()"
    
    for gamma in [1, 2, 4, 8, 16]:
        ce = crra_ce(testseries, gamma)
        assert ce == 0, "zero cash flow should always return 0 %d %f" % (gamma, ce)


def test_negative():
    testseries = np.random.uniform(1, 100, 100)
    testseries[50] = -1.0
    for gamma in [0, 0.5, 1, 2, 4, 8, 16]:    
        assert crra_ce(testseries, gamma) == 0, "negative cash flow should always return 0"


def test_constant_stream():
    # a constant series should always be sum of cash flows for any gamma
    testseries = np.ones(100)
    for gamma in [0, 0.5, 1, 2, 4, 8, 16]:
        print(gamma, np.sum(testseries), crra_ce(testseries, gamma), )
        assert crra_ce(testseries, gamma) == pytest.approx(np.sum(testseries), 0.0000001), \
            "constant cash flow should return sum of cash flows for any gamma"


def general_ce(cashflows, gamma):
    cashflows = np.longdouble(cashflows)
    calibration_factor = np.sum(cashflows) * 10
    cashflows /= calibration_factor

    if gamma == 1:
        u = np.mean(np.log(cashflows))
        ce = np.exp(u)
    else:
        u = np.mean((cashflows ** (1 - gamma) - 1) / (1 - gamma))
        ce = (1 + u * (1 - gamma)) ** (1 / (1 - gamma))
    ce = np.float(ce)
    return ce * len(cashflows) * calibration_factor


def test_equals_general():

    testseries = np.random.uniform(1, 100, 100)
    for gamma in [0, 0.5, 1, 2, 4, 8, 16]:
        print(gamma, crra_ce(testseries, gamma), general_ce(testseries, gamma))
        assert crra_ce(testseries, gamma) == pytest.approx(general_ce(testseries, gamma), 0.000001)


def test_deathrate():
    # if everyone dies in last year, mortality-adjusted CE = unadjusted CE
    testseries = np.random.uniform(1, 100, 100)
    deathrate = np.zeros(100)
    deathrate[99] = 1.0

    # can't go too high without some numerical problems
    for gamma in [0, 0.5, 1, 2, 4, 8, 16]:
        print(gamma, crra_ce(testseries, gamma), crra_ce_deathrate(testseries, gamma, deathrate))
        assert crra_ce(testseries, gamma) == pytest.approx(general_ce(testseries, gamma), 0.000001)
