# test_crra.py

# pip install pytest
# pytest -q test_crra.py
import pytest
import numpy as np
import pandas as pd
from SWRsimulation.SWRsimulation import crra_ce, crra_ce_deathrate

def test_zero():
    testseries = np.random.uniform(1, 100, 100)
    testseries[50] = 0.0

    assert crra_ce(testseries, 0.0) == np.sum(testseries), "if gamma = 0 should return np.sum()"
    
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
        assert crra_ce(testseries, gamma) == np.sum(testseries), "constant cash flow should return sum of cash flows for any gamma"

def test_equals_general():

    def general_ce(cashflows, gamma):
        """general CE formula"""
        cashflows = np.longdouble(cashflows)
        if gamma == 1:
            u = np.mean(np.log(cashflows))
            ce = np.exp(u)
        else:
            u = np.mean((cashflows ** (1-gamma) -1) / (1-gamma))
            ce = (1 + u * (1-gamma)) ** (1/(1-gamma))
            ce = np.float(ce)

        return ce * len(cashflows)
    
    testseries = np.random.uniform(1, 100, 100)
    # can't go to 16 without some numerical problems
    for gamma in [0, 0.5, 1, 2, 4, 8, 15]:
        print(gamma, crra_ce(testseries, gamma), general_ce(testseries, gamma))
        assert crra_ce(testseries, gamma) == pytest.approx(general_ce(testseries, gamma), 0.000001)

