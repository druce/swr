import numpy as np
import pytest


def crra_ce(cashflows, gamma):
    """takes a numpy array, returns total CRRA certainty-equivalent cash flow"""
    # for retirement study assume no negative cashflows
    if sum(np.where(cashflows < 0, 1, 0)):
        return 0.0
    elif gamma >= 1.0 and 0 in cashflows:
        return 0.0
    elif gamma == 1.0:
        # general formula for CRRA utility undefined for test_gamma = 1 but limit as test_gamma->1 = log
        u = np.mean(np.log(cashflows))
        ce = np.exp(u)
    elif gamma == 2.0:  # simple optimization
        u2 = np.mean(1 - 1.0 / cashflows)
        ce = 1.0 / (1.0 - u2)
    elif gamma > 4.0:
        # force computations as longdouble for more precision, but always return np.float
        gamma = np.longdouble(gamma)
        cashflows = cashflows.astype(np.longdouble)
        gamma_m1 = gamma - 1.0
        gamma_m1_inverse = 1.0 / gamma_m1
        u = np.mean(gamma_m1_inverse - 1.0 / (gamma_m1 * cashflows ** gamma_m1))
        ce = 1.0 / (1.0 - gamma_m1 * u) ** gamma_m1_inverse
        ce = np.float(ce)
    elif gamma > 1.0:
        gamma_m1 = gamma - 1.0
        gamma_m1_inverse = 1.0 / gamma_m1
        u = np.mean(gamma_m1_inverse - 1.0 / (gamma_m1 * cashflows ** gamma_m1))
        ce = 1.0 / (1.0 - gamma_m1 * u) ** gamma_m1_inverse
    else:  # general formula
        g_1m = 1 - gamma
        u = np.mean((cashflows ** g_1m - 1.0) / g_1m)
        ce = (g_1m * u + 1.0) ** (1.0 / g_1m)

    return ce * len(cashflows)


def crra_ce_deathrate(cashflows, gamma, deathrate):
    """ce cash flow with a mortality curve
    cash flows = real cash flows in each year of cohort
    test_gamma = risk aversion
    death rate = % of cohort that died in each year of cohort

    compute utility of each cash flow under test_gamma
    compute cumulative mean of utilities up to each year
    convert utilities back to CE cash flows
    each member of cohort that died in a given year experienced CE cash flow * years alive
    """
    # for retirement study assume no negative cashflows
    if sum(np.where(cashflows < 0, 1, 0)):
        return 0.0
    else:
        # 1..lastyear
        indices = np.indices(cashflows.shape)[0] + 1

        if gamma == 1.0:
            # utility
            u = np.log(cashflows)
            # cumulative mean utility
            u_mean = np.cumsum(u) / indices
            # cumulative mean ce cash flows
            ce = np.exp(u_mean) * indices
        elif gamma == 2.0:  # simple optimization
            u2 = 1 - 1.0 / cashflows
            u2_mean = np.cumsum(u2) / indices
            ce = 1.0 / (1.0 - u2_mean) * indices
        elif gamma > 4.0:
            # force computations as longdouble for more precision, but always return np.float
            gamma = np.longdouble(gamma)
            cashflows = cashflows.astype(np.longdouble)
            gamma_m1 = gamma - 1.0
            gamma_m1_inverse = 1.0 / gamma_m1
            u = gamma_m1_inverse - 1.0 / (gamma_m1 * cashflows ** gamma_m1)
            u_mean = np.cumsum(u) / indices
            ce = 1.0 / (1.0 - gamma_m1 * u_mean) ** gamma_m1_inverse
            ce = (ce * indices).astype(float)
        elif gamma > 1.0:
            gamma_m1 = gamma - 1.0
            gamma_m1_inverse = 1.0 / gamma_m1
            u = gamma_m1_inverse - 1.0 / (gamma_m1 * cashflows ** gamma_m1)
            u_mean = np.cumsum(u) / indices
            ce = 1.0 / (1.0 - gamma_m1 * u_mean) ** gamma_m1_inverse
            ce = ce * indices
        else:  # general formula
            gamma_1m = 1 - gamma
            u = (cashflows ** gamma_1m - 1.0) / gamma_1m
            u_mean = np.cumsum(u) / indices
            ce = (gamma_1m * u_mean + 1.0) ** (1.0 / gamma_1m)
            ce = ce * indices
        # mortality_adjusted ce cash flows
        madj_ce = np.sum(ce * deathrate)
        return madj_ce


def general_ce(cashflows, gamma):
    cashflows = np.longdouble(cashflows)
    if gamma == 1:
        u = np.mean(np.log(cashflows))
        ce = np.exp(u)
    else:
        u = np.mean((cashflows ** (1 - gamma) - 1) / (1 - gamma))
        ce = (1 + u * (1 - gamma)) ** (1 / (1 - gamma))
    ce = np.float(ce)
    return ce * len(cashflows)


if __name__ == '__main__':
    print('Executing as standalone script, running tests')
    # test 0 and -1
    testseries = np.random.uniform(1, 100, 100)
    testseries[50] = 0.0
    assert crra_ce(testseries, 1) == 0, "bad value"
    testseries[50] = -1.0
    assert crra_ce(testseries, 1) == 0, "bad value"

    # a constant series should always be sum of cash flows for any test_gamma
    testseries = np.ones(100)
    for test_gamma in [0, 0.5, 1, 2, 4, 8, 16]:
        print(test_gamma, crra_ce(testseries, test_gamma))
        assert crra_ce(testseries, test_gamma) == np.sum(testseries)

    testseries = np.random.uniform(1, 100, 100)

    # can't go to 16 without some numerical problems
    for test_gamma in [0, 0.5, 1, 2, 4, 8, 16]:
        print(test_gamma, crra_ce(testseries, test_gamma), general_ce(testseries, test_gamma))
        assert crra_ce(testseries, test_gamma) == pytest.approx(general_ce(testseries, test_gamma), 0.000001)

    # if everyone dies in last year, mortality-adjusted CE = unadjusted CE
    test_deathrate = np.zeros(100)
    test_deathrate[99] = 1.0

    # can't go to 16 without some numerical problems
    for test_gamma in [0, 0.5, 1, 2, 4, 8, 10]:
        print(test_gamma, crra_ce(testseries, test_gamma), crra_ce_deathrate(testseries, test_gamma, test_deathrate))
        assert crra_ce(testseries, test_gamma) == pytest.approx(general_ce(testseries, test_gamma), 0.000001)
