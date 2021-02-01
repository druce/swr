import numpy as np


def crra_certainty_equivalent(cashflows, gamma):
    """takes a numpy array, returns total CRRA certainty-equivalent cash flow"""
    if gamma > 1 and 0 in cashflows:
        return 0
    elif gamma == 1.0:
        # general formula for CRRA utility undefined for gamma = 1 but limit as gamma->1 = log
        u = np.mean(np.log(cashflows))
        ce = np.exp(u)
    elif gamma == 2.0:  # simple optimization
        u2 = np.mean(1 - 1.0 / cashflows)
        ce = 1.0 / (1.0 - u2)
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

survival
convert to number who died in that year
ce up to each row
ce * number who died in that year
