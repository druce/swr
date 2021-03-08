import numpy as np
import pdb

def crra_ce(cashflows, gamma):
    """takes a numpy array, returns total CRRA certainty-equivalent cash flow
    General formula for CRRA utility of a single cash flow w (wealth): 
        u = (w ** (1 - gamma) - 1) / (1-gamma)
    if gamma = 0, u = w
    if gamma = 1, u is undefined but limit as gamma->1 = log(w)
    for certainty equivalent of a stream, compute the total utility using specfied gamma,
    then convert that utility back to a the equivalent constant cash flow stream using that gamma
    using inverse function
    gamma <> 1: ce = (1 + u * (1 - gamma)) ** (1 / (1 - gamma))
    gamma ==1:  ce = exp(u)

    Args:
        cashflows (numpy.ndarray): array of cashflows
        gamma (float): risk aversion parameter

    Returns:
        float: certainty equivalent cashflow
    """
    
    # for retirement study assume no negative cashflows
    if np.any(np.where(cashflows < 0, 1, 0)):
        return 0.0
    elif gamma >= 1.0 and 0 in cashflows:
        return 0.0
    elif gamma == 0.0: # simple optimization
        ce = np.mean(cashflows)        
    elif gamma == 1.0:
        # general formula for CRRA utility undefined for test_gamma = 1 but limit as test_gamma->1 = log
        if np.any(np.where(cashflows == 0, 1, 0)):
            return 0.0
        u = np.mean(np.log(cashflows))
        ce = np.exp(u)
    elif gamma == 2.0:  # simple optimization
        if np.any(np.where(cashflows == 0, 1, 0)):
            return 0.0
        u2 = np.mean(1 - 1.0 / cashflows)
        ce = 1.0 / (1.0 - u2)
    elif gamma > 4.0:
        # force computations as longdouble for more precision, but always return np.float
        if np.any(np.where(cashflows == 0, 1, 0)):
            return 0.0
        gamma = np.longdouble(gamma)
        cashflows = cashflows.astype(np.longdouble)
        # first normalize by dividing by mean
        # we are taking high powers of cash flows so closer to 1 reduces numerical problems
        # if cash flows vary by factor of 1000 and we take a power of 32 we can run into overflow issues
        # for gamma > 1, there is an upper bound to utility 1/(gamma-1)
        # when utility approaches this limit, small improvements in cash flow don't numerically change utility
        # when you multiply cash flow by a factor, CE cash flow is multiplied by same factor
        # multiply by mean before returning to return same units as input
        calibration_factor = np.mean(cashflows)
        cashflows /= calibration_factor
        gamma_m1 = gamma - 1.0
        gamma_m1_inverse = 1.0 / gamma_m1
        u = np.mean(gamma_m1_inverse - 1.0 / (gamma_m1 * cashflows ** gamma_m1))
        ce = 1.0 / (1.0 - gamma_m1 * u) ** gamma_m1_inverse
        ce *= calibration_factor
        ce = np.float(ce)
    elif gamma > 1.0:
        if np.any(np.where(cashflows == 0, 1, 0)):
            return 0.0
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
    compute certainty-equivalent cash flow up to each year
    each member of cohort that died in a given year experienced CE cash flow * years alive
    return CE values weighted average using death rate as weights
    multiply CE values times death rate for each year and sum 
    # cash flows = 
    # test_gamma = risk aversion
    # death rate = % of cohort that died in each year of cohort


    Args:
        cashflows (numpy.ndarray): real cash flows in each year of cohort
        gamma (float): risk aversion parameter
        deathrate (numpy.ndarray): % who didn't survive in each year (must sum to 1)

    Returns:
        float: average CE experienced by cohort members based on the deathrate
    """

    # 
    # for retirement study assume no negative cashflows
    if np.any(np.where(cashflows < 0, 1, 0)):
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
            # since we are taking large powers, for numerical stability make mean = 1
            # convert back to input units at the end
            calibration_factor = np.mean(cashflows)
            cashflows /= calibration_factor
            gamma_m1 = gamma - 1.0
            gamma_m1_inverse = 1.0 / gamma_m1
            u = gamma_m1_inverse - 1.0 / (gamma_m1 * cashflows ** gamma_m1)
            u_mean = np.cumsum(u) / indices
            ce = 1.0 / (1.0 - gamma_m1 * u_mean) ** gamma_m1_inverse
            ce *= calibration_factor            
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


def crra_utility(cashflows, gamma):
    """takes a numpy array, returns total CRRA utility
    General formula for CRRA utility of a single cash flow w (wealth): 
        u = (w ** (1 - gamma) - 1) / (1-gamma)
    if gamma = 0, u = w
    if gamma = 1, log(2). general formula is undefined for gamma=1 but limit as gamma->1 = log(w)

    Args:
        cashflows (numpy.ndarray): array of cashflows
        gamma (float): risk aversion parameter

    Returns:
        float: utility (ranges from -infinity to plus infinity, when gamma is > 1, max is bounded)
    """

    # for retirement study assume no negative cashflows
    if np.any(np.where(cashflows < 0, 1, 0)):
        return -np.inf
    elif gamma >= 1.0 and 0.0 in cashflows:
        return -np.inf
    elif gamma == 0.0:
        return np.sum(cashflows)
    elif gamma == 1.0:
        # general formula for CRRA utility undefined for test_gamma = 1 but limit as test_gamma->1 = log
        if np.any(np.where(cashflows <= 0, 1, 0)):
            return -np.inf
        return np.sum(np.log(cashflows))
    elif gamma == 2.0:  # simple optimization
        if 0.0 in cashflows:
            return -np.inf
        return np.sum(1 - 1.0 / cashflows)
    elif gamma > 4.0:
        # force computations as longdouble for more precision, but always return np.float
        if 0.0 in cashflows:
            return -np.inf
        gamma = np.longdouble(gamma)
        cashflows = cashflows.astype(np.longdouble)
        gamma_m1 = gamma - 1.0
        gamma_m1_inverse = 1.0 / gamma_m1
        u = np.sum(gamma_m1_inverse - 1.0 / (gamma_m1 * cashflows ** gamma_m1))
        return np.float(u)
    elif gamma > 1.0:
        if 0.0 in cashflows:
            return -np.inf
        gamma_m1 = gamma - 1.0
        gamma_m1_inverse = 1.0 / gamma_m1
        return np.sum(gamma_m1_inverse - 1.0 / (gamma_m1 * cashflows ** gamma_m1))
    else:  # general formula
        g_1m = 1 - gamma
        return np.sum((cashflows ** g_1m - 1.0) / g_1m)


if __name__ == '__main__':
    print('Executing as standalone script')
