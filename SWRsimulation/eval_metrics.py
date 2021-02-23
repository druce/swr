import numpy as np

from .crra_ce import crra_ce, crra_utility


def eval_exhaustion(simulation):
    """exhaustion and min portfolio metrics for current trial

    Returns:
        float: year exhausted for current trial or n_ret_years if never exhausted
        float: minimum ending portfolio value (0 if exhaustion < n_ret_years)
        int: indexes of minimum ending portfolio value
    """
    min_end_port_index = int(np.argmin(simulation.latest_trial.end_ports))
    min_end_port_value = simulation.latest_trial.end_ports[min_end_port_index]
    exhaustion = min_end_port_index if min_end_port_value == 0.0 else simulation.simulation['n_ret_years']
    return exhaustion, min_end_port_value, min_end_port_index


def eval_ce(simulation):
    """certainty-equivalent metric for current trial

    Returns:
        float: CE cash flow for spending in current trial
    """
    return crra_ce(simulation.latest_trial.trial_df['spend'], simulation.evaluation['gamma'])


def eval_crra_utility(simulation):
    """CRRA utility metric for current trial

    Returns:
        float: total CRRA utility for spending in current trial
    """
    return crra_utility(simulation.latest_trial.trial_df['spend'], simulation.evaluation['gamma'])


def eval_median_spend(simulation):
    """median spend metric for current trial

    Returns:
        float: Median real spending for current trial
    """
    return simulation.latest_trial.trial_df['spend'].median()


def eval_mean_spend(simulation):
    """median spend metric for current trial

    Returns:
        float: Mean real spending for current trial
    """
    return simulation.latest_trial.trial_df['spend'].mean()


def eval_min_spend(simulation):
    """minimum spend metric for current trial

    Returns:
        float: Minimum real spending for current trial
    """
    return simulation.latest_trial.trial_df['spend'].min()


def eval_max_spend(simulation):
    """maximum spend metric for current trial

    Returns:
        float: Maximum real spending for current trial
    """
    return simulation.latest_trial.trial_df['spend'].max()


def eval_sd_spend(simulation):
    """standard deviation of spend metric for current trial

    Returns:
        float: standard deviation of real spending for current trial
    """
    return simulation.latest_trial.trial_df['spend'].std()
