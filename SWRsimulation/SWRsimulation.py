from dataclasses import dataclass, field
import pprint
from typing import List

import numpy as np
import pandas as pd


# from plotly import graph_objects as go
# from plotly.subplots import make_subplots
# import plotly.express as px

# TODO: by default don't keep all trials, only if specified explicitly

START_PORTVAL = 100.0


@dataclass
class Trialdata:
    """
    Class for keeping track of a latest_trial.
    """
    year: int = 0
    spend: float = 0
    iteration: int = 0
    portval: float = START_PORTVAL

    years: List[int] = field(default_factory=list)
    start_ports: List[float] = field(default_factory=list)
    asset_allocations: List[np.array] = field(default_factory=list)
    port_returns: List[float] = field(default_factory=list)
    before_spends: List[float] = field(default_factory=list)
    end_ports: List[float] = field(default_factory=list)
    spends: List[float] = field(default_factory=list)
    trial_df: pd.DataFrame = None


class SWRsimulation:
    """abstract base class template for safe withdrawal simulations
    """
    def __init__(self, config):
        """initialize class from a config dict

        Args:
            config (dict): dict containing at least required keys:
                {'simulation': {}
                'allocation': {}
                'withdrawal': {}
                'visualization': {}
            }
        """
        # promote everything in config to instance variables for more readable code
        self.simulation = config.get('simulation')
        self.init_simulation()
        self.allocation = config.get('allocation')
        self.init_allocation()
        self.withdrawal = config.get('withdrawal')
        self.init_withdrawal()
        self.evaluation = config.get('evaluation')
        self.visualization = config.get('visualization')

        self.latest_trial = Trialdata()
        self.latest_simulation = []  # list of all trial data in latest simulation

    def init_simulation(self):
        """initialize trial generator eg. historical, monte carlo. prep for next()
        """
        pass

    def simulate_trial(self, trial_rows):
        """simulate a single historical cohort or montecarlo generated cohort

        Args:
            trial_rows (list or other iterator): asset returns for this cohort trial
        """
        pass

    def simulate(self, do_eval=True, return_both=True):
        """simulate all available cohorts

        Args:
            do_eval (bool, optional): whether to run eval on each cohort. (Default True)
            return_both (bool, optional): whether to save both eval and full cohort
            dataframe in self.latest_simulation. (Default True)
        """
        pass

    def init_allocation(self):
        """set up equal weight, allocation parameters etc. based on simulation config
        """
        pass

    def get_allocations(self):
        """return array of allocations for current simulation iteration based on config, current state
        """
        pass

    def init_withdrawal(self):
        """initialize withdrawal parameters based on simulation config
        """
        pass

    def get_withdrawal(self):
        """return withdrawal amount for current simulation iteration based on config, current state
        """
        pass

    def eval_trial(self):
        """return dict of metrics for a current trial
        single historical cohort or montecarlo generated cohort
        eg years to exhaustion, CE adjusted spending
        """
        pass

    def visualize(self):
        """run the analytics and data viz for a completed simulation
        """
        pass

    def __repr__(self):
        """Generate string representation of simulation

        Returns:
            [string]: string representation
        """
        retstr = "Simulation:\n"
        retstr += pprint.pformat(self.simulation)
        retstr += "\n\nAllocation:\n"
        retstr += pprint.pformat(self.allocation)
        retstr += "\n\nWithdrawal:\n"
        retstr += pprint.pformat(self.withdrawal)
        return retstr


if __name__ == '__main__':
    print('Executing as standalone script')
