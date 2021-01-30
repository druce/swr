import numpy as np
import pandas as pd
import pprint
from dataclasses import dataclass, field
from typing import List

START_PORTVAL = 100.0
int_types = (int, np.int8, np.int16, np.int32, np.int64)
float_types = (float, np.float16, np.float32, np.float64)
numeric_types = (int, np.int8, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64)


@dataclass
class Trialdata:
    """Class for keeping track of a trial."""
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


class SWRsimulator:
    """simulate retirement outcomes.

        Attributes:
            config: dict with options for simulator, allocation, spending, evaluation
            simulator['n_ret_years']: number of years of retirement to simulate
            simulator['n_hist_years']: number of years of historical returns available
            simulator['n_assets']: number of assets we have returns for
            simulator['trials']: iterator that yields trials (each an iterator of n_ret_years of asset returns )
        """

    def __init__(self, config):
        """pass a dict of config"""
        # promote everything in config to instance variables for more readable code
        self.simulator = config['simulator']
        self.allocation = config['allocation']
        self.spending = config['spending']
        self.current_trial = Trialdata()

    def __repr__(self):
        retstr = "Simulator:\n"
        retstr += pprint.pformat(self.simulator)
        retstr += "\n\nAllocation:\n"
        retstr += pprint.pformat(self.allocation)
        retstr += "\n\nSpending:\n"
        retstr += pprint.pformat(self.spending)
        return retstr

    def get_allocations(self):
        if self.current_trial.iteration == 0:
            self.allocation['equal_weight'] = np.ones(self.simulator['n_assets']) / self.simulator['n_assets']

        return self.allocation['equal_weight']

    def get_spend(self):
        if self.current_trial.iteration == 0:
            # initialize spending parameters
            self.spending['variable'] = self.spending['variable_pct'] / 100
            self.spending['fixed'] = self.spending['fixed_pct'] / 100 * START_PORTVAL

        portval = self.current_trial.portval
        return portval * self.spending['variable'] + self.spending['fixed']

    def eval_trial(self):
        min_end_port_index = int(np.argmin(self.current_trial.end_ports))
        min_end_port_value = self.current_trial.end_ports[min_end_port_index]
        if min_end_port_value == 0.0:
            return min_end_port_index
        else:
            return self.simulator['n_ret_years']

    def simulate(self, do_eval=False, return_all=True):
        """simulate many trials"""

        retlist = []

        for trial in self.simulator['trials']:
            trial_df = self.simulate_trial(trial)

            if do_eval:
                eval_metric = self.eval_trial()
                if return_all:  # dataframe and eval_metric
                    retlist.append([trial_df, eval_metric])
                else:  # eval metric only
                    retlist.append(eval_metric)
            else:  # dataframe only
                retlist.append(trial_df)

        return retlist

    def simulate_trial(self, trial_rows):
        """simulate a single trial"""

        self.current_trial = Trialdata()
        current_trial = self.current_trial

        for i, t in enumerate(trial_rows):
            current_trial.iteration = i
            year, asset_returns = t[0], t[1:]
            current_trial.years.append(year)
            current_trial.start_ports.append(current_trial.portval)

            asset_allocations = self.get_allocations()
            current_trial.asset_allocations.append(asset_allocations)

            port_return = asset_allocations @ np.array(asset_returns)
            current_trial.port_returns.append(port_return)

            current_trial.portval *= (1 + port_return)
            current_trial.before_spends.append(current_trial.portval)

            current_trial.spend = self.get_spend()  # desired spend
            current_trial.spend = min(current_trial.spend, current_trial.portval)  # actual spend
            current_trial.spends.append(current_trial.spend)

            current_trial.portval = current_trial.portval - current_trial.spend
            current_trial.end_ports.append(current_trial.portval)

        # pd.DataFrame(data=np.vstack(z.asset_allocations), index=z.years, columns=["alloc_%d" % i for i in range(2)])
        ret_df = pd.DataFrame(index=current_trial.years,
                              data={'start_port': current_trial.start_ports,
                                    'port_return': current_trial.port_returns,
                                    'before_spend': current_trial.before_spends,
                                    'spend': current_trial.spends,
                                    'end_port': current_trial.end_ports,
                                    })
        alloc_df = pd.DataFrame(data=np.vstack(current_trial.asset_allocations),
                                index=current_trial.years,
                                columns=["alloc_%d" % i for i in range(2)])
        current_trial.trial_df = pd.concat([ret_df, alloc_df], axis=1)
        return current_trial.trial_df


if __name__ == '__main__':
    print('Executing as standalone script')
