import numpy as np
import pandas as pd
import pprint
from dataclasses import dataclass, field
from typing import List

from plotly import graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

import matplotlib.pyplot as plt

START_PORTVAL = 100.0


@dataclass
class Trialdata:
    """Class for keeping track of a latest_trial."""
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
    """simulate retirement outcomes.

        Attributes:
            config: dict with options for simulation, allocation, withdrawal, evaluation
            simulation['n_ret_years']: number of years of retirement to simulate
            simulation['n_hist_years']: number of years of historical returns available
            simulation['n_assets']: number of assets we have returns for
            simulation['trials']: iterator that yields trials (each an iterator of n_ret_years of asset returns )
        """

    def __init__(self, config):
        """pass a dict of config"""
        # promote everything in config to instance variables for more readable code
        self.simulation = config.get('simulation')
        self.allocation = config.get('allocation')
        self.withdrawal = config.get('withdrawal')
        self.analysis = config.get('analysis')
        self.latest_trial = Trialdata()
        self.latest_simulation = []  # list of all trials in latest simulation

        if 'trials' in self.simulation:
            # iterator passed directly, possibly do additional checks
            pass
        elif 'trial_gen' in self.simulation:
            # function passed that returns iterator
            self.simulation['trials'] = self.simulation['trial_gen']()
        elif 'returns_df' in self.simulation:
            # create trials based on returns_df
            self.simulation['n_asset_years'], self.simulation['n_assets'] = self.simulation['returns_df'].shape
            if self.simulation.get('montecarlo'):
                replace = self.simulation.get('montecarlo_replacement')
                self.simulation['trials'] = self.montecarlo_trials(n_trials=self.simulation.get('montecarlo'),
                                                                   replace=replace)
            else:  # create historical trials
                self.simulation['trials'] = self.historical_trials()
        else:
            raise Exception("Must config either 'trials' iterator, or 'trial_gen' generator function, or 'returns_df'")

    def __repr__(self):
        retstr = "Simulation:\n"
        retstr += pprint.pformat(self.simulation)
        retstr += "\n\nAllocation:\n"
        retstr += pprint.pformat(self.allocation)
        retstr += "\n\nWithdrawal:\n"
        retstr += pprint.pformat(self.withdrawal)
        return retstr

    def get_allocations(self):
        """equal-weight allocations"""
        if self.latest_trial.iteration == 0:
            self.allocation['equal_weight'] = np.ones(self.simulation['n_assets']) / self.simulation['n_assets']

        return self.allocation['equal_weight']

    def get_spend(self):
        """fixed + variable based on config"""
        if self.latest_trial.iteration == 0:
            # initialize withdrawal parameters
            self.withdrawal['variable'] = self.withdrawal['variable_pct'] / 100
            self.withdrawal['fixed'] = self.withdrawal['fixed_pct'] / 100 * START_PORTVAL

        portval = self.latest_trial.portval
        return portval * self.withdrawal['variable'] + self.withdrawal['fixed']

    def eval_trial(self):
        """which year portfolio is exhausted, or n_ret_years if never exhausted"""
        min_end_port_index = int(np.argmin(self.latest_trial.end_ports))
        min_end_port_value = self.latest_trial.end_ports[min_end_port_index]
        if min_end_port_value == 0.0:
            return min_end_port_index
        else:
            return self.simulation['n_ret_years']

    def historical_trial_generator(self, start_year):
        """generate asset returns for 1 latest_trial, n_ret_years long, given a dataframe of returns, starting year"""
        df = self.simulation['returns_df']
        n_ret_years = self.simulation['n_ret_years']
        for t in df.loc[start_year:start_year + n_ret_years - 1].itertuples():
            yield tuple(t)

    def historical_trials(self):
        """generate all available n_ret_years historical trials """
        df = self.simulation['returns_df']
        first_year = df.index[0]
        last_year = df.index[-1] - self.simulation['n_ret_years'] + 1

        for year in range(first_year, last_year + 1):
            yield self.historical_trial_generator(year)

    def montecarlo_trial_generator(self, replace=False):
        """generate 1 latest_trial, n_years long, by sampling randomly from returns"""
        df = self.simulation['returns_df']
        n_ret_years = self.simulation['n_ret_years']
        sample = np.random.choice(len(df), n_ret_years, replace=replace)
        for t in df.iloc[sample].itertuples():
            yield tuple(t)

    def montecarlo_trials(self, n_trials, replace=False):
        """generate n_trials trials, each n_years long, by sampling randomly"""
        for i in range(n_trials):
            yield self.montecarlo_trial_generator(replace=replace)

    def simulate(self, do_eval=False, return_both=True):
        """simulate many trials, return a list of latest_trial dataframes and/or optional evaluation metrics"""

        self.latest_simulation = []

        for trial in self.simulation['trials']:
            # run 1 latest_trial
            trial_df = self.simulate_trial(trial)

            if do_eval:
                # evaluate latest_trial
                eval_metric = self.eval_trial()
                if return_both:  # both dataframe and eval_metric
                    self.latest_simulation.append([trial_df, eval_metric])
                else:  # eval metric only
                    self.latest_simulation.append(eval_metric)
            else:  # dataframe only
                self.latest_simulation.append(trial_df)

        return self.latest_simulation

    def simulate_trial(self, trial_rows):
        """simulate a single latest_trial"""

        self.latest_trial = Trialdata()
        current_trial = self.latest_trial

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

    def analyze(self):

        if len(self.latest_simulation) > 100 or self.analysis.get('histogram'):
            # histogram
            # TODO: add more logic, save return_both and act accordingly
            start_years = [i for i in range(len(self.latest_simulation))]
            survival = [np.sum(np.where(trial['end_port'].values > 0, 1, 0))
                        for trial in self.latest_simulation]

            c, bins = np.histogram(survival, bins=np.linspace(0, 30, 31))
            pct_exhausted = np.sum(c[:-1]) / np.sum(c) * 100
            print("%.2f%% of portfolios exhausted before final year" % pct_exhausted)
            fig, axs = plt.subplots(2, figsize=(20, 20))
            axs[0].set_title("Histogram of Years to Exhaustion (1000 Trials)", fontsize=20)
            axs[0].set_yscale('log')
            axs[0].set_ylabel('Portfolio Years to Exhaustion (Log Scale)', fontsize=16)
            axs[0].set_xlabel('Retirement Year', fontsize=16)
            axs[0].bar(bins[1:], c)

        else:
            # bar chart of all simulation outcomes
            start_years = [trial.index[0] for trial in self.latest_simulation]
            survival = [np.sum(np.where(trial['end_port'].values > 0, 1, 0))
                        for trial in self.latest_simulation]
            years_survived_df = pd.DataFrame(data={'nyears': survival},
                                          index=start_years)

            fig, axs = plt.subplots(2, figsize=(20, 20))
            axs[0].set_title("Years to Exhaustion by Retirement Year", fontsize=20)
            axs[0].set_ylabel('Years to Exhaustion', fontsize=16)
            axs[0].set_xlabel('Retirement Year', fontsize=16)
            axs[0].bar(years_survived_df.index, years_survived_df['nyears'])

        portvals = np.array([trial['end_port'].values for trial in self.latest_simulation])
        portval_rows, portval_cols = portvals.shape
        portval_df = pd.DataFrame(data=np.hstack([(np.ones(portval_rows).reshape(portval_rows, 1) * 100), portvals]).T,
                                  columns=start_years)

        axs[1].set_title("Portfolio Value by Retirement Year", fontsize=20)
        axs[1].set_ylabel('Portfolio Value', fontsize=16)
        axs[1].set_xlabel('Retirement Year', fontsize=16)
        for startyear in start_years:
            axs[1].plot(portval_df.index, portval_df[startyear], alpha=0.2)
        axs[1].plot(portval_df.index, portval_df.mean(axis=1), lw=5, c='black')
        return plt.show()


    def analyze_plotly(self):
        start_years = [trial.index[0] for trial in self.latest_simulation]
        survival = [np.sum(np.where(trial['end_port'].values > 0, 1, 0))
                    for trial in self.latest_simulation]
        years_survived = pd.DataFrame(data={'nyears': survival},
                                      index=start_years).reset_index()

        portvals = np.array([trial['end_port'].values for trial in self.latest_simulation])
        years = [trial.index[0] for trial in self.latest_simulation]
        portval_df = pd.DataFrame(data=np.hstack([(np.ones(64).reshape(64, 1) * 100), portvals]).T,
                                  columns=years)
        portval_df['mean'] = portval_df.mean(axis=1)
        portval_df.reset_index(inplace=True)

        fig = make_subplots(rows=2, cols=1,
                            subplot_titles=("Years to Exhaustion", "Portfolio Value By Year"))

        fig.add_trace(
            go.Bar(x=years_survived['index'], y=years_survived['nyears']),
            row=1, col=1
        )

        for year in range(1928, 1992):
            fig.add_trace(go.Scatter(x=portval_df['index'],
                                     y=portval_df[year],
                                     mode='lines',
                                     name=str(year),
                                     line={'width': 1},
                                     ),
                          row=2, col=1
                          )

        fig.add_trace(go.Scatter(x=portval_df['index'],
                                 y=portval_df['mean'],
                                 mode='lines',
                                 name='Mean',
                                 line={'width': 3, 'color': 'black'},
                                 ),
                      row=2, col=1
                      )

        fig.update_layout(showlegend=False,
                          plot_bgcolor="white",
                          height=800, width=700,
                          )

        fig.update_yaxes(title="Number of years to exhaustion",
                         linecolor='black', mirror=True, ticks='inside',
                         row=1, col=1)

        fig.update_xaxes(title="Retirement Year",
                         linecolor='black', mirror=True, ticks='inside',
                         row=1, col=1)

        fig.update_yaxes(title="Portfolio Value",
                         linecolor='black', mirror=True, ticks='inside',
                         row=2, col=1)

        fig.update_xaxes(title="Retirement Year",
                         linecolor='black', mirror=True, ticks='inside',
                         row=2, col=1)

        return fig

    def analyze_plotly_express(self):
        """output px survival and port value charts"""
        start_years = [trial.index[0] for trial in self.latest_simulation]
        survival = [self.simulation['n_ret_years'] - len(np.where(trial['spend'].values == 0.0))
                    for trial in self.latest_simulation]
        years_survived = pd.DataFrame(data={'nyears': survival},
                                      index=start_years).reset_index()

        return px.bar(years_survived, x="index", y="nyears", color="nyears",
               hover_name="index", color_continuous_scale="spectral")

    def analyze_plotly_express2(self):

        portvals = np.array([trial['end_port'].values for trial in self.latest_simulation])
        portval_df = pd.DataFrame(data=np.hstack([(np.ones(64).reshape(64, 1) * 100), portvals])).transpose()
        col_list = [trial.index[0] for trial in self.latest_simulation]
        portval_df.columns = col_list
        portval_df['mean'] = portval_df.mean(axis=1)
        portval_df.reset_index(inplace=True)

        portval_melt = pd.melt(portval_df, id_vars=['index'], value_vars=col_list)
        portval_melt.columns = ['ret_year', 'start_year', 'portval']
        return px.line(portval_melt,
                       x="ret_year",
                       y="portval",
                       color="start_year",
                       hover_name="start_year")

if __name__ == '__main__':
    print('Executing as standalone script')
