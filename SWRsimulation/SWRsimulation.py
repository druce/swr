import numpy as np
import pandas as pd
import pprint
from dataclasses import dataclass, field
from typing import List

# from plotly import graph_objects as go
# from plotly.subplots import make_subplots
# import plotly.express as px

# TODO: by default don't keep all trials, only if specified explicitly

import matplotlib.pyplot as plt

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
                'analysis': {}
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
        self.analysis = config.get('analysis')

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

    def analyze(self):
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


class SWRsimulationCE(SWRsimulation):
    """simulate retirement outcomes and evaluate CRRA certainty-equivalent spending using a risk aversion parameter

    Inherits from SWRsimulation ([type])

    Attributes:
        self.simulation:  dict of simulation configs 
            'gamma': risk aversion parameter
            'n_assets': number of assets
            'n_asset_years': historical asset return years
            'n_ret_years': retirement years
            'trials': trial cohort iterator
            'latest_trial': TrialData for current trial cohort
            'latest_simulation': list of all data for current simulation

            can create trials iterator using optional configs
            'returns_df': historical returns dataframe
            'montecarlo': boolean, use montecarlo to generate trial cohorts (default is historical cohorts)
            'montecarlo_replacement': do montecarlo with reaplcement
            'trial_gen': generate trials using a supplied generator function
        self.allocation: dict of allocation configs
            'asset_weights': array of fixed weights for assets
        self.withdrawal: dict of withdrawal configs
            'fixed_pct': input fixed withdrawal pct
            'variable_pct': input variable withdrawal pct
            'fixed': withdrawal in fixed units, initial portfolio * fixed_pct / 100
            'variable': withdrawal in variable fraction, variable_pct/100
        self.evaluation: dict of evaluation configs
            'gamma': risk aversion parameter
        self.analysis: dict of analysis configs
            'histogram': bo'olean all return year metrics or histogram of metrics
            'chart_1', chart_2: matplotlib options for charts
    """

    def __init__(self, config):
        """initialize simulation from a config dict

        Args:
            config (dict): simulation, allocation, withdrawal, analysis keys
        """

        # promote everything in config to instance variables and call inits
        super().__init__(config)

    def init_simulation(self):
        """initialize / reinitialize simulation based on configs
        make trials ready for next()

        Raises:
            Exception: bad values

        """
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

    def init_allocation(self):
        """initialize for allocation based on configs
        compute fixed asset weights if not given
        """
        if self.allocation.get('asset_weights') is None:
            # default equal-weighted
            self.allocation['asset_weights'] = np.ones(self.simulation['n_assets']) / self.simulation['n_assets']

    def get_allocations(self):
        """ return default allocation

        Returns:
            numpy.ndarray: asset weights for this iteration
        """
        return self.allocation['asset_weights']

    def init_withdrawal(self):
        """initialize for withdrawal based on configs
        compute variable and fixed fractions
        """

        if self.withdrawal.get('variable_pct') is None:
            self.withdrawal['variable_pct'] = 0.0
        if self.withdrawal.get('fixed_pct') is None:
            self.withdrawal['fixed_pct'] = 0.0
        # initialize withdrawal parameters
        self.withdrawal['variable'] = self.withdrawal['variable_pct'] / 100
        self.withdrawal['fixed'] = self.withdrawal['fixed_pct'] / 100 * START_PORTVAL

    def get_withdrawal(self):
        """return withdrawal for current iteration
        fixed + variable based on config and current iteration state

        Returns:
            float: withdrawal for current iteration
        """
        portval = self.latest_trial.portval
        return portval * self.withdrawal['variable'] + self.withdrawal['fixed']

    def eval_exhaustion(self):
        """exhaustion metric for current trial

        Returns:
            float: years to exhaustion for current trial
        """
        min_end_port_index = int(np.argmin(self.latest_trial.end_ports))
        min_end_port_value = self.latest_trial.end_ports[min_end_port_index]
        if min_end_port_value == 0.0:
            return min_end_port_index
        else:
            return self.simulation['n_ret_years']

    def eval_ce(self):
        """certainty-equivalent metric for current trial

        Returns:
            float: CE cash flow for spending in current trial
        """
        return crra_ce(self.latest_trial.trial_df['spend'], self.evaluation['gamma'])
    
    def eval_trial(self):
        """compute all metrics and return in dict

        Returns:
            dict: key = name of metric, value = metric
        """
        return {'years_to_exhaustion': self.eval_exhaustion(),
                'ce_spend': self.eval_ce()}
    
    def historical_trial_generator(self, start_year):
        """generate asset returns for 1 latest_trial, n_ret_years long, given a dataframe of returns, starting year

        Args:
            start_year (int): index of starting trial

        Yields:
            tuple: named tuple of asset return for each asset for each year for n_ret_years starting from start_year
        """
        df = self.simulation['returns_df']
        n_ret_years = self.simulation['n_ret_years']
        for t in df.loc[start_year:start_year + n_ret_years - 1].itertuples():
            yield tuple(t)

    def historical_trials(self):
        """generate all available n_ret_years historical trials

        Yields:
            iterator: for each historical starting retirement year, iterator of all years in retirement cohort
        """
        df = self.simulation['returns_df']
        first_year = df.index[0]
        last_year = df.index[-1] - self.simulation['n_ret_years'] + 1

        for year in range(first_year, last_year + 1):
            yield self.historical_trial_generator(year)

    def montecarlo_trial_generator(self, replace=False):
        """generate 1 trial cohort, n_ret_years long, by sampling randomly from returns

        Args:
            replace (bool, optional): sample with replacement. Defaults to False.

        Yields:
            tuple: named tuple of asset returns for each year sampled for n_ret_years
        """

        df = self.simulation['returns_df']
        n_ret_years = self.simulation['n_ret_years']
        sample = np.random.choice(len(df), n_ret_years, replace=replace)
        for t in df.iloc[sample].itertuples():
            yield tuple(t)

    def montecarlo_trials(self, n_trials, replace=False):
        """generate n_trials trials, each n_years long, by sampling randomly


        Args:
            n_trials (int): number of montecarlo cohorts
            replace (bool, optional): sample with replacement. Defaults to False.

        Yields:
            iterator: montecarlo_generator, a single cohort iterator
        """

        for i in range(n_trials):
            yield self.montecarlo_trial_generator(replace=replace)

    def simulate(self, do_eval=True, return_both=True):
        """simulate many trials, return a list of latest_trial dataframes and/or optional evaluation metrics

        Args:
            do_eval (bool, optional): run evals for each cohort. Defaults to True.
            return_both (bool, optional): keep both eval metric dict and outocome df for each cohort. 
            Defaults to True.

        Returns:
            list: latest_simulation, list of metrics and/or trial dataframe for each cohort
        """
        self.latest_simulation = []

        for trial in self.simulation['trials']:
            # run 1 latest_trial
            trial_df = self.simulate_trial(trial)

            if do_eval:
                # evaluate latest_trial
                eval_metrics_dict = self.eval_trial()
                if return_both:  # both dataframe and eval_metric
                    df_dict = {'trial': trial_df}
                    self.latest_simulation.append({**df_dict, **eval_metrics_dict})  # merge dicts
                else:  # eval metric only
                    self.latest_simulation.append(eval_metrics_dict)
            else:  # dataframe only
                self.latest_simulation.append({'trial': trial_df})

        return self.latest_simulation

    def simulate_trial(self, trial_rows):
        """Simulate a single trial cohort, given asset returns iterator

        Args:
            trial_rows (list or iterator): asset returns for each year

        Returns:
            pandas dataframe: dataframe of trial outcomes by year
        """

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

            current_trial.spend = self.get_withdrawal()  # desired spend
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
        """display metrics and dataviz for the current simulation with matplotlib

        Returns:
            matplotlib chart object: charts
        """

        if len(self.latest_simulation) > 100 or self.analysis.get('histogram'):
            # histogram
            # TODO: add more logic, save return_both and act accordingly
            start_years = [i for i in range(len(self.latest_simulation))]

            mean_spends = [np.mean(trial_dict['trial']['spend'])
                           for trial_dict in self.latest_simulation]
            print("mean annual spending over all cohorts %.2f" % np.mean(mean_spends))
            
            survival = [np.sum(np.where(trial_dict['trial']['end_port'].values > 0, 1, 0))
                        for trial_dict in self.latest_simulation]

            c, bins = np.histogram(survival, bins=np.linspace(0, 30, 31))
            pct_exhausted = np.sum(c[:-1]) / np.sum(c) * 100
            print("%.2f%% of portfolios exhausted by final year" % pct_exhausted)

            fig, axs = plt.subplots(2, figsize=(20, 20))

            mpl_options = {
                'title': "Histogram of Years to Exhaustion",
                'title_fontsize': 20,
                'ylabel': 'Portfolio Years to Exhaustion (Log Scale)',
                'ylabel_fontsize': 16,
                'xlabel': 'Retirement Year',
                'xlabel_fontsize': 16,
            }
            # merge from analyze options
            chart_options = self.analysis.get('chart_1')
            if chart_options:
                mpl_options = {**mpl_options, **chart_options}

            axs[0].set_title(mpl_options['title'], fontsize=mpl_options['title_fontsize'])
            axs[0].set_yscale('log')
            axs[0].set_ylabel(mpl_options['ylabel'], fontsize=mpl_options['ylabel_fontsize'])
            axs[0].set_xlabel(mpl_options['xlabel'], fontsize=mpl_options['xlabel_fontsize'])
            axs[0].bar(bins[1:], c)

            if mpl_options.get('annotation'):
                axs[0].annotate(mpl_options.get('annotation'),
                                xy=(0.073, 0.92), xycoords='figure fraction', fontsize=14)

        else:
            # bar chart of all simulation outcomes
            start_years = [trial_dict['trial'].index[0] for trial_dict in self.latest_simulation]
            survival = [np.sum(np.where(trial_dict['trial']['end_port'].values > 0, 1, 0))
                        for trial_dict in self.latest_simulation]
            years_survived_df = pd.DataFrame(data={'nyears': survival},
                                             index=start_years)

            mpl_options = {
                'title': "Years to Exhaustion by Retirement Year",
                'title_fontsize': 20,
                'ylabel': 'Years to Exhaustion',
                'ylabel_fontsize': 16,
                'xlabel': 'Retirement Year',
                'xlabel_fontsize': 16,
            }

            # merge from analyze options
            chart_options = self.analysis.get('chart_1')
            if chart_options:
                mpl_options = {**mpl_options, **chart_options}
            
            fig, axs = plt.subplots(2, figsize=(20, 20))
            axs[0].set_title(mpl_options['title'], fontsize=mpl_options['title_fontsize'])
            axs[0].set_ylabel(mpl_options['ylabel'], fontsize=mpl_options['ylabel_fontsize'])
            axs[0].set_xlabel(mpl_options['xlabel'], fontsize=mpl_options['xlabel_fontsize'])
            axs[0].bar(years_survived_df.index, years_survived_df['nyears'])
            
            if mpl_options.get('annotation'):
                axs[0].annotate(mpl_options.get('annotation'),
                                xy=(0.073, 0.92), xycoords='figure fraction', fontsize=14)

        portvals = np.array([trial_dict['trial']['end_port'].values for trial_dict in self.latest_simulation])
        portval_rows, portval_cols = portvals.shape
        portval_df = pd.DataFrame(data=np.hstack([(np.ones(portval_rows).reshape(portval_rows, 1) * 100), portvals]).T,
                                  columns=start_years)

        mpl_options = {
            'title': "Portfolio Value by Retirement Year",
            'title_fontsize': 20,
            'ylabel': 'Portfolio Value',
            'ylabel_fontsize': 16,
            'xlabel': 'Retirement Year',
            'xlabel_fontsize': 16,
        }

        # merge from analyze options
        chart_options = self.analysis.get('chart_2')
        if chart_options:
            mpl_options = {**mpl_options, **chart_options}
        
        axs[1].set_title(mpl_options['title'], fontsize=mpl_options['title_fontsize'])
        axs[1].set_ylabel(mpl_options['ylabel'], fontsize=mpl_options['ylabel_fontsize'])
        axs[1].set_xlabel(mpl_options['xlabel'], fontsize=mpl_options['xlabel_fontsize'])
        for startyear in start_years:
            axs[1].plot(portval_df.index, portval_df[startyear], alpha=0.2)
        axs[1].plot(portval_df.index, portval_df.median(axis=1), lw=5, c='black')

        return plt.show()

    def analyze_plotly(self):
        """display metrics and dataviz for the current simulation with plotly

        Returns:
            Plotly chart object: chart
        """

        start_years = [trial_dict['trial'].index[0] for trial_dict in self.latest_simulation]
        survival = [np.sum(np.where(trial_dict['trial']['end_port'].values > 0, 1, 0))
                    for trial_dict in self.latest_simulation]
        years_survived = pd.DataFrame(data={'nyears': survival},
                                      index=start_years).reset_index()

        portvals = np.array([trial_dict['trial']['end_port'].values for trial_dict in self.latest_simulation])
        years = [trial_dict['trial'].index[0] for trial_dict in self.latest_simulation]
        portval_df = pd.DataFrame(data=np.hstack([(np.ones(64).reshape(64, 1) * 100), portvals]).T,
                                  columns=years)
        portval_df['median'] = portval_df.median(axis=1)
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
                                 y=portval_df['median'],
                                 mode='lines',
                                 name='Median',
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
        """display metrics and dataviz for the current simulation

        Returns:
            Plotly Express chart object: chart
        """
        start_years = [trial_dict['trial'].index[0] for trial_dict in self.latest_simulation]
        survival = [self.simulation['n_ret_years'] - len(np.where(trial_dict['trial']['spend'].values == 0.0))
                    for trial_dict in self.latest_simulation]
        years_survived = pd.DataFrame(data={'nyears': survival},
                                      index=start_years).reset_index()

        return px.bar(years_survived, x="index", y="nyears", color="nyears",
                      hover_name="index", color_continuous_scale="spectral")

    def analyze_plotly_express2(self):
        """display metrics and dataviz for the current simulation

        Returns:
            Plotly Express chart object: chart
        """
        portvals = np.array([trial_dict['trial']['end_port'].values for trial_dict in self.latest_simulation])
        portval_df = pd.DataFrame(data=np.hstack([(np.ones(64).reshape(64, 1) * 100), portvals])).transpose()
        col_list = [trial_dict['trial'].index[0] for trial_dict in self.latest_simulation]
        portval_df.columns = col_list
        portval_df['median'] = portval_df.median(axis=1)
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
