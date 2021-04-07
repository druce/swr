import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import pdb
from plotly import graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

from .SWRsimulation import SWRsimulation,  START_PORTVAL, Trialdata
from .eval_metrics import eval_ce, eval_exhaustion, eval_mean_spend, eval_median_spend, eval_min_spend, \
    eval_max_spend, eval_sd_spend


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
        self.visualization: dict of visualization configs
            'histogram': bo'olean all return year metrics or histogram of metrics
            'chart_1', 'chart_2', 'chart_3': matplotlib options for charts
    """

    def __init__(self, config):
        """initialize simulation from a config dict

        Args:
            config (dict): simulation, allocation, withdrawal, visualization keys
        """

        # promote everything in config to instance variables and call inits
        super().__init__(config)

        self.latest_trial = Trialdata()
        self.latest_simulation = []  # list of all trial data in latest simulation

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
        set smoothing factor (default=1)
        """

        # initialize withdrawal parameters
        if self.withdrawal.get('variable_pct') is None:
            self.withdrawal['variable_pct'] = 0.0
        if self.withdrawal.get('fixed_pct') is None:
            self.withdrawal['fixed_pct'] = 0.0
        self.withdrawal['variable'] = self.withdrawal['variable_pct'] / 100
        self.withdrawal['fixed'] = self.withdrawal['fixed_pct'] / 100 * START_PORTVAL
        self.withdrawal['floor'] = self.withdrawal['floor_pct'] / 100 * START_PORTVAL

        # initialize smoothing parameter (disabled)
        # if self.withdrawal.get('smoothing_factor') is None:
        #     self.withdrawal['smoothing_factor'] = 1.0
        
    def get_withdrawal(self):
        """return withdrawal for current iteration
        compute desired withdrawal based on config and current iteration state (fixed + variable)
        return min(desired, EMA(smoothing_factor)

        Returns:
            float: withdrawal for current iteration
        """
        portval = self.latest_trial.portval
        desired_withdrawal = portval * self.withdrawal['variable'] + self.withdrawal['fixed']
        # max floor, desired
        desired_withdrawal = max(desired_withdrawal, self.withdrawal['floor'])
        
        # smoothing factor (not currently supported, didn't improve outcomes)
        # if self.latest_trial.spends:
        #     previous_withdrawal = self.latest_trial.spends[-1]
        #     smoothed_withdrawal = previous_withdrawal + (desired_withdrawal - previous_withdrawal)/self.withdrawal['smoothing_factor']
        #     desired_withdrawal = min(desired_withdrawal, smoothed_withdrawal)

        # can't spend more than portfolio
        return min(desired_withdrawal, self.latest_trial.portval)

    def eval_trial(self):
        """compute all metrics and return in dict

        Returns:
            dict: key = name of metric, value = metric
        """
        exhaustion, min_end_port, min_end_port_index = eval_exhaustion(self)
        return {'exhaustion': exhaustion,
                'min_end_port': min_end_port,
                'min_end_port_index': min_end_port_index,
                'ce_spend': eval_ce(self),
                'median_spend': eval_median_spend(self),
                'mean_spend': eval_mean_spend(self),
                'min_spend': eval_min_spend(self),
                'max_spend': eval_max_spend(self),
                'sd_spend': eval_sd_spend(self),
                }
    
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

    def table_metrics(self):

        table_dict = {}
        mean_spends = [trial_dict['mean_spend']
                       for trial_dict in self.latest_simulation]
        table_dict["mean annual spending over all cohorts"] = np.mean(mean_spends)

        sd_spends = [trial_dict['sd_spend']
                     for trial_dict in self.latest_simulation]
        table_dict["mean within-cohort standard deviation of spending"] = np.mean(sd_spends)
        
        min_spends = [trial_dict['min_spend']
                      for trial_dict in self.latest_simulation]
        table_dict["lowest annual spending over all cohorts"] = np.min(min_spends)
        # could add cohort year and year index

        min_port_values  = np.array([trial_dict['min_end_port'] for trial_dict in self.latest_simulation])
        min_port_value = np.min(min_port_values)
        table_dict["minimum ending portfolio over all cohorts"] = min_port_value
        # min_port_value_indexes = np.where(min_port_values==min_port_value, 1, 0)
        # min_port_value_indexes = [i for i, m in enumerate(min_port_value_indexes) if m==1]
        # min_port_value_years = [self.latest_simulation[i]['trial'].index[0] for i in min_port_value_indexes]
        # table_dict["minimum ending portfolio in years"] = str(min_port_value_years)
        
        survival = [trial_dict['exhaustion']
                    for trial_dict in self.latest_simulation]
        # beware the off-by-1, need N_RET_YEARS + 1 bins, e.g. 1 to 31, so range(1,32)
        # 1 = spend all money first year, 31 = never run out of money even after 30 years
        # 29 = spend last cent in last year
        c, bins = np.histogram(survival, bins=list(range(1,self.simulation['n_ret_years']+2)))
        pct_exhausted = np.sum(c[:-1]) / np.sum(c) * 100
        table_dict["% cohort portfolios exhausted by final year"] = pct_exhausted
        
        retdf = pd.DataFrame(table_dict, index=[0]).transpose().reset_index()
        retdf.columns=['metric','value']
        return retdf

    
    def chart_1_histogram(self, ax, optionlabel="chart_1"):
        
        # histogram
        # TODO: add more logic, save return_both and act accordingly
        start_years = [i for i in range(len(self.latest_simulation))]

        survival = [trial_dict['exhaustion']
                    for trial_dict in self.latest_simulation]
        # beware the off-by-1, need N_RET_YEARS + 1 bins, e.g. 1 to 31, so range(1,32)
        # 1 = spend all money first year, 31 = never run out of money even after 30 years
        # 29 = spend last cent in last year
        c, bins = np.histogram(survival, bins=list(range(1,self.simulation['n_ret_years']+2)))

        mpl_options = {
            'title': "Histogram of Years to Exhaustion",
            'title_fontsize': 20,
            'ylabel': 'Portfolio Years to Exhaustion (Log Scale)',
            'ylabel_fontsize': 16,
            'xlabel': 'Retirement Year',
            'xlabel_fontsize': 16,
        }
        # merge from visualize options
        chart_options = self.visualization.get(optionlabel)
        if chart_options:
            mpl_options = {**mpl_options, **chart_options}

        ax.set_title(mpl_options['title'], fontsize=mpl_options['title_fontsize'])
        ax.set_yscale('log')
        ax.set_ylabel(mpl_options['ylabel'], fontsize=mpl_options['ylabel_fontsize'])
        ax.set_xlabel(mpl_options['xlabel'], fontsize=mpl_options['xlabel_fontsize'])
        ax.tick_params(axis='both', labelsize=16, )
        ax.bar(bins[1:], c)

        if mpl_options.get('annotation'):
            ax.annotate(mpl_options.get('annotation'),
                            xy=(0.073, 0.925), xycoords='figure fraction', fontsize=16)

    def chart_1_bar(self, ax, optionlabel="chart_1"):
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

        # merge from visualize options
        chart_options = self.visualization.get(optionlabel)
        if chart_options:
            mpl_options = {**mpl_options, **chart_options}
            
        ax.set_title(mpl_options['title'], fontsize=mpl_options['title_fontsize'])
        ax.set_ylabel(mpl_options['ylabel'], fontsize=mpl_options['ylabel_fontsize'])
        ax.set_xlabel(mpl_options['xlabel'], fontsize=mpl_options['xlabel_fontsize'])
        ax.tick_params(axis='both', labelsize=16, )
        ax.bar(years_survived_df.index, years_survived_df['nyears'])

        if mpl_options.get('annotation'):
            ax.annotate(mpl_options.get('annotation'),
                            xy=(0.073, 0.92), xycoords='figure fraction', fontsize=16)


    def chart_2_lines(self, ax, optionlabel="chart_2"):
        start_years = [i for i in range(len(self.latest_simulation))]
        spends = np.array([trial_dict['trial']['spend'].values for trial_dict in self.latest_simulation])
        spend_df = pd.DataFrame(data=spends.T,
                                columns=start_years)

        mpl_options = {
            'title': "Spending by Retirement Year",
            'title_fontsize': 20,
            'ylabel': 'Spending (% of starting portfolio)',
            'ylabel_fontsize': 16,
            'xlabel': 'Retirement Year',
            'xlabel_fontsize': 16,
        }

        # merge from visualize options
        chart_options = self.visualization.get(optionlabel)
        if chart_options:
            mpl_options = {**mpl_options, **chart_options}
        
        ax.set_title(mpl_options['title'], fontsize=mpl_options['title_fontsize'])
        ax.set_ylabel(mpl_options['ylabel'], fontsize=mpl_options['ylabel_fontsize'])
        ax.set_xlabel(mpl_options['xlabel'], fontsize=mpl_options['xlabel_fontsize'])
        ax.tick_params(axis='both', labelsize=16, )

        # color by ending spend
        ending_vals = spend_df.values[-1, :].copy()
        ending_vals /= np.max(ending_vals)
        colors = [cm.plasma(x) for x in ending_vals]
        
        for i, startyear in enumerate(start_years):
            ax.plot(spend_df.index, spend_df[startyear], lw=2, alpha=0.2, c=colors[i])
        # ax.plot(spend_df.index, spend_df.median(axis=1), lw=2, c='black')

        quantile25 = np.quantile(spend_df, .25, axis=1)
        quantile75 = np.quantile(spend_df, .75, axis=1)
        ax.fill_between(spend_df.index, quantile25, quantile75, alpha=0.2, color='orange')
        ax.plot(spend_df.index, np.array([0]*len(spend_df)), lw=2, c='black', ls='dashed', alpha=0.5)


    def chart_3_lines(self, ax, optionlabel="chart_3"):
        
        start_years = [i for i in range(len(self.latest_simulation))]
        portvals = np.array([trial_dict['trial']['end_port'].values for trial_dict in self.latest_simulation])
        portval_rows, portval_cols = portvals.shape
        portval_df = pd.DataFrame(data=np.hstack([(np.ones(portval_rows).reshape(portval_rows, 1) * START_PORTVAL), portvals]).T,
                                  columns=start_years)

        mpl_options = {
            'title': "Portfolio Value by Retirement Year",
            'title_fontsize': 20,
            'ylabel': 'Portfolio Value',
            'ylabel_fontsize': 16,
            'xlabel': 'Retirement Year',
            'xlabel_fontsize': 16,
        }

        # merge from visualize options
        chart_options = self.visualization.get(optionlabel)
        if chart_options:
            mpl_options = {**mpl_options, **chart_options}
        
        ax.set_title(mpl_options['title'], fontsize=mpl_options['title_fontsize'])
        ax.set_ylabel(mpl_options['ylabel'], fontsize=mpl_options['ylabel_fontsize'])
        ax.set_xlabel(mpl_options['xlabel'], fontsize=mpl_options['xlabel_fontsize'])
        ax.tick_params(axis='both', labelsize=16, )

        # color by ending portval
        ending_vals = portval_df.values[-1, :].copy()
        ending_vals /= np.max(ending_vals)
        colors = [cm.plasma(x) for x in ending_vals]
        
        for i, startyear in enumerate(start_years):
            ax.plot(portval_df.index, portval_df[startyear], lw=2, alpha=0.2, c=colors[i])
        ax.plot(portval_df.index, portval_df.median(axis=1), lw=3, c='black')
        ax.plot(portval_df.index, np.array([100]*len(portval_df)), lw=2, c='black', ls='dashed', alpha=0.5)
        ax.plot(portval_df.index, np.array([0]*len(portval_df)), lw=2, c='black', ls='dashed', alpha=0.5)
        
        quantile25 = np.quantile(portval_df, .25, axis=1)
        quantile75 = np.quantile(portval_df, .75, axis=1)
        ax.fill_between(portval_df.index, quantile25, quantile75, color='orange', alpha=0.2)

        
    def chart_spending_by_portval(self, ax, optionlabel="chart_4"):
        
        portvals = np.linspace(0, 200, 2001)
        test_sim = SWRsimulationCE({
            'simulation': {'returns_df': self.simulation['returns_df'],
                           'n_ret_years': self.simulation['n_ret_years'],
            },
            'allocation': {'asset_weights': np.array([0.5, 0.5])},
            'withdrawal': {'fixed_pct': self.withdrawal['fixed_pct'],
                           'variable_pct': self.withdrawal['variable_pct'],
                           'floor_pct': self.withdrawal['floor_pct'],
            },
        })

        spendvals = []

        for portval in portvals:
            test_sim.latest_trial.portval = portval
            spendvals.append(test_sim.get_withdrawal())

        mpl_options = {
            'title': "Withdrawal Rule Profile (Withdrawal Value vs. Portfolio Value)",
            'title_fontsize': 20,
            'ylabel': 'Portfolio Value',
            'ylabel_fontsize': 16,
            'xlabel': 'Withdrawal value',
            'xlabel_fontsize': 16,
        }

        # merge from visualize options
        chart_options = self.visualization.get(optionlabel)
        if chart_options:
            mpl_options = {**mpl_options, **chart_options}
        
        ax.set_title(mpl_options['title'], fontsize=mpl_options['title_fontsize'])
        ax.set_ylabel(mpl_options['ylabel'], fontsize=mpl_options['ylabel_fontsize'])
        ax.set_xlabel(mpl_options['xlabel'], fontsize=mpl_options['xlabel_fontsize'])
        ax.tick_params(axis='both', labelsize=16, )

        ax.plot(portvals, spendvals, lw=2, c='black')

        
    def visualize(self):
        """display metrics and dataviz for the current simulation with matplotlib

        Returns:
            matplotlib chart object: charts
        """

        # display table of results
        with pd.option_context('display.precision', 8):
            display(self.table_metrics())

        # do 3 charts 
        fig, axs = plt.subplots(4, figsize=(20, 40))

        if len(self.latest_simulation) > 100 or self.visualization.get('histogram'):
            self.chart_1_histogram(axs[0])
        else:
            self.chart_1_bar(axs[0])

        self.chart_2_lines(axs[1])

        self.chart_3_lines(axs[2])

        self.chart_spending_by_portval(axs[3])

        return plt.show()

    def visualize_plotly(self):
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
        n_cohorts = len(self.latest_simulation)
        portval_df = pd.DataFrame(data=np.hstack([(np.ones(n_cohorts).reshape(n_cohorts, 1) * START_PORTVAL), portvals]).T,
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

    def visualize_plotly_express(self):
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

    def visualize_plotly_express2(self):
        """display metrics and dataviz for the current simulation

        Returns:
            Plotly Express chart object: chart
        """
        portvals = np.array([trial_dict['trial']['end_port'].values for trial_dict in self.latest_simulation])
        n_cohorts = len(self.latest_simulation)
        portval_df = pd.DataFrame(data=np.hstack([(np.ones(n_cohorts).reshape(n_cohorts, 1) * START_PORTVAL), portvals])).transpose()
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
