import numpy as np
# import pandas as pd

int_types = (int, np.int8, np.int16, np.int32, np.int64)
float_types = (float, np.float16, np.float32, np.float64)
numeric_types = (int, np.int8, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64)


class Simulator:
    """simulate retirement outcomes.

        Attributes:
            return_df: pandas dataframe of asset returns
            n_years: number of years of returns available
            n_assets: number of assets we have returns for
            n_ret_years: number of years of retirement to simulate
            first_year: first year we have returns for
            last_year: last year we have returns for (first_year + n_years -1)
            montecarlo: boolean: pick years at random for simulation, or simulate all historical periods
            replacement = if montecarlo, pick random years with replacement or no replacement
        """

    def __init__(self, return_df, n_ret_years, montecarlo=False, replacement=False):
        """takes dataframe of annual asset returns, length of retirement to simulate"""

        self.return_df = return_df.copy()
        self.n_ret_years = n_ret_years
        self.n_years, self.n_assets = return_df.shape
        self.first_year = return_df.index[0]
        self.last_year = return_df.index[-1]
        self.montecarlo = montecarlo
        self.replacement = replacement

    def simulate_trial(self, start_year, weights, fixed, variable):
        """simulate a single trial

        Args:
            start_year: Retirement trial starts in start_year, lasts for self.n_ret_years
            weights: a list of floats or arrays.
                list must be length n_assets
                if weight is a float, represents constant asset weight throughout n_ret_years trial
                if weight is an array, must be length n_ret_years,
                weights must sum to 1
            fixed: fixed annual spending
            variable: variable annual spending as fraction of current port

        Returns:
            dataframe of columns describing retirement outcome (annual starting port values, ending, amount spent)

        Raises:
          AssertionError: a bad value.

        """
        assert start_year >= self.first_year, "no data for start year"
        assert start_year <= self.last_year - self.n_ret_years, "not enough data"
        ret_df = self.return_df.loc[start_year:start_year + self.n_ret_years-1].copy()

        assert len(weights) == self.n_assets, "wrong number of weights"
        if type(weights[0]) in numeric_types:
            assert all([type(w) in numeric_types for w in weights]), "wrong weight numeric type"
            assert sum(weights) == 1.0, "weights don't add up to 1"
        elif type(weights[0] == list):
            assert all([type(w) == list for w in weights]), "all weights must be numeric or list"
            assert all([len(w) == self.n_ret_years for w in weights]), "wrong weight list length"
        ret_df['perf'] = 0
        for i, w in enumerate(weights):
            ret_df['w_%d' % i] = w
            ret_df['perf'] += w * ret_df.iloc[:, i]

        portval = 100
        ret_df['fixed'] = fixed * portval
        ret_df['variable'] = variable
        ret_df['variable_1m'] = 1 - variable
        ret_df['perf_1p'] = 1 + ret_df['perf']

        startvals = []
        endvals = []
        spend = []

        for t in ret_df.itertuples():
            startvals.append(portval)
            spend.append(portval * t.variable + t.fixed)
            portval = portval * t.perf_1p * t.variable_1m - t.fixed
            endvals.append(portval)
        ret_df['starting'] = startvals
        ret_df['spend'] = spend
        ret_df['ending'] = endvals

        return ret_df

# simulate
#   actual_year= value in range(n-k+1)

# simulate_many
#   startyear
#   nyears
#   returns list of nyears dataframes n x m with the cash flows

# simulate_many_mc
#   n_trials
#   returns list of n_trials dataframes n x m with the cash flows


if __name__ == '__main__':
    print('Executing as standalone script')
