# test_simulation.py

# pip install pytest
# pytest -q test_simulation.py
import pytest
import numpy as np
import pandas as pd
from SWRsimulation import SWRsimulationCE

# mixed return and weights
RETURN_URL = 'http://www.stern.nyu.edu/~adamodar/pc/datasets/histretSP.xls'
RETURN_FILE = 'histretSP'


def download_returns():
    data_sheet = "Returns by year"
    # these may change as rows as Damodaran website is updated
    skiprows = range(17)
    skipfooter = 10
    download_df = pd.read_excel(RETURN_URL,
                                sheet_name=data_sheet,
                                skiprows=skiprows,
                                skipfooter=skipfooter)
    download_df = download_df.set_index('Year')
    download_df.to_pickle('%s.pickle' % RETURN_FILE)
    return download_df


def load_returns():
    return pd.read_pickle('%s.pickle' % RETURN_FILE)


def trial_generator(df, start_year, n_years):
    """given a dataframe of returns, starting year, number of years, generate schedule of returns"""
    for t in df.loc[start_year:start_year+n_years-1].itertuples():
        yield tuple(t)


def test_zero():
    """no returns, no spending, just check shape"""
    download_df = load_returns()
    return_df = download_df.iloc[:, [0, 2]].copy()
    return_df.columns = ['stocks', 'tbonds']
    return_df['stocks'] = 0
    return_df['tbonds'] = 0

    trials = [trial_generator(return_df, 1928, 30)].copy()

    s = SWRsimulationCE.SWRsimulationCE({
        'simulation': {'n_ret_years': 30,
                       'n_assets': 2,
                       'trials': trials},
        'allocation': {},
        'withdrawal': {'fixed_pct': 0.0,
                       'variable_pct': 0.0,
                       'floor_pct': 0.0,
        },
        'evaluation': {},
    })

    z = s.simulate_trial(trial_generator(return_df, 1928, 30))
    assert len(z) == 30
    assert (z.index[0]) == 1928, "start year == 1928"
    assert (z.index[-1]) == 1957, "end year == 1957"


def test_fixed1():
    """zero returns, fixed spending, check starting, ending vals"""
    # zero returns, spend 2% per year, check ending value declines to 0.4

    RETURN = 0.0
    FIXED = 2.0
    VARIABLE = 0.0
    FLOOR = 0.0
    NYEARS = 30

    download_df = load_returns()
    return_df = download_df.iloc[:, [0, 2]].copy()
    return_df.columns = ['stocks', 'tbonds']
    return_df['stocks'] = RETURN
    return_df['tbonds'] = RETURN

    trials = [trial_generator(return_df, 1928, NYEARS)]

    s = SWRsimulationCE.SWRsimulationCE({
        'simulation': {'n_ret_years': NYEARS,
                       'n_assets': 2,
                       'trials': trials},
        'allocation': {},
        'withdrawal': {'fixed_pct': FIXED,
                       'variable_pct': VARIABLE,
                       'floor_pct': FLOOR},
        'evaluation': {},
    })

    z = s.simulate_trial(trial_generator(return_df, 1928, 30))

    assert (z['start_port'].iloc[0]) == 100, "start port value == 100"
    assert (z['end_port'].iloc[-1]) == 40, "ending port value == 40"


def test_variable1():
    """zero returns, variable spending, check starting, ending vals"""

    RETURN = 0.0
    FIXED = 0
    VARIABLE = 2.0
    FLOOR = 0.0
    NYEARS = 30

    download_df = load_returns()
    return_df = download_df.iloc[:, [0, 2]].copy()
    return_df.columns = ['stocks', 'tbonds']
    return_df['stocks'] = RETURN
    return_df['tbonds'] = RETURN

    trials = [trial_generator(return_df, 1928, NYEARS)]

    s = SWRsimulationCE.SWRsimulationCE({
        'simulation': {'n_ret_years': NYEARS,
                      'n_assets': 2,
                      'trials': trials},
        'allocation': {},
        'withdrawal': {'fixed_pct': FIXED,
                       'variable_pct': VARIABLE,
                       'floor_pct': FLOOR,
        },
        'evaluation': {},
    })

    print(s)

    z = s.simulate_trial(trial_generator(return_df, 1928, 30))

    assert (z['start_port'].iloc[0]) == 100, "start port value == 100"
    assert z['end_port'].iloc[-1] == pytest.approx(100 * ((1 - VARIABLE / 100) ** NYEARS), 0.000001)
    z


def test_fixed2():
    """fixed returns, fixed spending, check starting, ending vals"""

    # 4% real return, spend fixed 4% of starting, assert ending value unchanged
    RETURN = 0.04
    FIXED = 4
    VARIABLE = 0.0
    FLOOR = 0.0
    NYEARS = 30

    download_df = load_returns()
    return_df = download_df.iloc[:, [0, 2]].copy()
    return_df.columns = ['stocks', 'tbonds']
    return_df['stocks'] = RETURN
    return_df['tbonds'] = RETURN

    trials = [trial_generator(return_df, 1928, NYEARS)]

    s = SWRsimulationCE.SWRsimulationCE({
        'simulation': {'n_ret_years': 30,
                      'n_assets': 2,
                      'trials': trials},
        'allocation': {},
        'withdrawal': {'fixed_pct': FIXED,
                       'variable_pct': VARIABLE,
                       'floor_pct': FLOOR,
        },
        'evaluation': {},
    })

    z = s.simulate_trial(trial_generator(return_df, 1928, 30))

    assert (z['start_port'].iloc[0]) == 100, "start port value == 100"
    assert (z['end_port'].iloc[-1]) == 100, "end port value correct"


def test_variable2():
    """fixed returns, fixed spending, check starting, ending vals"""
    # return 0.02% variable spending 0.02/1.02, check final value unchanged
    RETURN = 0.02
    FIXED = 0.0
    FLOOR = 0.0
    VARIABLE = 0.02 / 1.02 * 100
    NYEARS = 30

    download_df = load_returns()
    return_df = download_df.iloc[:, [0, 2]].copy()
    return_df.columns = ['stocks', 'tbonds']
    return_df['stocks'] = RETURN
    return_df['tbonds'] = RETURN

    trials = [trial_generator(return_df, 1928, NYEARS)]

    s = SWRsimulationCE.SWRsimulationCE({
        'simulation': {'n_ret_years': 30,
                      'n_assets': 2,
                      'trials': trials},
        'allocation': {},
        'withdrawal': {'fixed_pct': FIXED,
                       'variable_pct': VARIABLE,
                       'floor_pct': FLOOR,
        },
        'evaluation': {},
    })

    z = s.simulate_trial(trial_generator(return_df, 1928, 30))

    assert (z['start_port'].iloc[0]) == 100, "start port value == 100"
    assert (z['end_port'].iloc[-1]) == 100, "end port value correct"


def test_bengen1():
    """matches Bengen values, modulo using real values throughout"""
    # per appendix of Bengen paper https://www.retailinvestor.org/pdf/Bengen1.pdf
    # nominal return 10% for stocks, 5% for bonds
    # inflation 3%
    # fixed spending of 4% of orig port
    STOCK_RETURN = (1.1 / 1.03) - 1
    BOND_RETURN = (1.05 / 1.03) - 1
    VARIABLE = 0.0
    FIXED = 4.0
    FLOOR = 0.0
    NYEARS = 30

    download_df = load_returns()
    return_df = download_df.iloc[:, [0, 2]].copy()
    return_df.columns = ['stocks', 'tbonds']
    return_df['stocks'] = STOCK_RETURN
    return_df['tbonds'] = BOND_RETURN

    trials = [trial_generator(return_df, 1928, NYEARS)]

    s = SWRsimulationCE.SWRsimulationCE({
        'simulation': {'n_ret_years': NYEARS,
                      'n_assets': 2,
                      'trials': trials},
        'allocation': {},
        'withdrawal': {'fixed_pct': FIXED,
                       'variable_pct': VARIABLE,
                       'floor_pct': FLOOR,
        },
        'evaluation': {},
    })

    z = s.simulate_trial(trial_generator(return_df, 1928, 30))
    # match figures in appendix
    # example uses nominal vals with 3% inflation, we use real vals
    assert z.iloc[0]['before_spend'] * 1.03 == pytest.approx(107.5, 0.000001)
    assert z.iloc[0]['spend'] * 1.03 == 4.12, "spend does not match Bengen"
    assert z.iloc[0]['end_port'] * 1.03 == pytest.approx(103.38, 0.000001), "ending port does not match Bengen"

def test_bengen2():
    """matches Bengen values, use historical return data"""
    FIXED = 4.0
    VARIABLE = 0.0
    FLOOR = 4.0
    NYEARS = 30

    download_df = load_returns()
    return_df = download_df.iloc[:, [0, 3, 12]]
    return_df.columns=['stocks', 'bonds', 'cpi']
    real_return_df = return_df.copy()
    real_return_df['stocks'] = (1 + real_return_df['stocks']) / (1 + real_return_df['cpi']) - 1
    real_return_df['bonds'] = (1 + real_return_df['bonds']) / (1 + real_return_df['cpi']) - 1
    real_return_df.drop('cpi', axis=1, inplace=True)
    
    s = SWRsimulationCE.SWRsimulationCE({
        'simulation': {'returns_df': real_return_df,
                       'n_ret_years': NYEARS,
        },
        'allocation': {},  # no args, default equal weight
        'withdrawal': {'fixed_pct': FIXED,
                       'variable_pct': VARIABLE,
                       'floor_pct': FLOOR},
        'evaluation': {'gamma': 0},
    })

    z = s.simulate()

    assert z[0]['trial'].iloc[0]['spend'] == 4.0, "bad value: cohort 0 year 0 spend"
    assert z[0]['trial'].iloc[0]['end_port'] == pytest.approx(120.955061, 0.000001), "bad value: cohort 0 year 0 end port"
    assert z[0]['trial'].iloc[-1]['spend'] == 4.0, "bad value: cohort 0 final year spend"
    assert z[0]['trial'].iloc[-1]['end_port'] == pytest.approx(189.255136, 0.000001), "bad value: cohort 0 final year end port"


def test_45():
    """4/5% using historical returns"""
    FIXED = -1
    VARIABLE = 5.0
    FLOOR = 4.0
    NYEARS = 30

    download_df = load_returns()
    return_df = download_df.iloc[:, [0, 3, 12]]
    return_df.columns=['stocks', 'bonds', 'cpi']
    real_return_df = return_df.copy()
    real_return_df['stocks'] = (1 + real_return_df['stocks']) / (1 + real_return_df['cpi']) - 1
    real_return_df['bonds'] = (1 + real_return_df['bonds']) / (1 + real_return_df['cpi']) - 1
    real_return_df.drop('cpi', axis=1, inplace=True)
    
    s = SWRsimulationCE.SWRsimulationCE({
        'simulation': {'returns_df': real_return_df,
                       'n_ret_years': NYEARS,
        },
        'allocation': {},  # no args, default equal weight
        'withdrawal': {'fixed_pct': FIXED,
                       'variable_pct': VARIABLE,
                       'floor_pct': FLOOR},
        'evaluation': {'gamma': 0},
    })

    z = s.simulate()

    assert z[0]['trial'].iloc[0]['spend'] == pytest.approx(5.247753, 0.000001), "bad value: cohort 0 year 0 spend"
    assert z[0]['trial'].iloc[0]['end_port'] == pytest.approx(119.707308, 0.000001), "bad value: cohort 0 year 0 end port"
    assert z[0]['trial'].iloc[-1]['spend'] == pytest.approx(5.690874, 0.000001), "bad value: cohort 0 final year spend"
    assert z[0]['trial'].iloc[-1]['end_port'] == pytest.approx(128.126609, 0.000001), "bad value: cohort 0 final year end port"


def test_high_gamma():
    """a gamma 16 rule using historical returns"""

    FIXED = 3.5
    VARIABLE = 1.1
    FLOOR = 3.8
    NYEARS = 30
    STOCK_PCT = 0.73
    BOND_PCT = 0.27

    download_df = load_returns()
    return_df = download_df.iloc[:, [0, 3, 12]]
    return_df.columns=['stocks', 'bonds', 'cpi']
    real_return_df = return_df.copy()
    real_return_df['stocks'] = (1 + real_return_df['stocks']) / (1 + real_return_df['cpi']) - 1
    real_return_df['bonds'] = (1 + real_return_df['bonds']) / (1 + real_return_df['cpi']) - 1
    real_return_df.drop('cpi', axis=1, inplace=True)
    
    s = SWRsimulationCE.SWRsimulationCE({
        'simulation': {'returns_df': real_return_df,
                       'n_ret_years': NYEARS,
        },
        'allocation': {'asset_weights': np.array([STOCK_PCT, BOND_PCT])},
        'withdrawal': {'fixed_pct': FIXED,
                       'variable_pct': VARIABLE,
                       'floor_pct': FLOOR},
        'evaluation': {'gamma': 0},
    })

    z = s.simulate()

    assert z[0]['trial'].iloc[0]['spend'] == pytest.approx(4.978399, 0.000001), "bad value: cohort 0 year 0 spend"
    assert z[0]['trial'].iloc[0]['end_port'] == pytest.approx(129.421552, 0.000001), "bad value: cohort 0 year 0 end port"
    assert z[0]['trial'].iloc[-1]['spend'] == pytest.approx(5.068669, 0.000001), "bad value: cohort 0 final year spend"
    assert z[0]['trial'].iloc[-1]['end_port'] == pytest.approx(137.537621, 0.000001), "bad value: cohort 0 final year end port"


def test_low_gamma():
    """a lower gamma rule using historical returns"""

    FIXED = 0.7
    VARIABLE = 5.8
    FLOOR = 3.4
    NYEARS = 30
    STOCK_PCT = 0.89
    BOND_PCT = 0.11

    download_df = load_returns()
    return_df = download_df.iloc[:, [0, 3, 12]]
    return_df.columns=['stocks', 'bonds', 'cpi']
    real_return_df = return_df.copy()
    real_return_df['stocks'] = (1 + real_return_df['stocks']) / (1 + real_return_df['cpi']) - 1
    real_return_df['bonds'] = (1 + real_return_df['bonds']) / (1 + real_return_df['cpi']) - 1
    real_return_df.drop('cpi', axis=1, inplace=True)
    
    s = SWRsimulationCE.SWRsimulationCE({
        'simulation': {'returns_df': real_return_df,
                       'n_ret_years': NYEARS,
        },
        'allocation': {'asset_weights': np.array([STOCK_PCT, BOND_PCT])},
        'withdrawal': {'fixed_pct': FIXED,
                       'variable_pct': VARIABLE,
                       'floor_pct': FLOOR},
        'evaluation': {'gamma': 0},
    })

    z = s.simulate()

    assert z[0]['trial'].iloc[0]['spend'] == pytest.approx(8.876278, 0.000001), "bad value: cohort 0 year 0 spend"
    assert z[0]['trial'].iloc[0]['end_port'] == pytest.approx(132.094032, 0.000001), "bad value: cohort 0 year 0 end port"
    assert z[0]['trial'].iloc[-1]['spend'] == pytest.approx(5.135414, 0.000001), "bad value: cohort 0 final year spend"
    assert z[0]['trial'].iloc[-1]['end_port'] == pytest.approx(71.337240, 0.000001), "bad value: cohort 0 final year end port"


print("running standalone")
