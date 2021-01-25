# test_simulator.py

# pip install pytest
# pytest -q test_simulator.py
import pytest
import pandas as pd
from cers import cers
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


def test_zero():
    """no returns, no spending, just check shape"""
    download_df = load_returns()
    return_df = download_df.iloc[:, [0, 2]]
    return_df.columns = ['stocks', 'tbonds']
    zero_df = return_df.copy()
    zero_df['stocks'] = 0
    zero_df['tbonds'] = 0
    s = cers.Simulator(zero_df, 30, )
    z = s.simulate_trial(1928, [0.5, 0.5], 0.0, 0.0)
    assert(z.index[0]) == 1928, "start year == 1928"
    assert(z.index[-1]) == 1957, "end year == 1957"
    assert len(z) == 30, "length == 30"

def test_fixed():
    """zero returns, fixed spending, check starting, ending vals"""

    FIXED_RETURN = 0.02
    NYEARS = 30
    download_df = load_returns()
    return_df = download_df.iloc[:, [0, 2]]
    return_df.columns = ['stocks', 'tbonds']
    zero_df = return_df.copy()
    zero_df['stocks'] = 0
    zero_df['tbonds'] = 0
    s = cers.Simulator(zero_df, NYEARS, )
    z = s.simulate_trial(1928, [0.5, 0.5], FIXED_RETURN, 0.0)
    assert(z['starting'].iloc[0]) == 100, "start port value == 100"
    assert(z['ending'].iloc[-1]) == 100 - NYEARS * FIXED_RETURN * 100, "end port value == 40"


def test_variable():
    """zero returns, variable spending, check starting, ending vals"""

    VARIABLE_RETURN = 0.02
    NYEARS = 30
    download_df = load_returns()
    return_df = download_df.iloc[:, [0, 2]]
    return_df.columns = ['stocks', 'tbonds']
    zero_df = return_df.copy()
    zero_df['stocks'] = 0
    zero_df['tbonds'] = 0
    s = cers.Simulator(zero_df, NYEARS, )
    z = s.simulate_trial(1928, [0.5, 0.5], 0.0, VARIABLE_RETURN)
    assert(z['starting'].iloc[0]) == 100, "start port value == 100"
    assert(z['ending'].iloc[-1]) == 100 * ((1 - VARIABLE_RETURN) ** NYEARS), "end port value correct"


def test_fixed_spend():
    """fixed returns, fixed spending, check starting, ending vals"""

    RETURN = 0.02
    FIXED = 0.02
    NYEARS = 30
    download_df = load_returns()
    return_df = download_df.iloc[:, [0, 2]]
    return_df.columns = ['stocks', 'tbonds']
    zero_df = return_df.copy()
    zero_df['stocks'] = RETURN
    zero_df['tbonds'] = RETURN
    s = cers.Simulator(zero_df, NYEARS, )
    z = s.simulate_trial(1928, [0.5, 0.5], FIXED, 0)
    assert (z['starting'].iloc[0]) == 100, "start port value == 100"
    assert (z['ending'].iloc[-1]) == 100, "end port value correct"


def test_variable_spend():
    """fixed returns, fixed spending, check starting, ending vals"""

    RETURN = 0.02
    VARSPEND = 0.02 / 1.02
    NYEARS = 30
    download_df = load_returns()
    return_df = download_df.iloc[:, [0, 2]]
    return_df.columns = ['stocks', 'tbonds']
    zero_df = return_df.copy()
    zero_df['stocks'] = RETURN
    zero_df['tbonds'] = RETURN
    s = cers.Simulator(zero_df, NYEARS, )
    z = s.simulate_trial(1928, [0.5, 0.5], 0, VARSPEND)
    assert (z['starting'].iloc[0]) == 100, "start port value == 100"
    assert (z['ending'].iloc[-1]) == 100, "end port value correct"


def test_simple():
    """historical returns, 2% fixed and variable, check starting, ending vals"""

    # download_df = download_returns()
    download_df = load_returns()
    return_df = download_df.iloc[:, [0, 2]]
    return_df.columns = ['stocks', 'tbonds']
    s = cers.Simulator(return_df, 30, )
    z = s.simulate_trial(1928, [0.5, 0.5], 0.02, 0.02)

    assert z['starting'].iloc[0] == 100, "starting value == 100"
    assert z['ending'].iloc[-1] == 182.38655685622794, "ending value == 213.935"
    assert len(z) == 30, "length == 30"

def test_simple():
    """matches Bengen values, modulo using real values throughout"""

    # per appendix of Bengen paper https://www.retailinvestor.org/pdf/Bengen1.pdf
    # nominal return 10% for stocks, 5% for bonds
    # inflation 3%
    # fixed spending of 4% of orig port
    STOCK_RETURN = (1.1 / 1.03) -1
    BOND_RETURN = (1.05 / 1.03) -1
    VAR_SPEND = 0
    FIXED_SPEND = 0.04
    NYEARS = 30

    download_df = load_returns()
    return_df = download_df.iloc[:, [0, 2]].copy()
    return_df.columns = ['stocks', 'tbonds']
    return_df['stocks'] = STOCK_RETURN
    return_df['tbonds'] = BOND_RETURN
    s = cers.Simulator(return_df, NYEARS, )
    z = s.simulate_trial(1928, [0.5, 0.5], FIXED_SPEND, VAR_SPEND)
    assert z.iloc[0]['spend'] * 1.03 == 4.12, "spend does not match Bengen"
    assert z.iloc[0]['ending'] * 1.03 == pytest.approx(103.38, 0.01), "ending port does not match Bengen"
