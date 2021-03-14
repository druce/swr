# swr
A framework for determining safe withdrawal rates, designed to accommodate:

- Any generator of historical asset returns (historical, Monte Carlo, roll your own market environment)
- Any asset allocation strategy (fixed weights, glidepath schedules, roll your own based on any parameters)
- Any withdrawal strategy (fixed withdrawal, variable percentage, combinations, glidepaths)
- Any metrics to evaluate retirement cohort outcomes (e.g. total spending, certainty equivalent spending, roll your own. Support for survival tables, i.e. calculate expected metric for living retirees taking into account retirement age and survivorship)
- Any (gradient-free) optimizer to find optimal parameters (e.g. asset allocation, withdrawal parameters) to maximize a metric in the given market environment

Example (see example.ipynb):

   - Market environment: historical returns 1928-2020; 2 assets, stocks and intermediate-term corporate bonds; analyze 30-year retirement cohorts.
   - Allocation rule: 1 parameter = `stock_alloc`. Allocate a fixed percentage to stocks. `bond_alloc` = 1 - `stock_alloc`.
   - Withdrawal rule: 3 parameters = `fixed_pct`, `variable_pct`, `floor_pct`. Withdrawal is a linear function of the portfolio value, with a floor. Withdraw `fixed_pct` (intercept), plus `variable_pct` (slope) * portfolio value, with a floor: `max(floor_pct, fixed_pct + variable_pct * portval / 100)`.
   - Metric to maximize: certainty-equivalent spending under CRRA utility with a gamma risk aversion parameter.
   - In this example, for each gamma value, we run optimizers to find the parameters (`stock_alloc`, `fixed_pct`, `variable_pct`, `floor_pct`) that would have maximized certainty-equivalent spending over all available 30-year retirement cohorts 1928-1991.

![optimal_by_gamma_table.png](optimal_by_gamma_table.png)

![outcome.png](outcome.png)

```python
N_RET_YEARS = 30
FIXED_PCT = 3.5
VARIABLE_PCT = 1.0
FLOOR_PCT = 0.0
ALLOC_STOCKS = 0.75
ALLOC_BONDS = 0.25
GAMMA  = 1.0

s = SWRsimulationCE({
    'simulation': {'returns_df': real_return_df,
                   'n_ret_years': N_RET_YEARS,
                  },
    'allocation': {'asset_weights': np.array([ALLOC_STOCKS, ALLOC_BONDS])}, 
    'withdrawal': {'fixed_pct': FIXED_PCT,
                   'variable_pct': VARIABLE_PCT,
                   'floor_pct': FLOOR_PCT,
                  },
    'evaluation': {'gamma': GAMMA},
    'visualization': {'histogram': True, 
                      'chart_1' : {'title': 'Years to Exhaustion by Retirement Year',
                                   'annotation': "Fixed spend %.1f, Variable spend %.1f, stocks %.1f%%" % (FIXED_PCT, 
                                                                                                           VARIABLE_PCT, 
                                                                                                           100 * ALLOC_STOCKS)
                                  },
                      'chart_2' : {'title': 'Spending By Retirement Year',
                                  },
                      'chart_3' : {'title': 'Portfolio Value By Retirement Year',
                                  },
                     }    
    
})

s.simulate()

s.visualize()

```

Work-in-progress, YMMV, reach out with any questions, suggestions...pull requests are welcome.

[React app that does interactive calculations and visualizations - see react/swr](http://www.streeteye.com/static/swr/)
