# streamlist app
# streamlit run swr.py

import numpy as np
import pandas as pd

import streamlit as st

import matplotlib.pyplot as plt

from SWRsimulation.SWRsimulationCE import SWRsimulationCE

REAL_RETURN_FILE = 'real_return_df.pickle'
N_RET_YEARS = 30

# Create a text element and let the reader know the data is loading.
# TODO: loading data with pinwheel or other notification that goes away

real_return_df = pd.read_pickle(REAL_RETURN_FILE)

# Notify the reader that the data was successfully loaded.

'''
# Safe Withdrawal for Retirement Calculator

[See blog post](https://druce.ai/2021/02/optimal-safe-withdrawal-for-retirement-using-certainty-equivalent-spending-revisited)
'''

defaults = st.selectbox("Choose defaults",
                        [
                            { "name": "Bengen 4% rule",
                              "stock_alloc": 50.0,
                              "fixed_pct": 4.0,
                              "variable_pct": 0.0,
                              "floor_pct": 4.0,
                            },
                            {"name": "A 'safe' rule",
                             "stock_alloc": 75.2,
                             "fixed_pct": 3.54,
                             "variable_pct": 1.06,
                             "floor_pct": 3.54
                            },
                            {"name": "A 'risky' rule",
                             "stock_alloc": 87.94,
                             "fixed_pct": 2.68,
                             "variable_pct": 2.96,
                             "floor_pct": 2.68
                            },
                        ],
                        format_func=lambda option:option["name"]
)

stock_alloc = st.slider('Stock allocation', min_value=0.0, max_value=100.0, value=defaults['stock_alloc'])
bond_alloc = 100 - stock_alloc
st.write(f'Bond allocation: {bond_alloc:.2f}%')

fixed_pct = st.slider('Fixed %', min_value=-2.0, max_value=6.0, value=defaults['fixed_pct'])
variable_pct = st.slider('Variable %', min_value=0.0, max_value=10.0, value=defaults['variable_pct'])
floor_pct = st.slider('Floor %', min_value=0.0, max_value=6.0, value=defaults['floor_pct'])

param_names = ["Stock %", "Bond %", "Fixed %", "Variable %", "Floor %"]
param_values = [stock_alloc, bond_alloc, fixed_pct, variable_pct, floor_pct]
param_table = pd.DataFrame(data=np.array([param_names, param_values]).T, columns=['Parameter', 'Value'])
param_table['Parameter'] = param_table['Parameter'].astype(str)
param_table['Value'] = param_table['Value'].astype(float)
format_dict = {"Value": "{:.2f}"}
st.write(param_table.style.format(format_dict))

s = SWRsimulationCE({
    'simulation': {'returns_df': real_return_df,
                   'n_ret_years': N_RET_YEARS,
                  },
    'allocation': {'asset_weights': np.array([stock_alloc / 100, bond_alloc / 100])},
    'withdrawal': {'fixed_pct': fixed_pct,
                   'variable_pct': variable_pct,
                   'floor_pct': floor_pct,
                  },
    'evaluation': {'gamma': 1.0},
    'visualization': {'histogram': True, 
                      'chart_1' : {'title': 'Years to Exhaustion by Retirement Year',
                                   'annotation': "Fixed spend %.1f, Variable spend %.1f, stocks %.1f%%" % (fixed_pct, 
                                                                                                           variable_pct, 
                                                                                                           100 * stock_alloc)
                      },
                      'chart_2' : {'title': 'Spending By Retirement Year',
                      },
                      'chart_3' : {'title': 'Portfolio Value By Retirement Year',
                      },
    }    
})

s.simulate()

fig, axs = plt.subplots(3, figsize=(20, 30))
s.chart_1_histogram(axs[0])
s.chart_2_lines(axs[1])
s.chart_3_lines(axs[2])

st.pyplot(fig)
df = s.table_metrics()
format_dict = {"value": "{:.2f}"}
st.write(df.style.format(format_dict))

