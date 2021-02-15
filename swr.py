import numpy as np
import pandas as pd

import streamlit as st

from SWRsimulation import SWRsimulationCE

REAL_RETURN_FILE = 'real_return_df.pickle'
N_RET_YEARS = 30

# Create a text element and let the reader know the data is loading.
# TODO: loading data with pinwheel or other notification that goes away
# TODO: load file based on current cached params
# TODO: load remotely

real_return_df = pd.read_pickle(REAL_RETURN_FILE)

# Notify the reader that the data was successfully loaded.

st.title("A 'safe withdrawal rate' visualization")
stock_alloc = st.slider('Stock allocation', 0, 100, 75)
bond_alloc = 100 - stock_alloc
fixed_spending = st.slider('Fixed spending', 0.0, 10.0, 3.0, 0.1)
variable_spending = st.slider('Variable spending', 0.0, 10.0, 3.0, 0.1)

param_names = ["Stock allocation", "Bond allocation", "Fixed spending", "Variable spending"]
param_values = [stock_alloc, bond_alloc, fixed_spending, variable_spending]
param_table = pd.DataFrame(data=np.array([param_names, param_values]).T, columns=['Parameter', 'Value'])
st.write(param_table)

s = SWRsimulationCE.SWRsimulationCE({
    'simulation': {'returns_df': real_return_df,
                   'n_ret_years': N_RET_YEARS,
                                         },
    'allocation': {'asset_weights': np.array([stock_alloc/100, bond_alloc/100])}, # default is equal-weight
    'withdrawal': {'fixed_pct': fixed_spending,
                   'variable_pct': variable_spending},
    'evaluation': {'gamma': 1},
    'visualization': {'histogram': True,
                      'chart_1' : {'title': 'Years to Exhaustion by Retirement Year'},
                      'chart_2' : {'title': 'Portfolio Spending By Retirement Year'},
                      'chart_3' : {'title': 'Portfolio Value By Retirement Year'},
    }    # chart options etc.
})

s.simulate()
import matplotlib.pyplot as plt
fig, axs = plt.subplots(3, figsize=(20, 30))
s.chart_1_histogram(axs[0])
s.chart_2_lines(axs[1])
s.chart_3_lines(axs[2])

st.pyplot(fig)
          
st.write(s.table_metrics())




