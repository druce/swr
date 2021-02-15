import streamlit as st
import numpy as np
import pandas as pd

DATE_COLUMN = 'date/time'
DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
         'streamlit-demo-data/uber-raw-data-sep14.csv.gz')

@st.cache
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data

# Create a text element and let the reader know the data is loading.
# TODO: loading data with pinwheel or other notification that goes away
# Load 10,000 rows of data into the dataframe.
data = load_data(10000)
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

st.subheader('Number of pickups by hour')
hist_values = np.histogram(
    data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]
st.bar_chart(hist_values)

hour_to_filter = st.slider('hour', 0, 23, 17)  # min: 0h, max: 23h, default: 17h
filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]
st.subheader(f'Map of all pickups at {hour_to_filter}:00')
st.map(filtered_data)

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)

