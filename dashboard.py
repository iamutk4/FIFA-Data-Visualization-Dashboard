import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Function to generate different figures based on the selected option
def generate_figure(selected_option):
    if selected_option == 'Option 1':
        # Generate figure for Option 1
        data = {'x': np.arange(10), 'y': np.random.randn(10)}
        df = pd.DataFrame(data)
        fig, ax = plt.subplots()
        ax.plot(df['x'], df['y'])
        ax.set_title('Figure for Option 1')
    elif selected_option == 'Option 2':
        # Generate figure for Option 2
        data = {'x': np.arange(10), 'y': np.random.randn(10) + 2}
        df = pd.DataFrame(data)
        fig, ax = plt.subplots()
        ax.scatter(df['x'], df['y'])
        ax.set_title('Figure for Option 2')
    else:
        # Generate figure for Option 3
        data = {'category': ['A', 'B', 'C'], 'value': [3, 5, 2]}
        df = pd.DataFrame(data)
        fig, ax = plt.subplots()
        ax.bar(df['category'], df['value'])
        ax.set_title('Figure for Option 3')

    return fig

st.title('Streamlit Dashboard with Dropdown')

# Dropdown for selecting options
selected_option = st.selectbox('Select an option:', ['Option 1', 'Option 2', 'Option 3'])

# Display the selected figure
selected_figure = generate_figure(selected_option)
st.pyplot(selected_figure)
