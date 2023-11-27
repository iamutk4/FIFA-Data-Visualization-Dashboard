# THIS IS A SAMPLE PYTHON SCRIPT TO UNDERSTAND STREAMLIT WORKING

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt


st.title("Lab 8. Task Analysis & Geospatial data")

st.write("CSE 5544")


# st.subheader("bootstrap alerts")
# st.success("Success")
# st.info("Information")
# st.warning("Warning")
# st.error("Error")

# st.header("Display data")


data_counties = pd.read_csv("oh-counties.csv")
# data

data_cases = pd.read_csv("osu-cases.csv")
# data

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("## Dashboard Section 1")

st.header("Line chart for OSU Covid-19 cases")

st.subheader("Drawing Line chart using Matplotlib")

dates = data_cases['date']
chart_cases = data_cases['case']


#render results
fig, ax = plt.subplots(figsize=(14, 6), dpi = 50)
ax.plot(dates, chart_cases)
ax.set_axisbelow(True)  #ensure the grid is under the graph elements
ax.margins(x=0.01) #set up the margin of graph
ax.grid(alpha = 0.3) #show the grid line
ax.set_xlabel('Dates')
ax.set_ylabel('Cases')
ax.set_title('OSU Covid-19 cases')
xaxis = plt.xticks(rotation=90, ha='center', fontsize=8)
yaxis = plt.yticks(fontsize=8)

st.pyplot(fig)

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("## Dashboard Section 2")

st.header("Heatmap for Ohio counties Covid-19 cases")

st.subheader("Drawing heatmap chart using Altair")

color_scale = alt.Scale(scheme='reds')

heatmap = alt.Chart(data_counties).mark_rect().encode(
    x=alt.X('county:N', title = 'county'),
    y=alt.Y('date:N',  title = 'date'),
    color=alt.Color('cases:Q', scale=color_scale, title='Number of Cases'),
    tooltip=['county', 'date', 'cases']
).properties(
    title='COVID-19 Cases Heatmap by County and Date'
)

# heatmap.show()
st.altair_chart(heatmap, use_container_width=True)

