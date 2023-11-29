import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import numpy as np
import plotly.graph_objs as go  # This import statement defines 'go'
import streamlit as st
import base64


matches  = pd.read_csv("data/WorldCupMatches.csv")
players  = pd.read_csv("data/WorldCupPlayers.csv")
cups     = pd.read_csv("data/WorldCups.csv")

home_goals = matches.groupby('Home Team Name')['Home Team Goals'].sum()
away_goals = matches.groupby('Away Team Name')['Away Team Goals'].sum()
total_goals_per_country = home_goals.add(away_goals, fill_value=0)

total_goals_by_year = cups[['Year', 'GoalsScored']].set_index('Year')['GoalsScored']


######### CSS STyling #########
def get_base64_of_image(path):
    with open(path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return encoded_string

def add_bg_from_base64(base64_string):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{base64_string}");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


def set_title_color():
    st.markdown(
        """
        <style>
        .stApp h1 {
            color: white;
            font-weight: bold;

        }
        </style>
        """,
        unsafe_allow_html=True
    )
def add_fade_effect_only():
    st.markdown(
        """
        <style>
        .stApp::before {
            content: '';
            position: absolute;
            top: 0;
            right: 0;
            bottom: 0;
            left: 0;
            background-color: rgba(0, 0, 0, 0.5); /* Semi-transparent overlay */
            z-index: -1;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


    
    
#### Functions of the vis ####

def plot_goals_per_country(data, top_n=20):
    # If top_n is provided, select the top_n countries with the most goals
    data = data.sort_values(ascending=False).head(top_n)
    
    fig, ax = plt.subplots(figsize=(20, 10))  # Increase figure width to 20 inches
    countries = list(data.index)
    goals = list(data.values)
    ax.barh(countries, goals, color='skyblue')  # Change to a horizontal bar chart
    plt.xlabel('Goals')
    plt.title('Top ' + str(top_n) + ' Countries by Goals Scored in World Cup')
    plt.tight_layout()  # Adjust layout
    return fig


# Function to plot Total Goals Scored by Year (Bubble Graph)
def plot_total_goals_by_year(data):
    df = pd.DataFrame(list(data.items()), columns=['Year', 'Goals'])
    fig = px.scatter(df, x='Year', y='Goals', size='Goals', color='Goals',
                     size_max=60, title='Total Goals Scored by Year')
    return fig


def plot_attendance_over_years(data):
    fig = px.line(data, x='Year', y='Attendance', title='Attendance Over the Years',
                  labels={'Attendance': 'Total Attendance'})
    fig.update_traces(mode='lines+markers')
    return fig




#####3#####


def plot_half_time_goals(matches):
    if 'Half-time Home Goals' in matches.columns and 'Half-time Away Goals' in matches.columns:
        matches = matches.rename(columns={'Half-time Home Goals': "first half home goals",
                                          'Half-time Away Goals': "first half away goals"})
    else:
        raise ValueError("Required columns not found in the DataFrame")

    # Calculating second half goals
    if 'first half home goals' in matches.columns and 'Home Team Goals' in matches.columns:
        matches["second half home goals"] = matches["Home Team Goals"].subtract(matches["first half home goals"], fill_value=0)
    if 'first half away goals' in matches.columns and 'Away Team Goals' in matches.columns:
        matches["second half away goals"] = matches["Away Team Goals"].subtract(matches["first half away goals"], fill_value=0)




    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.arange(0, 9, 1),
        y=matches["first half home goals"],
        mode='lines',
        name='First Half Home Goals',
        line=dict(color='blue', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=np.arange(0, 9, 1),
        y=matches["second half home goals"],
        mode='lines',
        name='Second Half Home Goals',
        line=dict(color='red', width=2)
    ))

    fig.update_layout(title='Distribution of First and Second Half - Home Team Goals')
    return fig
def plot_goals_by_year(matches):
    # Ensure 'Year', 'Home Team Goals', and 'Away Team Goals' are in the correct format
    matches['Year'] = pd.to_numeric(matches['Year'], errors='coerce')
    matches['Home Team Goals'] = pd.to_numeric(matches['Home Team Goals'], errors='coerce')
    matches['Away Team Goals'] = pd.to_numeric(matches['Away Team Goals'], errors='coerce')

    # Create DataFrames for home and away goals
    gh = matches[["Year", "Home Team Goals"]].copy()
    gh.columns = ["year", "goals"]
    gh["type"] = "Home Team Goals"

    ga = matches[["Year", "Away Team Goals"]].copy()
    ga.columns = ["year", "goals"]
    ga["type"] = "Away Team Goals"

    # Concatenate and rename columns
    gls = pd.concat([gh, ga], axis=0)

    # Create the violin plot
    fig = px.violin(gls, x="year", y="goals", color="type", violinmode='overlay')
    fig.update_layout(title='Home and Away Goals by Year')
    return fig



# Function to plot Distribution of Goals
def plot_goals_distribution(matches, goal_type, title, color):
    fig = px.histogram(
        matches, 
        x=goal_type, 
        nbins=10, 
        title=title,
        marginal='rug', # can be 'rug', 'box', 'violin'
        color_discrete_sequence=[color] # specify color
    )
    return fig

############ Streamlit UI ##########


image_path = 'image1.jpeg'  

st.title("From Kick off to Glory : FIFA Worldcup")

analysis_type = st.selectbox("Choose Analysis Type", ["Goal Analysis", "Match and Attendance Analysis", "Match Outcome Analysis"])

if analysis_type == "Goal Analysis":

    st.subheader("Goal Analysis")
    # Goals per Country
    st.write("Goals per Country")
    bar_fig = plot_goals_per_country(total_goals_per_country)
    st.pyplot(bar_fig)
    
    # Total Goals Scored by Year
    st.write("Total Goals Scored by Year")
    bubble_fig = plot_total_goals_by_year(total_goals_by_year)
    st.plotly_chart(bubble_fig)
    

if analysis_type == "Match and Attendance Analysis":

    st.subheader("Match and Attendance Analysis")
    
    # Total Attendance Over the Years
    st.write("Attendance Over the Years")
    attendance_fig = plot_attendance_over_years(cups)
    st.plotly_chart(attendance_fig)
    
#     Average Attendance by World Cup
#     st.write("Average Attendance by World Cup")
#     cups['AverageAttendance'] = cups['Attendance'] / cups['MatchesPlayed']
#     average_attendance_fig = plot_average_attendance_by_worldcup(cups)
#     st.plotly_chart(average_attendance_fig)

if analysis_type == "Match Outcome Analysis":
    st.subheader("Match Outcome Analysis")

    # Plot and display Distribution of First and Second Half - Home Team Goals
    st.write("Distribution of First and Second Half - Home Team Goals")
    half_time_goals_fig = plot_half_time_goals(matches)
    st.plotly_chart(half_time_goals_fig)

    # Plot and display Home and Away Goals by Year
    st.write("Home and Away Goals by Year")
    goals_by_year_fig = plot_goals_by_year(matches)
    st.plotly_chart(goals_by_year_fig)

#     # Plot and display Distribution of Home Team Goals
#     st.write("Distribution of Home Team Goals")
#     home_goals_fig = plot_goals_distribution(matches, 'Home Team Goals', 'Home Team Goals', 'blue')
#     st.plotly_chart(home_goals_fig)

#     # Plot and display Distribution of Away Team Goals
#     st.write("Distribution of Away Team Goals")
#     away_goals_fig = plot_goals_distribution(matches, 'Away Team Goals', 'Away Team Goals', 'red')
#     st.plotly_chart(away_goals_fig)