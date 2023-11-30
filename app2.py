import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import numpy as np
import plotly.graph_objs as go  # This import statement defines 'go'
import streamlit as st
import base64
from plotly.offline import iplot
import seaborn as sns

st.set_page_config(page_title="Final Project", page_icon=":bar_chart:", layout="wide", initial_sidebar_state='collapsed')

matches  = pd.read_csv("data/WorldCupMatches.csv")
players  = pd.read_csv("data/WorldCupPlayers.csv")
cups     = pd.read_csv("data/WorldCups.csv")

home_goals = matches.groupby('Home Team Name')['Home Team Goals'].sum()
away_goals = matches.groupby('Away Team Name')['Away Team Goals'].sum()
total_goals_per_country = home_goals.add(away_goals, fill_value=0)

total_goals_by_year = cups[['Year', 'GoalsScored']].set_index('Year')['GoalsScored']

###### DATA CLEANING ##########

def clean_data(players, matches, world_cup):
    # Drop rows with missing 'Year' in matches
    matches.dropna(subset=['Year'], inplace=True)

    # Clean 'Home Team Name' in matches
    names = matches[matches['Home Team Name'].str.contains('rn">')]['Home Team Name'].value_counts()
    wrong = list(names.index)
    correct = [name.split('>')[1] for name in wrong]
    old_name = ['Germany FR', 'Maracan� - Est�dio Jornalista M�rio Filho', 'Estadio do Maracana']
    new_name = ['Germany', 'Maracan Stadium', 'Maracan Stadium']
    wrong = wrong + old_name
    correct = correct + new_name

    for index, wr in enumerate(wrong):
        matches = matches.replace(wrong[index], correct[index])

    # Clean 'Winner', 'Runners-Up', and 'Third' in world_cup
    names_to_clean = ['Winner', 'Runners-Up', 'Third']
    for col in names_to_clean:
        for index, wr in enumerate(wrong):
            world_cup = world_cup.replace(wrong[index], correct[index])

    return players, matches, world_cup

players, matches, cups = clean_data(players, matches, cups)

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
# Goal Analysis Functions

def plot_goals_per_country(data, top_n=20):
    # If top_n is provided, select the top_n countries with the most goals
    data = data.sort_values(ascending=False).head(top_n)
    
    fig, ax = plt.subplots(figsize=(40, 30))  # Increase figure width to 20 inches
    countries = list(data.index)
    goals = list(data.values)
    ax.barh(countries, goals, color='skyblue')  # Change to a horizontal bar chart
    plt.xlabel('Goals')
    plt.title('Top ' + str(top_n) + ' Countries by Goals Scored in World Cup')
    #plt.tight_layout()  # Adjust layout
    return fig


# Function to plot Total Goals Scored by Year (Bubble Graph)
def plot_total_goals_by_year(data):
    df = pd.DataFrame(list(data.items()), columns=['Year', 'Goals'])
    color_scale_reversed = px.colors.sequential.Viridis[::-1]
    fig = px.scatter(df, x='Year', y='Goals', size='Goals', color='Goals',
                     size_max=60, title='Total Goals Scored by Year',
                     color_continuous_scale=color_scale_reversed)
    return fig


def plot_attendance_over_years(data):
    fig = px.line(data, x='Year', y='Attendance', title='Attendance Over the Years',
                  labels={'Attendance': 'Total Attendance'})
    fig.update_traces(mode='lines+markers')
    return fig

def generate_top5_teams_goals_plot(matches_data):
    home = matches_data.groupby(['Year', 'Home Team Name'])['Home Team Goals'].sum()
    away = matches_data.groupby(['Year', 'Away Team Name'])['Away Team Goals'].sum()
    goals = pd.concat([home, away], axis=1)
    goals.fillna(0, inplace=True)
    goals['Goals'] = goals['Home Team Goals'] + goals['Away Team Goals']
    goals = goals.drop(labels=['Home Team Goals', 'Away Team Goals'], axis=1)
    goals = goals.reset_index()
    goals.columns = ['Year', 'Country', 'Goals']
    goals = goals.sort_values(by=['Year', 'Goals'], ascending=[True, False])
    top5 = goals.groupby('Year').head()

    data = []
    for team in top5['Country'].drop_duplicates().values:
        year = top5[top5['Country'] == team]['Year']
        goal = top5[top5['Country'] == team]['Goals']

        data.append(go.Bar(x=year, y=goal, name=team))

    layout = go.Layout(barmode='stack', title='Top 5 Teams with Most Goals every World Cup', showlegend=False)

    fig = go.Figure(data=data, layout=layout)
    
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

# UTK FUNCTIONS



def generate_goals_per_country_plot(matches_data):
    home = matches_data[['Home Team Name', 'Home Team Goals']].dropna()
    away = matches_data[['Away Team Name', 'Away Team Goals']].dropna()
    home.columns = ['Countries', 'Goals']
    away.columns = home.columns

    goals = pd.concat([home, away], ignore_index=True)
    goals = goals.groupby('Countries').sum()
    goals = goals.sort_values(by='Goals', ascending=False)

    fig = px.bar(
        goals[:20],
        x=goals.index[:20],
        y='Goals',
        labels={'Goals': 'Number of Goals'},
        title='Total Goals Scored by Country',
    )

    fig.update_layout(xaxis_title='Country Names', yaxis_title='Goals')

    return fig


def generate_attendance_per_year_plot(world_cup_data):
    world_cup_data['Attendance'] = world_cup_data['Attendance'].str.replace(".", "")
    
    fig = px.bar(
        world_cup_data,
        x='Year',
        y='Attendance',
        title='Attendance Per Year',
        labels={'Attendance': 'Attendance'},
    )
    fig.update_layout(xaxis_tickangle=80)

    return fig

def generate_qualified_teams_per_year_plot(world_cup_data):
    fig = px.bar(
        world_cup_data,
        x='Year',
        y='QualifiedTeams',
        title='Qualified Teams Per Year',
        labels={'QualifiedTeams': 'Qualified Teams'},
    )
    fig.update_layout(xaxis_tickangle=80)

    return fig

def generate_matches_played_per_year_plot(world_cup_data):
    fig = px.scatter(
        world_cup_data,
        x='Year',
        y='MatchesPlayed',
        size='MatchesPlayed',  # Use 'size' to represent MatchesPlayed as bubble size
        title='Matches Played by Teams Per Year',
        labels={'MatchesPlayed': 'Matches Played'},
    )
    fig.update_layout(xaxis_tickangle=80)

    return fig

def generate_top_attendance_matches_plot(matches_data):
    top10 = matches_data.sort_values(by='Attendance', ascending=False)[:10]
    top10['vs'] = top10['Home Team Name'] + " vs " + top10['Away Team Name']

    fig = px.bar(
        top10,
        y='vs',
        x='Attendance',
        orientation='h',
        title='Matches with the Highest Number of Attendance',
        labels={'Attendance': 'Attendance'},
    )

    fig.update_layout(
        yaxis=dict(title='Match Teams'),
        xaxis=dict(title='Attendance'),
    )

    # Add annotations for Stadium and Date inside the bars
    for i, (stadium, date, attendance) in enumerate(zip(top10['Stadium'], top10['Datetime'], top10['Attendance'])):
        # Extract only day, month, and year from the datetime
        date_str = pd.to_datetime(date).strftime('%d %b %Y')
        
        fig.add_annotation(
            x=attendance,
            y=i,
            text=f"Stadium: {stadium}, Date: {date_str}",
            font=dict(size=12, color='white'),
            showarrow=False,
            xanchor='right',  # Set anchor to the right for text inside the bars
            yanchor='auto',   # Align text to the center of the bar
            xshift=-50,         # Adjust xshift to control the position of the text
        )

    # Reverse the order of bars to have the max attendance at the top
    # fig.update_yaxes(categoryorder='total ascending')

    return fig

def plot_podium_count(world_cup_data):
    gold = world_cup_data["Winner"]
    silver = world_cup_data["Runners-Up"]
    bronze = world_cup_data["Third"]

    gold_count = pd.DataFrame.from_dict(gold.value_counts())
    silver_count = pd.DataFrame.from_dict(silver.value_counts())
    bronze_count = pd.DataFrame.from_dict(bronze.value_counts())

    podium_count = gold_count.join(silver_count, how='outer', lsuffix='_gold', rsuffix='_silver').join(bronze_count, how='outer', rsuffix='_bronze')
    podium_count = podium_count.fillna(0)
    podium_count.columns = ['WINNER', 'SECOND', 'THIRD']
    podium_count = podium_count.astype('int64')
    podium_count = podium_count.sort_values(by=['WINNER', 'SECOND', 'THIRD'], ascending=False)

    fig = px.bar(
        podium_count,
        y=['WINNER', 'SECOND', 'THIRD'],
        color_discrete_map={'WINNER': 'gold', 'SECOND': 'silver', 'THIRD': 'brown'},
        labels={'value': 'Number of Podiums'},
        title='Number of Podiums by Country',
    )

    fig.update_layout(
        xaxis=dict(title='Countries'),
        yaxis=dict(title='Podium Count'),
        barmode='stack',  # Stack bars on top of each other
    )

    return fig

def get_labels(matches):
    if matches['Home Team Goals'] > matches['Away Team Goals']:
        return 'Home Team Win'
    if matches['Home Team Goals'] < matches['Away Team Goals']:
        return 'Away Team Win'
    return 'DRAW'

def plot_match_outcomes(matches_data):
    matches_data['outcome'] = matches_data.apply(lambda x: get_labels(x), axis=1)
    outcome_counts = matches_data['outcome'].value_counts()

    # Create a Plotly Figure
    fig = go.Figure(data=[go.Pie(labels=outcome_counts.index, values=outcome_counts.values)])

    # Update layout if needed
    fig.update_layout(title='Match Outcomes by Home and Away Teams')

    return fig


############ Streamlit UI ##########


image_path = 'image1.jpeg'  

st.title("From Kick off to Glory : FIFA Worldcup")

analysis_type = st.selectbox("Choose Analysis Type", ["Goal Analysis", "Attendance and Participation Analysis", "Match Outcome Analysis", "Cup Analysis"])

if analysis_type == "Goal Analysis":
    col1, col2, col3 = st.columns(3)
    # with col1:
    #     st.subheader("Goal Analysis")
    #     # Goals per Country
    #     st.write("Goals per Country")
    #     bar_fig = plot_goals_per_country(total_goals_per_country)
    #     st.pyplot(bar_fig)
    with col1:
        # Cup Winning Count
        st.subheader("Goal Analysis")
        # st.write("Goals vs Countries")
        bar_fig = generate_goals_per_country_plot(matches)
        st.plotly_chart(bar_fig)

    with col2:
        # Total Goals Scored by Year
        st.subheader("")
        st.subheader("")
        # st.write("Total Goals Scored by Year")
        bubble_fig = plot_total_goals_by_year(total_goals_by_year)
        st.plotly_chart(bubble_fig)
    with col3:
        # Cup Winning Count
        st.subheader("")
        st.subheader("")
        # st.write("Goals vs Countries")
        bar_fig = generate_top5_teams_goals_plot(matches)
        st.plotly_chart(bar_fig)
    

if analysis_type == "Attendance and Participation Analysis":
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Attendance and Participation Analysis")
        # Total Attendance Over the Years
        # st.write("Attendance Over the Years")
        attendance_fig = plot_attendance_over_years(cups)
        st.plotly_chart(attendance_fig)
    with col2:
        st.subheader("")
        st.subheader("")
        # st.write("Attendance Analysis")
        teams_qualified_fig = generate_qualified_teams_per_year_plot(cups)
        st.plotly_chart(teams_qualified_fig)
    with col3:
        st.subheader("")
        st.subheader("")
        # st.write("Team Analysis")
        matches_per_year_fig = generate_matches_played_per_year_plot(cups)
        st.plotly_chart(matches_per_year_fig)
    
#     Average Attendance by World Cup
#     st.write("Average Attendance by World Cup")
#     cups['AverageAttendance'] = cups['Attendance'] / cups['MatchesPlayed']
#     average_attendance_fig = plot_average_attendance_by_worldcup(cups)
#     st.plotly_chart(average_attendance_fig)

if analysis_type == "Match Outcome Analysis":
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Match Outcome Analysis")
        # Plot and display Distribution of First and Second Half - Home Team Goals
        st.write("Distribution of First and Second Half - Home Team Goals")
        half_time_goals_fig = plot_half_time_goals(matches)
        st.plotly_chart(half_time_goals_fig)
    with col2:
        # Plot and display Home and Away Goals by Year
        st.write("Home and Away Goals by Year")
        goals_by_year_fig = plot_goals_by_year(matches)
        st.plotly_chart(goals_by_year_fig)

if analysis_type == "Cup Analysis":
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Cup Analysis")
        # Plot and display Distribution of First and Second Half - Home Team Goals
        # st.write("Distribution of First and Second Half - Home Team Goals")
        top_attendance_fig = generate_top_attendance_matches_plot(matches)
        st.plotly_chart(top_attendance_fig)
    with col2:
        # Plot and display Home and Away Goals by Year
        st.subheader("")
        st.subheader("")
        # st.write("Podium Counts per Country")
        podium_count_fig = plot_podium_count(cups)
        st.plotly_chart(podium_count_fig)
    with col3:
        # Plot and display Home and Away Win %
        st.subheader("")
        st.subheader("")
        # st.write("Podium Counts per Country")
        home_away_fig = plot_match_outcomes(matches)
        st.plotly_chart(home_away_fig)

#     # Plot and display Distribution of Home Team Goals
#     st.write("Distribution of Home Team Goals")
#     home_goals_fig = plot_goals_distribution(matches, 'Home Team Goals', 'Home Team Goals', 'blue')
#     st.plotly_chart(home_goals_fig)

#     # Plot and display Distribution of Away Team Goals
#     st.write("Distribution of Away Team Goals")
#     away_goals_fig = plot_goals_distribution(matches, 'Away Team Goals', 'Away Team Goals', 'red')
#     st.plotly_chart(away_goals_fig)