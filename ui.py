import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime, timedelta
from unidecode import unidecode
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from selenium import webdriver
from selenium.webdriver.edge.service import Service

# Functions from your code
def validate_player_details(url, expected_team, expected_jersey):
    response = requests.get(url)
    if response.status_code != 200:
        st.error(f"Failed to fetch the page. Status code: {response.status_code}")
        return False

    soup = BeautifulSoup(response.content, 'html.parser')
    team_tag = soup.select_one('#meta strong:contains("Team") + a')
    team_name = team_tag.text.strip() if team_tag else None

    svg_tags = soup.find_all('svg', class_='jersey')
    jersey_number = None
    if svg_tags:
        last_svg = svg_tags[-1]
        text_tags = last_svg.find_all('text')
        jersey_number = ''.join([tag.text.strip() for tag in text_tags])

    return team_name == expected_team and jersey_number == expected_jersey

def extract_defensive_rating(team):
    url = 'https://www.basketball-reference.com/teams/' + team + '/2025.html'
    response = requests.get(url)
    if response.status_code != 200:
        st.error(f"Failed to fetch the page. Status code: {response.status_code}")
        return None

    soup = BeautifulSoup(response.content, 'html.parser')
    stats_section = soup.find('div', id='info')
    if not stats_section:
        st.error("Could not find the stats section.")
        return None

    for strong_tag in stats_section.find_all('strong'):
        if "Def Rtg" in strong_tag.text:
            def_rtg_value = strong_tag.next_sibling.strip()
            return def_rtg_value.split()[1]

    st.error("Defensive Rating not found on the page.")
    return None

def fetch_team_stats(team_code):
    url = "https://www.basketball-reference.com/teams/" + team_code + "/2025.html"
    s = Service(r"C:\edgedriver_win64\msedgedriver.exe")
    driver = webdriver.Edge(service=s)
    try:
        driver.get(url)
        if driver.title == "404 Not Found":
            pass
        else:
            page_source = driver.page_source
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
    df = pd.read_html(page_source, header=0, attrs={'id': 'team_and_opponent'})[0]
    driver.close()
    team_row = df[df["Unnamed: 0"].str.contains("Opponent/G", na=False)]
    
    if team_row.empty:
        st.error(f"No Opponent/G stats found for {team_code}")
        return None
    
    stats = {
        "Team_FG%": float(team_row["FG%"].values[0]),
        "Team_3P%": float(team_row["3P%"].values[0]),
        "Team_TRB": float(team_row["TRB"].values[0]),
        "Team_AST": float(team_row["AST"].values[0]),
        "Team_STL": float(team_row["STL"].values[0]),
        "Team_BLK": float(team_row["BLK"].values[0])
    }
    return stats

def convert_mp_to_minutes(mp):
    minutes, seconds = map(int, mp.split(':'))
    return minutes + seconds / 60

# Streamlit UI
st.title("NBA Statline Predictor")

teams = {
    'Atlanta Hawks': 'ATL', 'Boston Celtics': 'BOS', 'Brooklyn Nets': 'BRK',
    'Charlotte Hornets': 'CHO', 'Chicago Bulls': 'CHI', 'Cleveland Cavaliers': 'CLE',
    'Dallas Mavericks': 'DAL', 'Denver Nuggets': 'DEN', 'Detroit Pistons': 'DET',
    'Golden State Warriors': 'GSW', 'Houston Rockets': 'HOU', 'Indiana Pacers': 'IND',
    'Los Angeles Clippers': 'LAC', 'Los Angeles Lakers': 'LAL', 'Memphis Grizzlies': 'MEM',
    'Miami Heat': 'MIA', 'Milwaukee Bucks': 'MIL', 'Minnesota Timberwolves': 'MIN',
    'New Orleans Pelicans': 'NOP', 'New York Knicks': 'NYK', 'Oklahoma City Thunder': 'OKC',
    'Orlando Magic': 'ORL', 'Philadelphia 76ers': 'PHI', 'Phoenix Suns': 'PHO',
    'Portland Trail Blazers': 'POR', 'Sacramento Kings': 'SAC', 'San Antonio Spurs': 'SAS',
    'Toronto Raptors': 'TOR', 'Utah Jazz': 'UTA', 'Washington Wizards': 'WAS'
}

# Fetch upcoming games
today = datetime.now()
months_dict = {
    1: 'january', 2: 'february', 3: 'march', 4: 'april', 5: 'may', 6: 'june',
    7: 'july', 8: 'august', 9: 'september', 10: 'october', 11: 'november', 12: 'december'
}
month = months_dict[today.month]
url = 'https://www.basketball-reference.com/leagues/NBA_2025_games-' + month + '.html'

st.header("Upcoming Games")
schedule_df = pd.read_html(url, header=0, attrs={'id': 'schedule'})[0]
schedule_df = schedule_df[schedule_df['PTS'] != 'PTS']
schedule_df.drop(columns=['Unnamed: 6', 'Unnamed: 7', 'Notes'], inplace=True)
schedule_df['Date'] = pd.to_datetime(schedule_df['Date'])

current_date = datetime.now().date()
next_day = current_date + timedelta(days=1)
upcoming_games = schedule_df[schedule_df['Date'] == pd.Timestamp(current_date)]
if upcoming_games['PTS'].isnull().sum() == 0:
    upcoming_games = schedule_df[schedule_df['Date'] == pd.Timestamp(next_day)]
upcoming_games.drop(columns=['PTS','PTS.1','Attend.','LOG'],inplace=True)
upcoming_games['Date'] = upcoming_games['Date'].apply(lambda x: x.date() if pd.notnull(x) else x)

game_no=1
for _, game in upcoming_games.iterrows():
    home_team = game['Home/Neutral']
    away_team = game['Visitor/Neutral']
    start_time = game['Start (ET)']
    
    home_logo="https://cdn.ssref.net/req/202411271/tlogo/bbr/"+teams[home_team]+"-2025.png"
    away_logo="https://cdn.ssref.net/req/202411271/tlogo/bbr/"+teams[away_team]+"-2025.png"
    # Custom Game Display
    st.markdown(f"""
    <div style="text-align: center; font-family: Arial, sans-serif; margin-bottom: 20px; padding: 10px; border: 1px solid black; border-radius: 10px; background-color: white;">
        <div style="display: flex; justify-content: space-between; font-size: 18px; font-weight: bold;">
            <div style="text-align: center; flex: 1;text-color:black">
                    <span style="margin-right: 15px; font-size: 20px; font-weight: bold;">{game_no}.</span>
                    <img src={home_logo} alt="{home_team} Logo" style="width: 70px; height: 70px;"><br>
                    <span style="margin-left: 30px">{home_team}</spam>
            </div>
            <div style="text-align: center; flex: 1;text-color: black;margin-top: 24px">vs</div>
            <div style="text-align: center; flex: 1;text-color: black">
                    <img src={away_logo} alt="{away_team} Logo" style="width: 70px; height: 70px;"><br>
                    {away_team}
            </div>
        </div>
        <div style="margin-top: 5px; font-size: 14px; color: gray;">{start_time}</div>
    </div>
    """, unsafe_allow_html=True)
    game_no+=1

# Select a game
game = st.number_input('Enter game number:', min_value=1, max_value=len(upcoming_games), step=1)
if game:
    selected_game = upcoming_games.iloc[game - 1]
    st.write("Selected Game:")
    st.write(f"Home Side: {selected_game['Home/Neutral']} | Road Side: {selected_game['Visitor/Neutral']}")
    
    HorR = st.radio("Select Home or Road side:", ['H', 'R'])
    if HorR == 'H':
        teamname = selected_game['Home/Neutral']
        opponent = selected_game['Visitor/Neutral']
    elif HorR == 'R':
        teamname = selected_game['Visitor/Neutral']
        opponent = selected_game['Home/Neutral']
    
    st.write(f"Selected Team: {teamname}")
    st.write(f"Opponent Team: {opponent}")

team = teams.get(teamname)
if team:
    url = f'https://www.basketball-reference.com/teams/{team}/2025.html'
    roster_df = pd.read_html(url, header=0, attrs={'id': 'roster'})[0]
    roster_df.drop(columns=['Birth'],inplace=True)
    st.header("Roster:")
    st.write(roster_df)

player = st.number_input('Enter player index number:', min_value=0, max_value=len(roster_df), step=1)
if st.button('Predict Stats'):
    
    injuries_df=pd.DataFrame()
    url='https://www.basketball-reference.com/teams/'+team+'/2025.html'
    s = Service(r"C:\edgedriver_win64\msedgedriver.exe")
    driver=webdriver.Edge(service=s)
    driver.get(url)
    page_source = driver.page_source  # Get dynamically rendered HTML
    injuries_df = pd.read_html(page_source, header=0, attrs={'id': 'injuries'})[0]
    driver.close()
    name=roster_df.iloc[player]['Player']
    if name in list(injuries_df['Player'].values):
        st.error("Player listed in injury report")
    else:
        l=name.split()
        num=1
        name=l[1][:5]+l[0][:2]+'01'
        for i in range(len(name)):
            if name[i]=='ö':
                name=name[:i]+'o'+name[i+1:]
        url='https://www.basketball-reference.com/players/t/'+name+'/gamelog/2025'
        jerseyno=str(int(roster_df.iloc[player]['No.']))
        while(validate_player_details(url, teamname, jerseyno)==False):
            num+=1
            if num<10:
                name=l[1][:5]+l[0][:2]+'0'+str(num)
            else:
                name=l[1][:5]+l[0][:2]+str(num)
            url=url='https://www.basketball-reference.com/players/t/'+name+'/gamelog/2025'

        st.markdown(f'<h1 style="text-align:center;">{roster_df.iloc[player]["Player"]}</h1>', unsafe_allow_html=True)
        img='https://www.basketball-reference.com/req/202106291/images/headshots/'+name.lower()+'.jpg'
        st.markdown(f"""
            <div style="text-align: center; font-family: Arial, sans-serif; margin-bottom: 20px; padding: 10px; background-color: white;">
                <div style="display: flex; justify-content: space-between; font-size: 18px; font-weight: bold;">
                    <div style="text-align: center; flex: 1;text-color:black">
                            <img src="{img}" alt="{name}" style="width: 200px; height: 200px;"><br>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<h4>Pos: ' + roster_df.iloc[player]['Pos'] + '</h4>', unsafe_allow_html=True)
        st.write('<h4>Height: '+roster_df.iloc[player]['Ht']+'</h4>',unsafe_allow_html=True)

        st.markdown('<h2>Season Average</h2>', unsafe_allow_html=True)
        player_split_df=pd.DataFrame()
        url1='https://www.basketball-reference.com/players/c/'+name+'/splits/2025'
        player_split_df=pd.read_html(url1,header=1,attrs={'id':'splits'})[0]
        season_avg_row=player_split_df.iloc[0]
        name=roster_df.iloc[player]['Player']
        data = {
            "Minutes": [season_avg_row['MP.1']],
            "Pts": [season_avg_row['PTS.1']],
            "Reb": [season_avg_row['TRB.1']],
            "Ast": [season_avg_row['AST.1']],
            "Stl": [f"{float(season_avg_row['STL']) / float(season_avg_row['G']):.2f}"],
            "Blk": [f"{float(season_avg_row['BLK']) / float(season_avg_row['G']):.2f}"],
            "FG%": [season_avg_row['FG%']],
            "3P%": [season_avg_row['3P%']]
        }

        # Convert the data into a DataFrame
        table = pd.DataFrame(data)

        # Display the table in Streamlit
        st.table(table)

        player_gamelog=pd.read_html(url,header=0,attrs={'id':'pgl_basic'})[0]
        player_gamelog.drop(columns=['Rk','G','Age','Unnamed: 5'],inplace=True)
        player_gamelog.rename(columns={'Tm':'Team','Unnamed: 7':'Result'},inplace=True)
        player_gamelog['Opp DefRtg'] = player_gamelog['Opp'].apply(lambda opp: extract_defensive_rating(opp))

        tgl_df=pd.DataFrame()
        url='https://www.basketball-reference.com/teams/'+team+'/2025/gamelog/'
        tgl_df=pd.read_html(url,header=1,attrs={'id':'tgl_basic'})[0]
        tgl_df=tgl_df[['FG%','3P%','TRB','AST','STL','BLK']]
        tgl_df.rename(columns={'FG%':'Team_FG%','3P%':'Team_3P%','TRB':'Team_TRB','AST':'Team_AST','STL':'Team_STL','BLK':'Team_BLK'},inplace=True)
        tgl_df = tgl_df[(tgl_df['Team_FG%'] != 'Team') & (tgl_df['Team_FG%'] != 'FG%')].reset_index(drop=True)
        player_gamelog=player_gamelog[player_gamelog['GS']!='GS'].reset_index(drop=True)
        player_gamelog = pd.concat([player_gamelog, tgl_df], axis=1)

        player_gamelog = player_gamelog[(player_gamelog['PTS'] != 'Inactive') & (player_gamelog['PTS'] != 'Did Not Play') & (player_gamelog['MP'] != 'Did Not Dress')]
        st.markdown('<h2>Player Gamelog</h2>', unsafe_allow_html=True)
        st.write(player_gamelog)

        player_gamelog['MP'] = player_gamelog['MP'].apply(convert_mp_to_minutes)

        # Load the dataset
        df = player_gamelog  # Replace with your dataset path

        # Data Cleaning
        df = df.dropna(subset=['TRB', 'AST', 'FG', '3P', 'STL', 'BLK'])  # Remove rows with missing values
        df = df[df['PTS'] != 'Inactive']  # Filter out non-active games

        # Convert columns to numeric where applicable
        df['TRB'] = pd.to_numeric(df['TRB'])
        df['AST'] = pd.to_numeric(df['AST'])
        df['FG'] = pd.to_numeric(df['FG'])
        df['3P'] = pd.to_numeric(df['3P'])
        df['STL'] = pd.to_numeric(df['STL'])
        df['BLK'] = pd.to_numeric(df['BLK'])

        # Feature and Target Selection
        features = ['Opp DefRtg','MP','Team_FG%','Team_3P%','Team_TRB','Team_AST','Team_STL','Team_BLK']
        targets = ['PTS','TRB', 'AST', 'FG', '3P', 'STL', 'BLK']  # Stat line to predict

        X = df[features]
        y = df[targets]

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a Model
        model = RandomForestRegressor(random_state=42,max_depth= None,min_samples_leaf= 4,min_samples_split= 10,n_estimators= 200)
        model.fit(X_train, y_train)

        # Evaluate the Model
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        #st.write(f"Mean Absolute Error: {mae}")

        stats=fetch_team_stats(teams[opponent])
        new_data = pd.DataFrame({
        'Opp DefRtg': [extract_defensive_rating(teams[opponent])],
        'MP': [df.tail(5)['MP'].mean()],
        'Team_FG%': [stats['Team_FG%']],
        'Team_3P%': [stats['Team_3P%']],
        'Team_TRB': [stats['Team_TRB']],
        'Team_AST': [stats['Team_AST']],
        'Team_STL': [stats['Team_STL']],
        'Team_BLK': [stats['Team_BLK']]
        }) 
        predicted_stats = model.predict(new_data)
        predicted_stats = predicted_stats.flatten()  # Flatten the array to 1D
        st.markdown('<h2>Predicted Statline</h2>', unsafe_allow_html=True)
        prediction={
            "Pts":round(predicted_stats[0]),
            "Reb":round(predicted_stats[1]),
            "Ast":round(predicted_stats[2]),
            "FGM":round(predicted_stats[3]),
            "3PM":round(predicted_stats[4]),
            "Stl":round(predicted_stats[5]),
            "Blk":round(predicted_stats[6])
        }
        predicted_df=pd.DataFrame(prediction,index=[0])
        st.table(predicted_df)




