import streamlit as st
import pandas as pd
import google.generativeai as genai
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arc
import os
import requests
import time
import json

# Try importing SportyPy, handle error if missing
try:
    from sportypy.surfaces.basketball import FIBACourt
    SPORTYPY_AVAILABLE = True
except ImportError:
    SPORTYPY_AVAILABLE = False

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Euroleague AI Analyst",
    page_icon="🏀",
    layout="wide"
)

# --- API CONFIGURATION ---
API_BASE_URL = "https://live.euroleague.net/api"
SEASONS_TO_FETCH = ["E2024", "E2025"]
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept": "application/json"
}

# --- SCRAPING FUNCTIONS ---
def fetch_pbp_game(season, game_code):
    """Fetches and parses PBP data for a single game."""
    url = f"{API_BASE_URL}/PlaybyPlay?gamecode={game_code}&seasoncode={season}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=5)
        if resp.status_code != 200: return None
        data = resp.json()
        if not data.get('TeamA'): return None # Game doesn't exist yet

        # Parse Quarters
        all_plays = []
        quarters = [("Q1", "FirstQuarter"), ("Q2", "SecondQuarter"), 
                    ("Q3", "ThirdQuarter"), ("Q4", "ForthQuarter"), ("OT", "ExtraTime")]
        
        base_info = {
            "Season": season, "GameCode": game_code,
            "TeamA": data.get('TeamA'), "TeamB": data.get('TeamB'),
            "CodeTeamA": data.get('CodeTeamA', '').strip(), 
            "CodeTeamB": data.get('CodeTeamB', '').strip()
        }

        for q_label, q_key in quarters:
            if q_key in data and data[q_key]:
                for play in data[q_key]:
                    row = base_info.copy()
                    row.update({
                        "Quarter": q_label,
                        "Minute": play.get("MINUTE"),
                        "Time": play.get("MARKERTIME"),
                        "TeamCode": play.get("CODETEAM", "").strip(),
                        "Player_ID": play.get("PLAYER_ID", "").strip(),
                        "Player": play.get("PLAYER"),
                        "PlayType": play.get("PLAYTYPE"),
                        "Points_A": play.get("POINTS_A"),
                        "Points_B": play.get("POINTS_B"),
                        "Info": play.get("PLAYINFO")
                    })
                    all_plays.append(row)
        return all_plays
    except:
        return None

def fetch_shot_game(season, game_code):
    """Fetches and parses Shot/Points data for a single game."""
    url = f"{API_BASE_URL}/Points?gamecode={game_code}&seasoncode={season}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=5)
        if resp.status_code != 200: return None
        data = resp.json()
        if not data.get('Rows'): return None

        all_shots = []
        for shot in data['Rows']:
            action_id = shot.get("ID_ACTION", "")
            
            # Logic
            shot_result = "Make" if action_id.endswith("M") else "Miss" if (action_id.endswith("A") or action_id.endswith("MISS")) else "Unknown"
            if "FT" in action_id: continue # Optional: Skip free throws for pure shot chart? Keeping for now.

            # Coords
            try: cx, cy = float(shot.get("COORD_X")), float(shot.get("COORD_Y"))
            except: cx, cy = None, None

            is_dunk = "Dunk" in shot.get("ACTION", "")
            is_half = (cy is not None and cy > 1242)

            row = {
                "Season": season, "GameCode": game_code,
                "Team": shot.get("TEAM", "").strip(),
                "Player": shot.get("PLAYER"),
                "Action_Type": action_id,
                "Shot_Result": shot_result,
                "Points": shot.get("POINTS"),
                "Coord_X": cx, "Coord_Y": cy,
                "Zone": shot.get("ZONE"),
                "Minute": shot.get("MINUTE"),
                "Is_Dunk": is_dunk, "Is_Half_Court": is_half
            }
            all_shots.append(row)
        return all_shots
    except:
        return None

def run_updater():
    """Main logic to update both datasets with robust overlapping."""
    
    # 1. Load existing data
    existing_pbp = pd.DataFrame()
    existing_shots = pd.DataFrame()
    
    if os.path.exists("euroleague_pbp_2024_2025.parquet"):
        existing_pbp = pd.read_parquet("euroleague_pbp_2024_2025.parquet")
    
    if os.path.exists("euroleague_shot_chart_extended.parquet"):
        existing_shots = pd.read_parquet("euroleague_shot_chart_extended.parquet")

    new_pbp_rows = []
    new_shot_rows = []
    
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    # Estimates for progress bar
    total_steps = 50 # Arbitrary buffer for new games
    step_count = 0

    with st.status("Downloading latest Euroleague data...", expanded=True) as status:
        
        for season in SEASONS_TO_FETCH:
            st.write(f"Checking Season {season}...")
            
            # Determine start game (Robust Overlap Strategy)
            start_game = 1
            if not existing_pbp.empty:
                season_pbp = existing_pbp[existing_pbp['Season'] == season]
                if not season_pbp.empty:
                    # Start from the LAST recorded game code. 
                    # This re-fetches the last game to ensure it's final/complete (handling live games).
                    start_game = int(season_pbp['GameCode'].max())
            
            current_game = start_game
            consecutive_errors = 0
            
            while True:
                status_text.text(f"Fetching {season} Game {current_game}...")
                
                # Fetch PBP
                pbp_data = fetch_pbp_game(season, current_game)
                # Fetch Shots
                shot_data = fetch_shot_game(season, current_game)
                
                # Stop condition: 3 consecutive empty/error responses
                if not pbp_data and not shot_data:
                    consecutive_errors += 1
                    if consecutive_errors >= 3:
                        st.write(f"Finished {season} at Game {current_game-3}")
                        break
                else:
                    consecutive_errors = 0
                    if pbp_data: new_pbp_rows.extend(pbp_data)
                    if shot_data: new_shot_rows.extend(shot_data)
                
                current_game += 1
                step_count += 1
                time.sleep(0.8) # Rate limit
                if step_count % 5 == 0:
                    progress_bar.progress(min(step_count / total_steps, 1.0))

        # --- SAVE LOGIC (ATOMIC REPLACEMENT) ---
        # We replace the entire game's data with the newly fetched version 
        # to handle corrections or live game updates safely.

        if new_pbp_rows:
            new_df = pd.DataFrame(new_pbp_rows)
            
            if not existing_pbp.empty:
                # Remove games that are being updated from the old dataset
                # Create a unique ID for efficient filtering
                new_df['unique_id'] = new_df['Season'] + "_" + new_df['GameCode'].astype(str)
                existing_pbp['unique_id'] = existing_pbp['Season'] + "_" + existing_pbp['GameCode'].astype(str)
                
                updated_ids = new_df['unique_id'].unique()
                existing_pbp = existing_pbp[~existing_pbp['unique_id'].isin(updated_ids)]
                
                # Drop helper cols
                new_df = new_df.drop(columns=['unique_id'])
                existing_pbp = existing_pbp.drop(columns=['unique_id'])

            final_pbp = pd.concat([existing_pbp, new_df], ignore_index=True)
            final_pbp.to_parquet("euroleague_pbp_2024_2025.parquet")
            st.success(f"Updated PBP: Processed {len(new_df)} play rows.")
        else:
            st.info("PBP Data is up to date.")

        if new_shot_rows:
            new_df = pd.DataFrame(new_shot_rows)
            
            if not existing_shots.empty:
                new_df['unique_id'] = new_df['Season'] + "_" + new_df['GameCode'].astype(str)
                existing_shots['unique_id'] = existing_shots['Season'] + "_" + existing_shots['GameCode'].astype(str)
                
                updated_ids = new_df['unique_id'].unique()
                existing_shots = existing_shots[~existing_shots['unique_id'].isin(updated_ids)]
                
                new_df = new_df.drop(columns=['unique_id'])
                existing_shots = existing_shots.drop(columns=['unique_id'])

            final_shots = pd.concat([existing_shots, new_df], ignore_index=True)
            final_shots.to_parquet("euroleague_shot_chart_extended.parquet")
            st.success(f"Updated Shots: Processed {len(new_df)} shot rows.")
        else:
            st.info("Shot Data is up to date.")
            
        status.update(label="Update Complete!", state="complete", expanded=False)
        time.sleep(1)
        st.rerun()

# --- SIDEBAR SETUP ---
with st.sidebar:
    st.title("🏀 Euroleague Analytics")
    st.markdown("---")
    
    # API Key Input
    api_key = st.text_input("Enter Gemini API Key", type="password")
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
        genai.configure(api_key=api_key)
        st.success("API Key accepted!")
    else:
        st.warning("Please enter your Gemini API Key to use the AI Analyst.")
    
    st.markdown("---")
    st.markdown("### 📡 Data Manager")
    st.info("Auto-fetch data from Euroleague API (2024 & 2025).")
    if st.button("🔄 Update Data Now"):
        run_updater()

    st.markdown("---")
    st.markdown("### Manual Upload (Optional)")
    pbp_file = st.file_uploader("Upload Play-by-Play", type=["csv", "parquet"])
    shots_file = st.file_uploader("Upload Shot Data", type=["csv", "parquet"])

# --- DATA LOADING FUNCTIONS ---
@st.cache_data
def load_data(pbp_path, shots_path):
    pbp_df = None
    shots_df = None
    
    # Helper: Load CSV or Parquet
    def read_file(file_path_or_buffer):
        if hasattr(file_path_or_buffer, 'name'):
            if file_path_or_buffer.name.endswith('.parquet'):
                return pd.read_parquet(file_path_or_buffer)
            return pd.read_csv(file_path_or_buffer)
        
        if isinstance(file_path_or_buffer, str):
            if file_path_or_buffer.endswith('.parquet'):
                return pd.read_parquet(file_path_or_buffer)
            return pd.read_csv(file_path_or_buffer)
            
        return None

    if pbp_path:
        pbp_df = read_file(pbp_path)
    elif os.path.exists("euroleague_pbp_2024_2025.parquet"):
        pbp_df = pd.read_parquet("euroleague_pbp_2024_2025.parquet")
    elif os.path.exists("euroleague_pbp_2024_2025.csv"):
        pbp_df = pd.read_csv("euroleague_pbp_2024_2025.csv")
        
    if shots_path:
        shots_df = read_file(shots_path)
    elif os.path.exists("euroleague_shot_chart_extended.parquet"):
        shots_df = pd.read_parquet("euroleague_shot_chart_extended.parquet")
    elif os.path.exists("euroleague_shot_chart_extended.csv"):
        shots_df = pd.read_csv("euroleague_shot_chart_extended.csv")
        
    return pbp_df, shots_df

df_pbp, df_shots = load_data(pbp_file, shots_file)

# --- HELPER: TIME TO SECONDS ---
def time_to_seconds(t_str):
    """Converts MM:SS string to seconds."""
    if pd.isna(t_str): return 0
    try:
        m, s = map(int, str(t_str).split(':'))
        return m * 60 + s
    except:
        return 0

# --- HELPER: CALCULATE MINUTES ---
def calculate_minutes_per_game(df):
    """
    Calculates minutes played per game/season based on IN/OUT markers.
    Returns a Series indexed by (Season, GameCode, TeamA, TeamB).
    """
    def process_game(group):
        total_seconds = 0
        for quarter, q_data in group.groupby('Quarter'):
            q_duration = 300 if quarter in ['OT', 'E1', 'E2'] else 600
            
            q_data = q_data.copy()
            q_data['Seconds_Left'] = q_data['Time'].apply(time_to_seconds)
            q_data = q_data.sort_values('Seconds_Left', ascending=False)
            
            subs = q_data[q_data['PlayType'].isin(['IN', 'OUT'])]
            
            is_on_court = False
            last_switch = q_duration
            
            if len(subs) == 0:
                if len(q_data) > 0:
                    is_on_court = True 
                else:
                    is_on_court = False
            else:
                first_sub = subs.iloc[0]
                if first_sub['PlayType'] == 'OUT':
                    is_on_court = True
                else:
                    is_on_court = False
            
            for _, row in subs.iterrows():
                t = row['Seconds_Left']
                if row['PlayType'] == 'OUT':
                    if is_on_court:
                        total_seconds += (last_switch - t)
                        is_on_court = False
                    last_switch = t
                elif row['PlayType'] == 'IN':
                    if not is_on_court:
                        last_switch = t
                        is_on_court = True
            
            if is_on_court:
                total_seconds += last_switch
                
        return round(total_seconds / 60.0, 2)

    return df.groupby(['Season', 'GameCode', 'TeamA', 'TeamB']).apply(process_game)

# --- HELPER: CALCULATE SUMMARY STATS ---
def calculate_summary(df, is_player_view=False):
    """Aggregates Play-by-Play data into a box score summary."""
    if df.empty:
        return pd.DataFrame()

    stats = df.copy()
    stats['Points_Made'] = 0
    stats.loc[stats['PlayType'] == 'FTM', 'Points_Made'] = 1
    stats.loc[stats['PlayType'] == '2FGM', 'Points_Made'] = 2
    stats.loc[stats['PlayType'] == '3FGM', 'Points_Made'] = 3

    group_cols = ['Season', 'GameCode', 'TeamA', 'TeamB']
    
    summary = stats.groupby(group_cols).agg(
        PTS=('Points_Made', 'sum'),
        P2M=('PlayType', lambda x: (x == '2FGM').sum()),
        P2A=('PlayType', lambda x: (x.isin(['2FGM', '2FGA'])).sum()),
        P3M=('PlayType', lambda x: (x == '3FGM').sum()),
        P3A=('PlayType', lambda x: (x.isin(['3FGM', '3FGA'])).sum()),
        FTM=('PlayType', lambda x: (x == 'FTM').sum()),
        FTA=('PlayType', lambda x: (x.isin(['FTM', 'FTA'])).sum()),
        OREB=('PlayType', lambda x: (x == 'O').sum()),
        DREB=('PlayType', lambda x: (x == 'D').sum()),
        REB=('PlayType', lambda x: (x.isin(['O', 'D'])).sum()),
        AST=('PlayType', lambda x: (x == 'AS').sum()),
        STL=('PlayType', lambda x: (x == 'ST').sum()),
        BLK=('PlayType', lambda x: (x == 'FV').sum()),
        TO=('PlayType', lambda x: (x == 'TO').sum()),
        PF=('PlayType', lambda x: (x.isin(['CM', 'OF', 'U', 'T'])).sum()),
        FD=('PlayType', lambda x: (x == 'RV').sum())
    ).reset_index()

    if is_player_view:
        mins = calculate_minutes_per_game(stats)
        mins.name = "MIN"
        summary = summary.set_index(group_cols).join(mins).reset_index()
    else:
        summary['MIN'] = 40.0

    summary['FGA'] = summary['P2A'] + summary['P3A']
    summary['POSS'] = (summary['FGA'] + 0.44 * summary['FTA'] + summary['TO'] - summary['OREB']).round(1)

    summary['2P%'] = (summary['P2M'] / summary['P2A'] * 100).fillna(0).round(1).astype(str) + '%'
    summary['3P%'] = (summary['P3M'] / summary['P3A'] * 100).fillna(0).round(1).astype(str) + '%'
    summary['FT%'] = (summary['FTM'] / summary['FTA'] * 100).fillna(0).round(1).astype(str) + '%'

    display_cols = [
        'Season', 'GameCode', 'TeamA', 'TeamB', 'MIN',
        'PTS', 'POSS', '2P%', '3P%', 'FT%', 
        'REB', 'OREB', 'DREB', 'AST', 'STL', 'BLK', 'TO', 'PF', 'FD',
        'P2M', 'P2A', 'P3M', 'P3A', 'FTM', 'FTA'
    ]
    
    final_cols = [c for c in display_cols if c in summary.columns]
    return summary[final_cols].sort_values(['Season', 'GameCode'], ascending=[False, False])

# --- TAB LAYOUT ---
tab1, tab2 = st.tabs(["🤖 AI Analyst", "🎯 Shot Charts"])

# --- TAB 1: AI ANALYST ---
with tab1:
    st.header("Ask the Data")
    
    if df_pbp is None:
        st.info("⚠️ No data loaded. Click 'Update Data Now' in the sidebar to fetch Euroleague data.")
    else:
        with st.container():
            st.subheader("📊 Data Filters")
            f_col1, f_col2, f_col3, f_col4 = st.columns(4)
            
            with f_col1:
                seasons = sorted(df_pbp['Season'].unique())
                selected_season = st.selectbox("1. Season", ["All"] + list(seasons), key="pbp_season")
            
            df_filtered = df_pbp.copy()
            if selected_season != "All":
                df_filtered = df_filtered[df_filtered['Season'] == selected_season]

            with f_col2:
                teams = sorted(df_filtered['TeamCode'].dropna().unique())
                selected_team = st.selectbox("2. Team", ["All"] + teams, key="pbp_team")
            
            if selected_team != "All":
                df_filtered = df_filtered[df_filtered['TeamCode'] == selected_team]

            with f_col3:
                players = sorted(df_filtered['Player'].dropna().unique())
                selected_player = st.selectbox("3. Player", ["All"] + players, key="pbp_player")
            
            if selected_player != "All":
                df_filtered = df_filtered[df_filtered['Player'] == selected_player]

            with f_col4:
                if not df_filtered.empty:
                    games_df = df_filtered[['GameCode', 'TeamA', 'TeamB']].drop_duplicates().sort_values('GameCode')
                    games_df['Label'] = games_df['GameCode'].astype(str) + " - " + games_df['TeamA'] + " vs " + games_df['TeamB']
                    game_options = ["All"] + list(games_df['Label'])
                else:
                    game_options = ["All"]
                
                selected_game = st.selectbox("4. Game", game_options, key="pbp_game")

            if selected_game != "All":
                game_code = int(selected_game.split(' - ')[0])
                df_filtered = df_filtered[df_filtered['GameCode'] == game_code]

            if not df_filtered.empty:
                is_player_view = selected_player != "All"
                summary_df = calculate_summary(df_filtered, is_player_view)
                
                st.markdown("#### 📈 Match Summary")
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
                
                if len(summary_df) > 1:
                    st.markdown("#### 📊 Statistical Aggregates (Filtered Matches)")
                    agg_cols = ['MIN', 'PTS', 'POSS', 'REB', 'AST', 'STL', 'BLK', 'TO', 'PF', 'FD']
                    agg_cols = [c for c in agg_cols if c in summary_df.columns]
                    
                    desc = summary_df[agg_cols].describe().T
                    desc = desc[['mean', '50%', 'std', 'min', 'max']]
                    desc.columns = ['Average', 'Median', 'Std Dev', 'Min', 'Max']
                    
                    st.dataframe(desc.style.format("{:.2f}"), use_container_width=True)
                    
            else:
                st.warning("No data matches the current filters.")

            with st.expander(f"View Raw Play-by-Play Rows ({len(df_filtered)})", expanded=False):
                st.dataframe(df_filtered, use_container_width=True)

        st.markdown("---")
        user_query = st.text_area("Enter your question about this data:", height=100, placeholder="e.g., How many points did he score in the 4th quarter?")
        
        if st.button("Analyze"):
            if not api_key:
                st.error("Please provide a Gemini API Key in the sidebar.")
            elif df_filtered.empty:
                st.error("The filtered dataset is empty.")
            else:
                with st.spinner("Gemini is thinking..."):
                    try:
                        buffer_info = df_filtered.head(1).to_markdown(index=False)
                        columns_info = list(df_filtered.columns)
                        
                        prompt = f"""
                        You are a Python Data Analyst assistant. 
                        You have a Pandas DataFrame named `df` loaded with Euroleague basketball data.
                        IMPORTANT: The user has already filtered this DataFrame. It only contains data relevant to their selection.
                        
                        Here are the columns: {columns_info}
                        Here is a sample row:
                        {buffer_info}
                        
                        Write a Python script to answer this question: "{user_query}"
                        
                        Rules:
                        1. Assume `df` is already loaded (it is the filtered dataframe).
                        2. Store the final answer in a variable named `result`.
                        3. `result` can be a number, a string, or a pandas DataFrame.
                        4. Do NOT use `print()`.
                        5. Return ONLY the Python code, no markdown formatting (no ```python).
                        6. Handle potential division by zero.
                        7. Use case-insensitive string comparison for names.
                        """
                        
                        model = genai.GenerativeModel('gemini-2.0-flash-exp')
                        response = model.generate_content(prompt)
                        generated_code = response.text.replace("```python", "").replace("```", "").strip()
                        
                        local_vars = {"df": df_filtered, "pd": pd}
                        exec(generated_code, {}, local_vars)
                        result = local_vars.get("result")
                        
                        st.subheader("Result:")
                        if isinstance(result, pd.DataFrame):
                            st.dataframe(result, use_container_width=True)
                        else:
                            st.info(str(result))
                            
                        with st.expander("View Generated Code"):
                            st.code(generated_code, language='python')
                            
                    except Exception as e:
                        st.error(f"An error occurred: {e}")

# --- TAB 2: SHOT CHARTS ---
with tab2:
    st.header("Interactive Shot Charts")
    
    if df_shots is None:
        st.info("⚠️ No shot data loaded. Click 'Update Data Now' in the sidebar.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            seasons = sorted(df_shots['Season'].unique())
            selected_season = st.selectbox("Season", seasons, index=len(seasons)-1, key="sc_season")
        
        df_chart = df_shots[df_shots['Season'] == selected_season]
        
        with col2:
            teams = sorted(df_chart['Team'].unique())
            selected_team = st.selectbox("Team", ["All"] + teams, key="sc_team")
            
        if selected_team != "All":
            df_chart = df_chart[df_chart['Team'] == selected_team]
            
        with col3:
            players = sorted(df_chart['Player'].dropna().unique())
            selected_player = st.selectbox("Player", ["All"] + players, key="sc_player")
            
        if selected_player != "All":
            df_chart = df_chart[df_chart['Player'] == selected_player]
            
        with col4:
            games = sorted(df_chart['GameCode'].unique())
            selected_game = st.selectbox("Game Code", ["All"] + [str(g) for g in games], key="sc_game")

        if selected_game != "All":
            df_chart = df_chart[df_chart['GameCode'] == int(selected_game)]

        total_shots = len(df_chart)
        made_shots = len(df_chart[df_chart['Shot_Result'] == 'Make'])
        percentage = (made_shots / total_shots * 100) if total_shots > 0 else 0
        
        st.metric(label="Field Goal %", value=f"{percentage:.1f}%", delta=f"{made_shots}/{total_shots}")

        # --- DRAW COURT USING SPORTYPY ---
        if not SPORTYPY_AVAILABLE:
            st.error("SportyPy library not found. Please add 'sportypy' to requirements.txt.")
        else:
            fig, ax = plt.subplots(figsize=(12, 12))
            
            court = FIBACourt(rotation=90) 
            court.draw(ax=ax)
            
            x_meters = df_chart['Coord_X'] / 100
            y_meters = df_chart['Coord_Y'] / 100
            
            misses_mask = df_chart['Shot_Result'] == 'Miss'
            makes_mask = df_chart['Shot_Result'] == 'Make'
            
            ax.scatter(x_meters[misses_mask], y_meters[misses_mask], 
                       c='red', alpha=0.5, s=40, label='Miss', edgecolors='white', linewidth=0.5, zorder=2)
            
            ax.scatter(x_meters[makes_mask], y_meters[makes_mask], 
                       c='green', alpha=0.8, s=40, label='Make', edgecolors='white', linewidth=0.5, zorder=3)
            
            title_str = f"{selected_season} | {selected_team} | {selected_player}"
            ax.set_title(title_str, fontsize=16, y=1.02)
            ax.legend(loc='upper right')
            
            st.pyplot(fig)
        
        with st.expander("View Shot Data"):
            st.dataframe(df_chart)
