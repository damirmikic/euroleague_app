import streamlit as st
import pandas as pd
import google.generativeai as genai
import matplotlib.pyplot as plt
import os

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
    st.markdown("### Data Sources")
    pbp_file = st.file_uploader("Upload Play-by-Play Data", type=["csv", "parquet"])
    shots_file = st.file_uploader("Upload Shot Chart Data", type=["csv", "parquet"])

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
        st.error("Play-by-Play Data not loaded. Please upload 'euroleague_pbp_2024_2025.csv'.")
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
        st.warning("Shot Chart Data not loaded.")
    else:
        # Filters
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            seasons = df_shots['Season'].unique()
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
            # 1. Initialize Court (Vertical orientation for Portrait data)
            court = FIBACourt(rotation=90) 
            fig, ax = court.draw(figsize=(12, 12))
            
            # 2. Convert Units to Meters (cm -> m)
            # Euroleague data is cm. SportyPy uses meters.
            x_meters = df_chart['Coord_X'] / 100
            y_meters = df_chart['Coord_Y'] / 100
            
            # 3. Plotting Logic
            misses_mask = df_chart['Shot_Result'] == 'Miss'
            makes_mask = df_chart['Shot_Result'] == 'Make'
            
            # No coordinate rotation needed because we rotated the court to 90 degrees
            # to match the data's portrait orientation.
            
            # Plot Misses
            ax.scatter(x_meters[misses_mask], y_meters[misses_mask], 
                       c='red', alpha=0.5, s=40, label='Miss', edgecolors='white', linewidth=0.5, zorder=2)
            
            # Plot Makes
            ax.scatter(x_meters[makes_mask], y_meters[makes_mask], 
                       c='green', alpha=0.8, s=40, label='Make', edgecolors='white', linewidth=0.5, zorder=3)
            
            title_str = f"{selected_season} | {selected_team} | {selected_player}"
            ax.set_title(title_str, fontsize=16, y=1.02)
            ax.legend(loc='upper right')
            
            st.pyplot(fig)
        
        with st.expander("View Shot Data"):
            st.dataframe(df_chart)
