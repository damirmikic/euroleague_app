import streamlit as st
import pandas as pd
import google.generativeai as genai
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arc
import os

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
        # Check if it's a Streamlit UploadedFile (has .name attribute)
        if hasattr(file_path_or_buffer, 'name'):
            if file_path_or_buffer.name.endswith('.parquet'):
                return pd.read_parquet(file_path_or_buffer)
            return pd.read_csv(file_path_or_buffer)
        
        # Check if it's a string path (local file)
        if isinstance(file_path_or_buffer, str):
            if file_path_or_buffer.endswith('.parquet'):
                return pd.read_parquet(file_path_or_buffer)
            return pd.read_csv(file_path_or_buffer)
            
        return None

    # Try loading PBP (Check uploaded -> local parquet -> local csv)
    if pbp_path:
        pbp_df = read_file(pbp_path)
    elif os.path.exists("euroleague_pbp_2024_2025.parquet"):
        pbp_df = pd.read_parquet("euroleague_pbp_2024_2025.parquet")
    elif os.path.exists("euroleague_pbp_2024_2025.csv"):
        pbp_df = pd.read_csv("euroleague_pbp_2024_2025.csv")
        
    # Try loading Shots (Check uploaded -> local parquet -> local csv)
    if shots_path:
        shots_df = read_file(shots_path)
    elif os.path.exists("euroleague_shot_chart_extended.parquet"):
        shots_df = pd.read_parquet("euroleague_shot_chart_extended.parquet")
    elif os.path.exists("euroleague_shot_chart_extended.csv"):
        shots_df = pd.read_csv("euroleague_shot_chart_extended.csv")
        
    return pbp_df, shots_df

# Load data (either from upload or local default)
df_pbp, df_shots = load_data(pbp_file, shots_file)

# --- HELPER: DRAW COURT ---
def draw_court(ax=None, color='black', lw=2):
    """
    Draws a FIBA Basketball Court.
    Units: cm.
    Hoop center: (0, 0).
    """
    if ax is None:
        ax = plt.gca()

    # The Hoop
    hoop = Circle((0, 0), radius=45/2, linewidth=lw, color=color, fill=False)

    # Backboard
    backboard = Rectangle((-90, -120), 180, 1, linewidth=lw, color=color)

    # The Paint (Key)
    # Width: 490cm, Height (from baseline): 580cm
    # Baseline is at y = -157.5 (1.575m behind hoop)
    # Hoop is at 0,0. 
    # So Paint Top is at 580 - 157.5 = 422.5
    outer_box = Rectangle((-245, -157.5), 490, 580, linewidth=lw, color=color, fill=False)
    
    # Restricted Area (Semi-circle)
    restricted = Arc((0, 0), 250, 250, theta1=0, theta2=180, linewidth=lw, color=color)

    # Free Throw Circle
    top_free_throw = Arc((0, 422.5), 360, 360, theta1=0, theta2=180, linewidth=lw, color=color)
    bottom_free_throw = Arc((0, 422.5), 360, 360, theta1=180, theta2=360, linewidth=lw, color=color, linestyle='dashed')

    # 3-Point Line
    # Straight lines: 6.60m from baseline, 0.9m from sideline? 
    # FIBA 3pt is 6.75m (675cm) radius from hoop.
    # Corner straight lines are 299cm from center X (calculated)
    corner_three_a = Rectangle((-750, -157.5), 0, 300, linewidth=lw, color=color) # Just a line logic
    three_arc = Arc((0, 0), 675*2, 675*2, theta1=22, theta2=158, linewidth=lw, color=color)
    
    # Center Circle (Half court is at Y = 1400 - 157.5 = 1242.5)
    center_outer_arc = Arc((0, 1242.5), 360, 360, theta1=180, theta2=360, linewidth=lw, color=color)
    center_inner_arc = Arc((0, 1242.5), 120, 120, theta1=180, theta2=360, linewidth=lw, color=color)

    # Court Boundaries
    court = Rectangle((-750, -157.5), 1500, 1400, linewidth=lw, color=color, fill=False)

    ax.add_patch(court)
    ax.add_patch(hoop)
    ax.add_patch(backboard)
    ax.add_patch(outer_box)
    ax.add_patch(restricted)
    ax.add_patch(top_free_throw)
    ax.add_patch(bottom_free_throw)
    ax.add_patch(three_arc)
    ax.add_patch(center_outer_arc)
    ax.add_patch(center_inner_arc)

    # Set limits
    ax.set_xlim(-800, 800)
    ax.set_ylim(-200, 1300)
    ax.set_aspect('equal')
    ax.axis('off')
    return ax

# --- TAB LAYOUT ---
tab1, tab2 = st.tabs(["🤖 AI Analyst", "🎯 Shot Charts"])

# --- TAB 1: AI ANALYST ---
with tab1:
    st.header("Ask the Data")
    st.write("Examples: *'Who has the most blocks this season?'*, *'Calculate the 3-point percentage for Panathinaikos in Q4'*")
    
    if df_pbp is None:
        st.error("Play-by-Play Data not loaded. Please upload 'euroleague_pbp_2024_2025.csv'.")
    else:
        # --- FILTERS SECTION ---
        with st.expander("📊 Data Filters (Optional)", expanded=True):
            f_col1, f_col2, f_col3, f_col4 = st.columns(4)
            
            # Initialize filtered df
            df_filtered = df_pbp.copy()

            # 1. Season Filter
            with f_col1:
                seasons = sorted(df_filtered['Season'].unique())
                selected_season_pbp = st.selectbox("Season", ["All"] + list(seasons), key="pbp_season")
            
            if selected_season_pbp != "All":
                df_filtered = df_filtered[df_filtered['Season'] == selected_season_pbp]

            # 2. Game Filter
            with f_col2:
                # Create readable labels for games
                if not df_filtered.empty:
                    games_df = df_filtered[['GameCode', 'TeamA', 'TeamB']].drop_duplicates().sort_values('GameCode')
                    games_df['Label'] = games_df['GameCode'].astype(str) + " - " + games_df['TeamA'] + " vs " + games_df['TeamB']
                    game_options = ["All"] + list(games_df['Label'])
                else:
                    game_options = ["All"]
                
                selected_game_pbp = st.selectbox("Game", game_options, key="pbp_game")

            if selected_game_pbp != "All":
                game_code = int(selected_game_pbp.split(' - ')[0])
                df_filtered = df_filtered[df_filtered['GameCode'] == game_code]

            # 3. Team Filter
            with f_col3:
                # Use TeamCode for filtering as it represents the action performer
                teams = sorted(df_filtered['TeamCode'].dropna().unique())
                selected_team_pbp = st.selectbox("Team (Action)", ["All"] + teams, key="pbp_team")
            
            if selected_team_pbp != "All":
                df_filtered = df_filtered[df_filtered['TeamCode'] == selected_team_pbp]

            # 4. Player Filter
            with f_col4:
                players = sorted(df_filtered['Player'].dropna().unique())
                selected_player_pbp = st.selectbox("Player", ["All"] + players, key="pbp_player")
            
            if selected_player_pbp != "All":
                df_filtered = df_filtered[df_filtered['Player'] == selected_player_pbp]

            st.caption(f"Analyzing **{len(df_filtered)}** rows of data.")

        # --- QUERY SECTION ---
        user_query = st.text_area("Enter your question:", height=100)
        
        if st.button("Analyze"):
            if not api_key:
                st.error("Please provide a Gemini API Key in the sidebar.")
            elif df_filtered.empty:
                st.error("The filtered dataset is empty. Please adjust your filters.")
            else:
                with st.spinner("Gemini is thinking..."):
                    try:
                        # 1. Construct Prompt with Filtered Context
                        buffer_info = df_filtered.head(1).to_markdown(index=False)
                        columns_info = list(df_filtered.columns)
                        
                        prompt = f"""
                        You are a Python Data Analyst assistant. 
                        You have a Pandas DataFrame named `df` loaded with Euroleague basketball data.
                        IMPORTANT: The user has already filtered this DataFrame. It only contains data relevant to their selection (Season/Game/Team/Player).
                        
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
                        6. Handle potential division by zero if calculating percentages.
                        7. Use case-insensitive string comparison for names if needed.
                        """
                        
                        # 2. Get Code from Gemini
                        model = genai.GenerativeModel('gemini-2.0-flash-exp')
                        response = model.generate_content(prompt)
                        generated_code = response.text.replace("```python", "").replace("```", "").strip()
                        
                        # 3. Execute Code safely using the FILTERED dataframe
                        local_vars = {"df": df_filtered, "pd": pd}
                        exec(generated_code, {}, local_vars)
                        
                        # 4. Display Result
                        result = local_vars.get("result")
                        
                        st.subheader("Result:")
                        if isinstance(result, pd.DataFrame):
                            st.dataframe(result)
                        else:
                            st.info(str(result))
                            
                        with st.expander("View Generated Code"):
                            st.code(generated_code, language='python')
                            
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                        st.warning("Tip: If the error says a column is missing, check your filters or the data source.")

# --- TAB 2: SHOT CHARTS ---
with tab2:
    st.header("Interactive Shot Charts")
    
    if df_shots is None:
        st.warning("Shot Chart Data not loaded. Please upload 'euroleague_shot_chart_extended.csv' or run the parser script.")
    else:
        # Filters
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            seasons = df_shots['Season'].unique()
            selected_season = st.selectbox("Season", seasons, index=len(seasons)-1)
            
        # Filter df by season first to update other lists
        df_season = df_shots[df_shots['Season'] == selected_season]
        
        with col2:
            teams = sorted(df_season['Team'].unique())
            selected_team = st.selectbox("Team", ["All"] + teams)
            
        with col3:
            if selected_team != "All":
                players = sorted(df_season[df_season['Team'] == selected_team]['Player'].dropna().unique())
            else:
                players = sorted(df_season['Player'].dropna().unique())
            selected_player = st.selectbox("Player", ["All"] + players)
            
        with col4:
            # Optional Game Filter
            games = sorted(df_season['GameCode'].unique())
            selected_game = st.selectbox("Game Code (Optional)", ["All"] + [str(g) for g in games])

        # Apply Filters
        filtered_shots = df_season.copy()
        if selected_team != "All":
            filtered_shots = filtered_shots[filtered_shots['Team'] == selected_team]
        if selected_player != "All":
            filtered_shots = filtered_shots[filtered_shots['Player'] == selected_player]
        if selected_game != "All":
            filtered_shots = filtered_shots[filtered_shots['GameCode'] == int(selected_game)]

        # Metrics
        total_shots = len(filtered_shots)
        made_shots = len(filtered_shots[filtered_shots['Shot_Result'] == 'Make'])
        percentage = (made_shots / total_shots * 100) if total_shots > 0 else 0
        
        st.metric(label="Field Goal %", value=f"{percentage:.1f}%", delta=f"{made_shots}/{total_shots}")

        # Plotting
        fig, ax = plt.subplots(figsize=(12, 11))
        draw_court(ax, color="black")
        
        # Plot Misses first (so Makes are on top)
        misses = filtered_shots[filtered_shots['Shot_Result'] == 'Miss']
        makes = filtered_shots[filtered_shots['Shot_Result'] == 'Make']
        
        ax.scatter(misses['Coord_X'], misses['Coord_Y'], c='red', alpha=0.5, s=30, label='Miss', edgecolors='white', linewidth=0.5)
        ax.scatter(makes['Coord_X'], makes['Coord_Y'], c='green', alpha=0.8, s=30, label='Make', edgecolors='white', linewidth=0.5)
        
        # Title and Legend
        title_str = f"{selected_season} | {selected_team} | {selected_player}"
        plt.title(title_str, fontsize=16)
        plt.legend(loc='upper right')
        
        st.pyplot(fig)
        
        # Show Data Table
        with st.expander("View Shot Data"):
            st.dataframe(filtered_shots)
