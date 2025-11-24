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

    # Try loading PBP
    if pbp_path:
        pbp_df = read_file(pbp_path)
    elif os.path.exists("euroleague_pbp_2024_2025.parquet"):
        pbp_df = pd.read_parquet("euroleague_pbp_2024_2025.parquet")
    elif os.path.exists("euroleague_pbp_2024_2025.csv"):
        pbp_df = pd.read_csv("euroleague_pbp_2024_2025.csv")
        
    # Try loading Shots
    if shots_path:
        shots_df = read_file(shots_path)
    elif os.path.exists("euroleague_shot_chart_extended.parquet"):
        shots_df = pd.read_parquet("euroleague_shot_chart_extended.parquet")
    elif os.path.exists("euroleague_shot_chart_extended.csv"):
        shots_df = pd.read_csv("euroleague_shot_chart_extended.csv")
        
    return pbp_df, shots_df

# Load data
df_pbp, df_shots = load_data(pbp_file, shots_file)

# --- HELPER: DRAW COURT ---
def draw_court(ax=None, color='black', lw=2):
    if ax is None:
        ax = plt.gca()
    hoop = Circle((0, 0), radius=45/2, linewidth=lw, color=color, fill=False)
    backboard = Rectangle((-90, -120), 180, 1, linewidth=lw, color=color)
    outer_box = Rectangle((-245, -157.5), 490, 580, linewidth=lw, color=color, fill=False)
    restricted = Arc((0, 0), 250, 250, theta1=0, theta2=180, linewidth=lw, color=color)
    top_free_throw = Arc((0, 422.5), 360, 360, theta1=0, theta2=180, linewidth=lw, color=color)
    bottom_free_throw = Arc((0, 422.5), 360, 360, theta1=180, theta2=360, linewidth=lw, color=color, linestyle='dashed')
    three_arc = Arc((0, 0), 675*2, 675*2, theta1=22, theta2=158, linewidth=lw, color=color)
    center_outer_arc = Arc((0, 1242.5), 360, 360, theta1=180, theta2=360, linewidth=lw, color=color)
    center_inner_arc = Arc((0, 1242.5), 120, 120, theta1=180, theta2=360, linewidth=lw, color=color)
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
    
    if df_pbp is None:
        st.error("Play-by-Play Data not loaded. Please upload 'euroleague_pbp_2024_2025.csv'.")
    else:
        # --- CASCADING FILTERS ---
        with st.container():
            st.subheader("📊 Data Filters")
            f_col1, f_col2, f_col3, f_col4 = st.columns(4)
            
            # 1. Season Filter
            with f_col1:
                seasons = sorted(df_pbp['Season'].unique())
                selected_season = st.selectbox("1. Season", ["All"] + list(seasons), key="pbp_season")
            
            # Filter data by Season
            df_filtered = df_pbp.copy()
            if selected_season != "All":
                df_filtered = df_filtered[df_filtered['Season'] == selected_season]

            # 2. Team Filter (Dependent on Season)
            with f_col2:
                teams = sorted(df_filtered['TeamCode'].dropna().unique())
                selected_team = st.selectbox("2. Team", ["All"] + teams, key="pbp_team")
            
            # Filter data by Team
            if selected_team != "All":
                df_filtered = df_filtered[df_filtered['TeamCode'] == selected_team]

            # 3. Player Filter (Dependent on Season + Team)
            with f_col3:
                players = sorted(df_filtered['Player'].dropna().unique())
                selected_player = st.selectbox("3. Player", ["All"] + players, key="pbp_player")
            
            # Filter data by Player
            if selected_player != "All":
                df_filtered = df_filtered[df_filtered['Player'] == selected_player]

            # 4. Game Filter (Dependent on Season + Team + Player)
            with f_col4:
                if not df_filtered.empty:
                    # Create readable labels: "15 - Real Madrid vs Barcelona"
                    games_df = df_filtered[['GameCode', 'TeamA', 'TeamB']].drop_duplicates().sort_values('GameCode')
                    games_df['Label'] = games_df['GameCode'].astype(str) + " - " + games_df['TeamA'] + " vs " + games_df['TeamB']
                    game_options = ["All"] + list(games_df['Label'])
                else:
                    game_options = ["All"]
                
                selected_game = st.selectbox("4. Game", game_options, key="pbp_game")

            # Filter data by Game
            if selected_game != "All":
                game_code = int(selected_game.split(' - ')[0])
                df_filtered = df_filtered[df_filtered['GameCode'] == game_code]

            # --- DISPLAY FILTERED DATA ---
            with st.expander(f"View Filtered Data ({len(df_filtered)} rows)", expanded=False):
                st.dataframe(df_filtered, use_container_width=True)

        # --- QUERY SECTION ---
        st.markdown("---")
        user_query = st.text_area("Enter your question about the filtered data:", height=100, placeholder="e.g., How many points were scored in the 4th quarter?")
        
        if st.button("Analyze"):
            if not api_key:
                st.error("Please provide a Gemini API Key in the sidebar.")
            elif df_filtered.empty:
                st.error("The filtered dataset is empty. Please adjust your filters.")
            else:
                with st.spinner("Gemini is thinking..."):
                    try:
                        # Construct Prompt
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
                        
                        # Get Code
                        model = genai.GenerativeModel('gemini-2.0-flash-exp')
                        response = model.generate_content(prompt)
                        generated_code = response.text.replace("```python", "").replace("```", "").strip()
                        
                        # Execute Code
                        local_vars = {"df": df_filtered, "pd": pd}
                        exec(generated_code, {}, local_vars)
                        
                        # Display Result
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
                        st.warning("Tip: If the error says a column is missing, check your filters or the data source.")

# --- TAB 2: SHOT CHARTS ---
with tab2:
    st.header("Interactive Shot Charts")
    
    if df_shots is None:
        st.warning("Shot Chart Data not loaded. Please upload 'euroleague_shot_chart_extended.csv' or run the parser script.")
    else:
        # Filters
        col1, col2, col3, col4 = st.columns(4)
        
        # 1. Season
        with col1:
            seasons = df_shots['Season'].unique()
            selected_season = st.selectbox("Season", seasons, index=len(seasons)-1, key="sc_season")
        
        df_chart = df_shots[df_shots['Season'] == selected_season]
        
        # 2. Team
        with col2:
            teams = sorted(df_chart['Team'].unique())
            selected_team = st.selectbox("Team", ["All"] + teams, key="sc_team")
            
        if selected_team != "All":
            df_chart = df_chart[df_chart['Team'] == selected_team]
            
        # 3. Player
        with col3:
            players = sorted(df_chart['Player'].dropna().unique())
            selected_player = st.selectbox("Player", ["All"] + players, key="sc_player")
            
        if selected_player != "All":
            df_chart = df_chart[df_chart['Player'] == selected_player]
            
        # 4. Game
        with col4:
            games = sorted(df_chart['GameCode'].unique())
            selected_game = st.selectbox("Game Code", ["All"] + [str(g) for g in games], key="sc_game")

        if selected_game != "All":
            df_chart = df_chart[df_chart['GameCode'] == int(selected_game)]

        # Metrics
        total_shots = len(df_chart)
        made_shots = len(df_chart[df_chart['Shot_Result'] == 'Make'])
        percentage = (made_shots / total_shots * 100) if total_shots > 0 else 0
        
        st.metric(label="Field Goal %", value=f"{percentage:.1f}%", delta=f"{made_shots}/{total_shots}")

        # Plotting
        fig, ax = plt.subplots(figsize=(12, 11))
        draw_court(ax, color="black")
        
        misses = df_chart[df_chart['Shot_Result'] == 'Miss']
        makes = df_chart[df_chart['Shot_Result'] == 'Make']
        
        ax.scatter(misses['Coord_X'], misses['Coord_Y'], c='red', alpha=0.5, s=30, label='Miss', edgecolors='white', linewidth=0.5)
        ax.scatter(makes['Coord_X'], makes['Coord_Y'], c='green', alpha=0.8, s=30, label='Make', edgecolors='white', linewidth=0.5)
        
        title_str = f"{selected_season} | {selected_team} | {selected_player}"
        plt.title(title_str, fontsize=16)
        plt.legend(loc='upper right')
        
        st.pyplot(fig)
        
        with st.expander("View Shot Data"):
            st.dataframe(df_chart)
