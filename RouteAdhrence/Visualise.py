import streamlit as st
import pandas as pd
import folium
import json
from streamlit_folium import st_folium

# ==========================================
# 1. SETUP & DATA LOADING
# ==========================================
st.set_page_config(page_title="Trip Visualizer", layout="wide")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('../data/Sample_Trip Data.csv')
    except FileNotFoundError:
        st.error("File 'Sample_Trip Data.csv' not found.")
        return pd.DataFrame()

    # Helper to parse paths
    def parse_path(path_str):
        if pd.isna(path_str) or path_str == '':
            return []
        try:
            data = json.loads(path_str)
            return [[p.get('latitude'), p.get('longitude')] for p in data 
                    if p.get('latitude') is not None]
        except:
            return []

    # Parse all relevant columns
    df['parsed_ridePath'] = df['ridePath'].apply(parse_path)
    df['parsed_prescribed'] = df['ridePathPrescribed'].apply(parse_path)
    df['parsed_pickup'] = df['pathForUserPickupLocation'].apply(parse_path)
    
    # Calculate lengths for quick stats
    df['len_actual'] = df['parsed_ridePath'].apply(len)
    df['len_prescribed'] = df['parsed_prescribed'].apply(len)
    
    return df

df = load_data()

# ==========================================
# 2. SIDEBAR CONTROLS
# ==========================================
st.sidebar.title("Navigation")

# Filter options
only_with_data = st.sidebar.checkbox("Show only trips with data", value=True)

if only_with_data:
    # Filter df to rows that have at least one path
    display_df = df[
        (df['parsed_ridePath'].apply(len) > 0) | 
        (df['parsed_prescribed'].apply(len) > 0) | 
        (df['parsed_pickup'].apply(len) > 0)
    ]
else:
    display_df = df

# Trip Selector
if not display_df.empty:
    # Create a list of readable labels: "Index | TripID (Points)"
    trip_options = display_df.index.tolist()
    
    def format_func(idx):
        row = df.loc[idx]
        trip_id = str(row['_id'])[-6:] # Show last 6 chars of ID
        pts = row['len_actual']
        return f"Idx {idx} | ID ...{trip_id} | {pts} pts"

    selected_index = st.sidebar.selectbox(
        "Select a Trip:", 
        options=trip_options,
        format_func=format_func
    )
    
    # Get the selected row
    trip = df.loc[selected_index]
else:
    st.warning("No trips found matching criteria.")
    st.stop()

# Layer Toggles
st.sidebar.markdown("---")
st.sidebar.subheader("Layer Visibility")
show_actual = st.sidebar.checkbox("Actual Path (Blue)", value=True)
show_prescribed = st.sidebar.checkbox("Prescribed Path (Red)", value=True)
show_pickup = st.sidebar.checkbox("Pickup Path (Green)", value=True)

# ==========================================
# 3. MAIN DISPLAY
# ==========================================
st.title(f"Trip Analysis: `{trip['_id']}`")

# Metrics Row
col1, col2, col3 = st.columns(3)
col1.metric("Actual Points", len(trip['parsed_ridePath']))
col2.metric("Prescribed Points", len(trip['parsed_prescribed']))
col3.metric("Pickup Points", len(trip['parsed_pickup']))

# Map Creation
# Center map on the first available point
start_lat, start_lon = 19.1334, 72.9133
if trip['parsed_ridePath']:
    start_lat, start_lon = trip['parsed_ridePath'][0]
elif trip['parsed_prescribed']:
    start_lat, start_lon = trip['parsed_prescribed'][0]
elif not pd.isna(trip['originLat']):
    start_lat, start_lon = trip['originLat'], trip['originLong']

m = folium.Map(location=[start_lat, start_lon], zoom_start=15, tiles='cartodbpositron')

# A. Actual Path
if show_actual and trip['parsed_ridePath']:
    folium.PolyLine(
        trip['parsed_ridePath'], color='blue', weight=4, opacity=0.8, tooltip='Actual'
    ).add_to(m)

# B. Prescribed Path
if show_prescribed and trip['parsed_prescribed']:
    folium.PolyLine(
        trip['parsed_prescribed'], color='red', weight=4, dash_array='5, 5', opacity=1.0, tooltip='Prescribed'
    ).add_to(m)

# C. Pickup Path
if show_pickup and trip['parsed_pickup']:
    folium.PolyLine(
        trip['parsed_pickup'], color='green', weight=4, opacity=0.8, tooltip='Pickup'
    ).add_to(m)

# Markers
if not pd.isna(trip['originLat']):
    folium.Marker(
        [trip['originLat'], trip['originLong']], 
        popup='Origin', 
        icon=folium.Icon(color='green', icon='play')
    ).add_to(m)

if not pd.isna(trip['destLat']):
    folium.Marker(
        [trip['destLat'], trip['destLong']], 
        popup='Destination', 
        icon=folium.Icon(color='red', icon='stop')
    ).add_to(m)

# Render Map
st_folium(m, width=1000, height=600)

# Show Raw Data Table
with st.expander("See Raw Data for this Trip"):
    st.write(trip)