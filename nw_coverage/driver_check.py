import pandas as pd
import numpy as np
import folium

# ==========================================
# 1. CONFIGURATION
# ==========================================
GPS_FILE = '../data/matched_output (3).csv'
CHECKPOINT_FILE = './data/checkpoints.csv'
DRIVER_NAME = 'Balkrishna Dawre'
RADIUS_METERS = 50
MIN_CONSECUTIVE_POINTS = 3

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def get_distance_matrix(lats1, lons1, lats2, lons2):
    """Calculates distance matrix between two sets of coordinates (in meters)"""
    R = 6371000  # Earth radius
    
    # Convert to radians
    lat1, lon1 = np.radians(lats1), np.radians(lons1)
    lat2, lon2 = np.radians(lats2), np.radians(lons2)
    
    # Broadcasting
    dlat = lat2[None, :] - lat1[:, None]
    dlon = lon2[None, :] - lon1[:, None]
    
    a = np.sin(dlat/2)**2 + np.cos(lat1[:, None]) * np.cos(lat2[None, :]) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return c * R

# ==========================================
# 3. DATA PROCESSING
# ==========================================
print(f"Loading data for {DRIVER_NAME}...")
gps_df = pd.read_csv(GPS_FILE)
checkpoints_df = pd.read_csv(CHECKPOINT_FILE)

# Filter Driver & Sort
df = gps_df[gps_df['Driver_Name'] == DRIVER_NAME].copy()
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

# --- STEP A: Initial Classification (To find 'potential' stops) ---
# We look for ANY cluster of points, regardless of location
# We group by time/proximity simply to identify "stopped" behavior first.
# For simplicity, we can use the previous logic but be more lenient initially.

# Calculate dist to NEAREST checkpoint for every point
cp_lats = checkpoints_df['Lat'].values
cp_lons = checkpoints_df['Lon'].values
gps_lats = df['latitude'].values
gps_lons = df['longitude'].values

# Initial Classification
dists = get_distance_matrix(gps_lats, gps_lons, cp_lats, cp_lons)
min_dists = np.min(dists, axis=1)
min_idxs = np.argmin(dists, axis=1)
cp_names = checkpoints_df['Checkpoint'].values

# We label points initially. 
# Note: We temporarily separate "Inbetween" vs "Checkpoint" 
# to group them, but we will RE-CHECK the final group later.
df['Temp_Label'] = np.where(min_dists <= RADIUS_METERS, cp_names[min_idxs], "Inbetween")

# Identify Blocks (Consecutive same labels)
df['block_id'] = (df['Temp_Label'] != df['Temp_Label'].shift()).cumsum()

# Aggregate Blocks into Stops
stops = df.groupby(['block_id', 'Temp_Label']).agg(
    start_time=('timestamp', 'min'),
    end_time=('timestamp', 'max'),
    avg_lat=('latitude', 'mean'),
    avg_lon=('longitude', 'mean'),
    point_count=('timestamp', 'count')
).reset_index()

# Filter for duration/count
valid_stops = stops[stops['point_count'] >= MIN_CONSECUTIVE_POINTS].copy()
valid_stops['duration_min'] = (valid_stops['end_time'] - valid_stops['start_time']).dt.total_seconds() / 60

# --- STEP B: FINAL RE-VALIDATION (The Fix) ---
# Now we take the AVERAGE location of the stop and check distance again.
# This ensures visual consistency.

stop_lats = valid_stops['avg_lat'].values
stop_lons = valid_stops['avg_lon'].values

# Calculate distance from STOP CENTER to all checkpoints
stop_dists = get_distance_matrix(stop_lats, stop_lons, cp_lats, cp_lons)
min_stop_dists = np.min(stop_dists, axis=1)
min_stop_idxs = np.argmin(stop_dists, axis=1)

# Assign FINAL Label based on the Stop Center
final_labels = []
final_colors = []

for i, dist in enumerate(min_stop_dists):
    if dist <= RADIUS_METERS:
        # It IS inside the radius! Force Green.
        check_name = cp_names[min_stop_idxs[i]]
        final_labels.append(check_name)
        final_colors.append("green")
    else:
        # Still outside. Keep Red.
        final_labels.append("Inbetween Point")
        final_colors.append("red")

valid_stops['Final_Label'] = final_labels
valid_stops['Color'] = final_colors
valid_stops['Dist_to_Checkpoint'] = min_stop_dists

print(f"Re-validated {len(valid_stops)} stops. Plotting map...")

# ==========================================
# 4. VISUALIZATION
# ==========================================
if not valid_stops.empty:
    center_lat = valid_stops['avg_lat'].mean()
    center_lon = valid_stops['avg_lon'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles='OpenStreetMap')

    # Draw Checkpoints (Blue Zones)
    for _, row in checkpoints_df.iterrows():
        # Center Dot
        folium.CircleMarker(
            location=[row['Lat'], row['Lon']],
            radius=2, color='blue', fill=True, popup=row['Checkpoint']
        ).add_to(m)
        
        # 50m Radius Circle
        folium.Circle(
            location=[row['Lat'], row['Lon']],
            radius=RADIUS_METERS,
            color='#3186cc',
            weight=1,
            fill=True,
            fill_opacity=0.1,
            popup=f"50m Radius: {row['Checkpoint']}"
        ).add_to(m)

    # Draw Stops
    for _, row in valid_stops.iterrows():
        label = row['Final_Label']
        color = row['Color']
        dist = row['Dist_to_Checkpoint']
        
        # Icon
        icon_type = 'ok-sign' if color == 'green' else 'info-sign'
        
        popup_html = f"""
        <div style="font-family: Arial; width: 220px;">
            <h4 style="color:{color}">{label}</h4>
            <b>Start:</b> {row['start_time'].strftime('%H:%M:%S')}<br>
            <b>Duration:</b> {row['duration_min']:.2f} min<br>
            <b>Dist to Checkpoint:</b> {dist:.1f} m
        </div>
        """
        
        folium.Marker(
            location=[row['avg_lat'], row['avg_lon']],
            popup=folium.Popup(popup_html, max_width=250),
            icon=folium.Icon(color=color, icon=icon_type)
        ).add_to(m)

    output_file = 'driver_stops_corrected.html'
    m.save(output_file)
    print(f"Done! Map saved as '{output_file}'.")
else:
    print("No stops found.")