import pandas as pd
import numpy as np
import folium
import geopandas as gpd

# ==========================================
# 1. CONFIGURATION
# ==========================================
GPS_FILE = '../data/matched_output (3).csv'
CHECKPOINT_FILE = './data/checkpoints.csv'
MAP_FILE = '../map/IITB_Auto_POV_Shapefile.gpkg'
DRIVER_NAME = 'Balkrishna Dawre'
RADIUS_METERS = 50
MIN_CONSECUTIVE_POINTS = 3

# ==========================================
# 2. DATA LOADING
# ==========================================
print(f"Loading data for {DRIVER_NAME}...")
gps_df = pd.read_csv(GPS_FILE)
checkpoints_df = pd.read_csv(CHECKPOINT_FILE)

# Filter Driver & Sort
df = gps_df[gps_df['Driver_Name'] == DRIVER_NAME].copy()
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

# Load Road Network
print("Loading IITB Road Network...")
try:
    gdf_roads = gpd.read_file(MAP_FILE)
    if gdf_roads.crs != 'EPSG:4326':
        gdf_roads = gdf_roads.to_crs('EPSG:4326')
    map_layer_available = True
except Exception as e:
    print(f"Warning: Could not load map file ({e}). Map will be generated without road network.")
    map_layer_available = False

# ==========================================
# 3. ANALYSIS LOGIC (USING MATCHED COORDS)
# ==========================================
print("Analyzing stops using Map-Matched coordinates...")

def get_distance_matrix(lats1, lons1, lats2, lons2):
    R = 6371000
    lat1, lon1 = np.radians(lats1), np.radians(lons1)
    lat2, lon2 = np.radians(lats2), np.radians(lons2)
    dlat = lat2[None, :] - lat1[:, None]
    dlon = lon2[None, :] - lon1[:, None]
    a = np.sin(dlat/2)**2 + np.cos(lat1[:, None]) * np.cos(lat2[None, :]) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return c * R

# --- KEY CHANGE: Use Matched_Lat / Matched_Lon ---
gps_lats = df['Matched_Lat'].values
gps_lons = df['Matched_Lon'].values
cp_lats = checkpoints_df['Lat'].values
cp_lons = checkpoints_df['Lon'].values
cp_names = checkpoints_df['Checkpoint'].values

# A. Point-wise Classification
dists = get_distance_matrix(gps_lats, gps_lons, cp_lats, cp_lons)
min_dists = np.min(dists, axis=1)
min_idxs = np.argmin(dists, axis=1)

df['Temp_Label'] = np.where(min_dists <= RADIUS_METERS, cp_names[min_idxs], "Inbetween Point")

# B. Identify Blocks
df['block_id'] = (df['Temp_Label'] != df['Temp_Label'].shift()).cumsum()

# Aggregation: Average the MATCHED coordinates now
stops = df.groupby(['block_id', 'Temp_Label']).agg(
    start_time=('timestamp', 'min'),
    end_time=('timestamp', 'max'),
    start_ist=('IST_time', 'first'),
    avg_lat=('Matched_Lat', 'mean'),  # <--- CHANGED to Matched
    avg_lon=('Matched_Lon', 'mean'),  # <--- CHANGED to Matched
    count=('timestamp', 'count')
).reset_index()

valid_stops = stops[stops['count'] >= MIN_CONSECUTIVE_POINTS].copy()
valid_stops['duration_sec'] = (valid_stops['end_time'] - valid_stops['start_time']).dt.total_seconds()

# C. Centroid Correction (Using Matched Means)
stop_lats = valid_stops['avg_lat'].values
stop_lons = valid_stops['avg_lon'].values
stop_dists = get_distance_matrix(stop_lats, stop_lons, cp_lats, cp_lons)
min_stop_dists = np.min(stop_dists, axis=1)
min_stop_idxs = np.argmin(stop_dists, axis=1)

final_labels = []
final_colors = []

for i, dist in enumerate(min_stop_dists):
    if dist <= RADIUS_METERS:
        check_name = cp_names[min_stop_idxs[i]]
        final_labels.append(check_name)
        final_colors.append("green")
    else:
        final_labels.append("Inbetween Point")
        final_colors.append("red")

valid_stops['Final_Location'] = final_labels
valid_stops['Color'] = final_colors
valid_stops['Centroid_Dist'] = min_stop_dists

# ==========================================
# 4. SUMMARY & VISUALIZATION
# ==========================================
valid_stops['IST_Hour'] = pd.to_datetime(valid_stops['start_ist'], format='%H:%M:%S').dt.hour

print(f"\n--- ANALYSIS SUMMARY FOR {DRIVER_NAME} ---")
print("\n[Top 5 Most Frequent Stop Locations]")
print(valid_stops['Final_Location'].value_counts().head(5))

print("\n[Stops by Time of Day (IST)]")
print(valid_stops.groupby('IST_Hour').size())

valid_stops.to_csv('final_stationary_analysis.csv', index=False)

# Create Map
if not valid_stops.empty:
    center_lat = valid_stops['avg_lat'].mean()
    center_lon = valid_stops['avg_lon'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=15)

    if map_layer_available:
        folium.GeoJson(
            gdf_roads,
            name='IITB Road Network',
            style_function=lambda x: {'color': '#666666', 'weight': 1.5, 'opacity': 0.7}
        ).add_to(m)

    # Checkpoints
    for _, row in checkpoints_df.iterrows():
        folium.CircleMarker(
            location=[row['Lat'], row['Lon']],
            radius=3, color='blue', fill=True, popup=row['Checkpoint']
        ).add_to(m)
        folium.Circle(
            location=[row['Lat'], row['Lon']],
            radius=RADIUS_METERS, color='blue', fill=True, fill_opacity=0.05
        ).add_to(m)

    # Stops (Now Perfectly on Road)
    for _, row in valid_stops.iterrows():
        label = row['Final_Location']
        color = row['Color']
        
        popup_html = f"""
        <div style="width: 200px;">
            <b style="color:{color}">{label}</b><br>
            Start: {row['start_ist']}<br>
            Duration: {row['duration_sec']:.0f} sec<br>
            Dist to Checkpoint: {row['Centroid_Dist']:.1f} m
        </div>
        """
        folium.Marker(
            location=[row['avg_lat'], row['avg_lon']],
            popup=folium.Popup(popup_html, max_width=250),
            icon=folium.Icon(color=color, icon='info-sign' if color=='red' else 'ok-sign')
        ).add_to(m)

    folium.LayerControl().add_to(m)
    m.save('driver_stops_map_matched.html')
    print("\nMap saved as 'driver_stops_map_matched.html'")