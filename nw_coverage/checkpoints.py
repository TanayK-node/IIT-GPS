import pandas as pd
import geopandas as gpd
import folium
from shapely.geometry import Point

# 1. Load the Data
# ----------------
# Load Checkpoints
df = pd.read_csv('./data/checkpoints.csv')

# Ensure coordinates are numeric and drop invalid rows
df['Lat'] = pd.to_numeric(df['Lat'], errors='coerce')
df['Lon'] = pd.to_numeric(df['Lon'], errors='coerce')
df = df.dropna(subset=['Lat', 'Lon'])

# Load the IITB Map (GPKG)
try:
    # Read the GPKG file
    gdf_map = gpd.read_file('../map/IITB_Auto_POV_Shapefile.gpkg')
    
    # Ensure it's in WGS84 (Lat/Lon) for Folium
    if gdf_map.crs != 'EPSG:4326':
        gdf_map = gdf_map.to_crs('EPSG:4326')
except Exception as e:
    print(f"Could not load map file: {e}")
    gdf_map = None

# 2. Create the Interactive Map
# ---------------------------
# Center the map on the average location of the points
center_lat = df['Lat'].mean()
center_lon = df['Lon'].mean()
m = folium.Map(location=[center_lat, center_lon], zoom_start=15)

# Add the IITB Map Layer (if loaded)
if gdf_map is not None:
    folium.GeoJson(
        gdf_map,
        name='IITB Map',
        style_function=lambda x: {'color': 'gray', 'weight': 2, 'opacity': 0.5}
    ).add_to(m)

# 3. Add Points and Circles
# -------------------------
for idx, row in df.iterrows():
    # A. Marker with Popup (Click to see name)
    folium.Marker(
        location=[row['Lat'], row['Lon']],
        popup=row['Checkpoint'],  # Shows name on click
        tooltip=row['Checkpoint'] # Shows name on hover
    ).add_to(m)
    
    # B. 70m Radius Circle
    folium.Circle(
        location=[row['Lat'], row['Lon']],
        radius=50,      # Radius in meters
        color='blue',   # Outline color
        fill=True,
        fill_color='blue',
        fill_opacity=0.1
    ).add_to(m)

# 4. Save the Map
# ---------------
output_file = 'iitb_checkpoints_map.html'
m.save(output_file)
print(f"Map saved as {output_file}. Open this file in your browser.")