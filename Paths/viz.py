import pandas as pd
import geopandas as gpd
import folium
from shapely.ops import unary_union

# =========================
# CONFIGURATION
# =========================
# 🔴 ENTER THE DRIVER ID YOU WANT TO CHECK HERE:
TARGET_DRIVER_ID = "6841659e3b823c2921eec0e6"  # Example ID

DRIVER_POINTS_FILE = "../data/matched_output (3).csv"
DRIVER_ID_FILE = "0_ID_name.csv"
BUFFER_METERS = 15
METRIC_CRS = 32643
WGS_CRS = 4326

# ID LISTS (Keep these updated)
YELLOW_IDS = [
    "6841659e3b823c2921eec0e6", "684160283b823c2921ed8ace", "684161ee3b823c2921edc555",
    "68416cce3b823c2921eff7b7", "6854fd3f33907799afa85dbf", "684163ae3b823c2921ee86a4",
    "68416cc03b823c2921eff504", "68416aaf3b823c2921ef90c6", "684167983b823c2921ef08fa",
    "684168be3b823c2921ef3847", "684166973b823c2921eee1b0", "684168f83b823c2921ef4216",
    "684167c33b823c2921ef0fbe", "686892708e4cc1172dc39f79", "686b339ac355d8d67a31413d",
    "688adb513529d261de4d7dd6", "69390be8a90d98af3a6f0685", "6939052fa90d98af3a65bd82",
    "68bfbfd15293872ea5c9f333", "68c3ef94134c3430ac9be57b", "68626cc571a184af7bee6ae0",
    "685e7f1863ae3f9a9be232dd", "6847d691bb47ec5ecf9a6ad4", "68626d5edfd9660840e7d4af",
    "685fb268934028c9386aa988"
]

GREEN_IDS = [
    "6849198e679906ce86a5b88d", "6846a1e291045f20230c4ed1", "6846a6ba91045f20230d16ad",
    "6846a6c591045f20230d1774", "68491a3c679906ce86a5d282", "6847d691bb47ec5ecf9a6ad4",
    "68a5690ffe5e61d750ef585b", "6846a2fd91045f20230c5bde"
]

BLUE_IDS = [
    "6846a34191045f20230c5f23", "684915c0679906ce86a3b7dd", "684916ac679906ce86a5480d",
    "6849185e679906ce86a58b06", "6889b22cb94407c1fb400dc5", "6895c6664d39646dc793060e",
    "6895cb75002f883f302a26e5", "6895cbb3161c6e96886022ec", "68be6a2b1ca322c5e34122a4",
    "68c3ecc5134c3430ac979274"
]

# =========================
# 1. IDENTIFY ROUTE & FILE
# =========================
if TARGET_DRIVER_ID in YELLOW_IDS:
    route_color = "yellow"
    gpkg_path = "./Gkpg/yellow_assigned_route.gpkg"
elif TARGET_DRIVER_ID in GREEN_IDS:
    route_color = "green"
    gpkg_path = "./Gkpg/green_assigned_route.gpkg"
elif TARGET_DRIVER_ID in BLUE_IDS:
    route_color = "blue"
    gpkg_path = "./Gkpg/blue_assigned_route.gpkg"
else:
    print(f"❌ Driver ID {TARGET_DRIVER_ID} not found in any list!")
    exit()

print(f"🔍 Driver {TARGET_DRIVER_ID} is assigned to: {route_color.upper()} route")

# =========================
# 2. LOAD ROUTE (ALL LAYERS)
# =========================
try:
    # Load Main Route
    gdf_main = gpd.read_file(gpkg_path, layer='main_route').to_crs(METRIC_CRS)
    
    # Try to load Extra Routes (might be empty/non-existent)
    try:
        gdf_extra = gpd.read_file(gpkg_path, layer='extra_routes').to_crs(METRIC_CRS)
        full_route = pd.concat([gdf_main, gdf_extra], ignore_index=True)
        print(f"✅ Loaded main + extra routes.")
    except:
        full_route = gdf_main
        print(f"⚠️ Only main route found (no extra layer).")

    # Create Buffer for adherence check
    route_geom = unary_union(full_route.geometry)
    route_buffer = route_geom.buffer(BUFFER_METERS)
    
except Exception as e:
    print(f"❌ Error loading GPKG: {e}")
    exit()

# =========================
# 3. LOAD & FILTER DRIVER POINTS
# =========================
print("📥 Loading driver points...")
df = pd.read_csv(DRIVER_POINTS_FILE)
id_df = pd.read_csv(DRIVER_ID_FILE)

# Normalize and Merge
df['name_norm'] = df['Driver_Name'].astype(str).str.strip().str.lower()
id_df['name_norm'] = id_df['name'].astype(str).str.strip().str.lower()
merged_df = pd.merge(df, id_df[['name_norm', 'driver_id']], on='name_norm', how='left')

# Filter for specific driver
driver_pts = merged_df[merged_df['driver_id'] == TARGET_DRIVER_ID].copy()

if driver_pts.empty:
    print("❌ No points found for this driver ID. Check name matching.")
    exit()

# Convert to GeoDataFrame
driver_gdf = gpd.GeoDataFrame(
    driver_pts,
    geometry=gpd.points_from_xy(driver_pts["Matched_Lon"], driver_pts["Matched_Lat"]),
    crs=WGS_CRS
).to_crs(METRIC_CRS)

# =========================
# 4. CHECK ADHERENCE
# =========================
# Determine which points are On/Off route
is_on_route = driver_gdf.geometry.within(route_buffer)
driver_gdf['status'] = is_on_route.map({True: 'On Route', False: 'Off Route'})

print(f"📊 Stats for {driver_pts.iloc[0]['Driver_Name']}:")
print(driver_gdf['status'].value_counts())

# =========================
# 5. VISUALIZE
# =========================
# Convert back to WGS84 for Folium
route_wgs = full_route.to_crs(WGS_CRS)
points_wgs = driver_gdf.to_crs(WGS_CRS)

center_lat = points_wgs.geometry.y.mean()
center_lon = points_wgs.geometry.x.mean()

m = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles="cartodbpositron")

# A. Draw Route (Blue for Main, Cyan for Extra if distinguishable, or just one color)
folium.GeoJson(
    route_wgs,
    style_function=lambda x: {'color': 'blue', 'weight': 5, 'opacity': 0.6},
    name="Assigned Route"
).add_to(m)

# B. Draw Points (Green = Good, Red = Bad)
for idx, row in points_wgs.iterrows():
    color = "green" if row['status'] == 'On Route' else "red"
    
    folium.CircleMarker(
        location=[row.geometry.y, row.geometry.x],
        radius=3,
        color=color,
        fill=True,
        fill_opacity=0.9,
        popup=f"Time: {row.get('Timestamp', 'N/A')}<br>Status: {row['status']}"
    ).add_to(m)

# Add Legend/Layer Control
folium.LayerControl().add_to(m)

# Save
filename = f"viz_driver_{TARGET_DRIVER_ID}.html"
m.save(filename)
print(f"🎉 Map saved: {filename}")