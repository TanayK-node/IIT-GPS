import pandas as pd
import geopandas as gpd
import folium
from shapely.ops import unary_union

# =========================
# 🔴 CONFIGURATION
# =========================
TARGET_DRIVER_ID = "6841659e3b823c2921eec0e6"  # The driver in question
DRIVER_POINTS_FILE = "../data/matched_output (3).csv"
DRIVER_ID_FILE = "0_ID_name.csv"

# Check your file paths carefully!
ROUTE_FILES = {
    "yellow": "./Gkpg/yellow_assigned_route.gpkg",
    "green": "./Gkpg/green_assigned_route.gpkg",
    "blue": "./Gkpg/blue_assigned_route.gpkg"
}

METRIC_CRS = 32643
WGS_CRS = 4326

# =========================
# 1. VERIFY DRIVER LIST MEMBERSHIP
# =========================
# (Copying lists exactly from your file)
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

assigned_route = None
if TARGET_DRIVER_ID in YELLOW_IDS:
    assigned_route = "yellow"
elif TARGET_DRIVER_ID in GREEN_IDS:
    assigned_route = "green"
else:
    assigned_route = "blue" # Assuming blue if not in others

print(f"🔍 Driver {TARGET_DRIVER_ID}")
print(f"✅ Found in list: {assigned_route.upper()}")
print(f"📂 Loading GPKG file: {ROUTE_FILES[assigned_route]}")

# =========================
# 2. LOAD ROUTE (With Explicit Color)
# =========================
gpkg_path = ROUTE_FILES[assigned_route]
route_gdf = pd.DataFrame()

try:
    # Load Main
    main_gdf = gpd.read_file(gpkg_path, layer='main_route')
    # Load Extra (if exists)
    try:
        extra_gdf = gpd.read_file(gpkg_path, layer='extra_routes')
        route_gdf = pd.concat([main_gdf, extra_gdf], ignore_index=True)
    except:
        route_gdf = main_gdf
        
    route_gdf = route_gdf.to_crs(METRIC_CRS)
    buffer_geom = route_gdf.unary_union.buffer(15) # 15m buffer
    
except Exception as e:
    print(f"❌ Critical Error loading GPKG: {e}")
    exit()

# =========================
# 3. LOAD DRIVER POINTS
# =========================
df = pd.read_csv(DRIVER_POINTS_FILE)
id_df = pd.read_csv(DRIVER_ID_FILE)

# Merge IDs
df['name_norm'] = df['Driver_Name'].astype(str).str.strip().str.lower()
id_df['name_norm'] = id_df['name'].astype(str).str.strip().str.lower()
merged = pd.merge(df, id_df[['name_norm', 'driver_id']], on='name_norm', how='left')

# Filter
driver_pts = merged[merged['driver_id'] == TARGET_DRIVER_ID].copy()
driver_gdf = gpd.GeoDataFrame(
    driver_pts, 
    geometry=gpd.points_from_xy(driver_pts.Matched_Lon, driver_pts.Matched_Lat),
    crs=WGS_CRS
).to_crs(METRIC_CRS)

# Calculate Adherence
driver_gdf['on_route'] = driver_gdf.geometry.within(buffer_geom)
adherence = driver_gdf['on_route'].mean() * 100
print(f"📊 Calculated Adherence: {adherence:.2f}%")

# =========================
# 4. VISUALIZE (Sanity Check)
# =========================
m = folium.Map(location=[19.1334, 72.9133], zoom_start=14, tiles="cartodbpositron")

# Set color based on route name
color_map = {"yellow": "#FFD700", "green": "#008000", "blue": "#0000FF"}
route_color = color_map.get(assigned_route, "black")

# Plot Route
folium.GeoJson(
    route_gdf.to_crs(WGS_CRS),
    name=f"Assigned Route: {assigned_route}",
    style_function=lambda x: {'color': route_color, 'weight': 6, 'opacity': 0.7},
    tooltip=f"{assigned_route.upper()} Route"
).add_to(m)

# Plot Points (Red=Off, Green=On)
for _, row in driver_gdf.to_crs(WGS_CRS).iterrows():
    folium.CircleMarker(
        [row.geometry.y, row.geometry.x],
        radius=4,
        color="green" if row['on_route'] else "red",
        fill=True,
        fill_opacity=1,
        popup=f"Timestamp: {row.get('Timestamp', 'N/A')}"
    ).add_to(m)

folium.LayerControl().add_to(m)
m.save("debug_driver_check.html")
print("🎉 Created 'debug_driver_check.html'")
print("👉 Open this map. If the thick line is NOT the path you expect for 'Yellow',")
print("   then your 'yellow_assigned_route.gpkg' file has the wrong data inside it.")