import pandas as pd
import geopandas as gpd
from shapely.ops import unary_union

# =========================
# CONFIGURATION
# =========================
DRIVER_POINTS_FILE = "../data/matched_output (3).csv"
DRIVER_ID_FILE = "0_ID_name.csv"

# Ensure these paths are correct relative to where you run the script
ROUTE_FILES = {
    "yellow": "./Gkpg/yellow_assigned_route.gpkg",
    "green": "./Gkpg/green_assigned_route.gpkg",
    "blue": "./Gkpg/blue_assigned_route.gpkg"
}

BUFFER_METERS = 15
METRIC_CRS = 32643  # UTM Zone 43N (meters)
WGS_CRS = 4326      # Lat/Lon

# =========================
# DRIVER ID LISTS
# =========================
YELLOW_DRIVER_IDS = [
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

GREEN_DRIVER_IDS = [
    "6849198e679906ce86a5b88d", "6846a1e291045f20230c4ed1", "6846a6ba91045f20230d16ad",
    "6846a6c591045f20230d1774", "68491a3c679906ce86a5d282", "6847d691bb47ec5ecf9a6ad4",
    "68a5690ffe5e61d750ef585b", "6846a2fd91045f20230c5bde"
]

BLUE_DRIVER_IDS = [
    "6846a34191045f20230c5f23", "684915c0679906ce86a3b7dd", "684916ac679906ce86a5480d",
    "6849185e679906ce86a58b06", "6889b22cb94407c1fb400dc5", "6895c6664d39646dc793060e",
    "6895cb75002f883f302a26e5", "6895cbb3161c6e96886022ec", "68be6a2b1ca322c5e34122a4",
    "68c3ecc5134c3430ac979274"
]

# Build Unified Map: Driver_ID -> Route_Name
DRIVER_ID_ROUTE_MAP = {}
for d_id in YELLOW_DRIVER_IDS: DRIVER_ID_ROUTE_MAP[d_id] = "yellow"
for d_id in GREEN_DRIVER_IDS:  DRIVER_ID_ROUTE_MAP[d_id] = "green"
for d_id in BLUE_DRIVER_IDS:   DRIVER_ID_ROUTE_MAP[d_id] = "blue"

# =========================
# 1. LOAD DATA & ATTACH IDs
# =========================
print("📥 Loading Data...")

# Load GPS Data
try:
    df = pd.read_csv(DRIVER_POINTS_FILE)
except FileNotFoundError:
    print(f"❌ Error: GPS file not found at {DRIVER_POINTS_FILE}")
    exit()

# Load ID Map
try:
    id_df = pd.read_csv(DRIVER_ID_FILE)
except FileNotFoundError:
    print(f"❌ Error: ID file not found at {DRIVER_ID_FILE}")
    exit()

# Normalize names for merging (lowercase, strip whitespace)
df['name_norm'] = df['Driver_Name'].astype(str).str.strip().str.lower()
id_df['name_norm'] = id_df['name'].astype(str).str.strip().str.lower()

# Merge to attach Driver ID to GPS points
merged_df = pd.merge(df, id_df[['name_norm', 'driver_id']], on='name_norm', how='left')

# Convert to GeoDataFrame
drivers_gdf = gpd.GeoDataFrame(
    merged_df,
    geometry=gpd.points_from_xy(merged_df["Matched_Lon"], merged_df["Matched_Lat"]),
    crs=WGS_CRS
).to_crs(METRIC_CRS)

print(f"✅ Loaded {len(drivers_gdf)} GPS points.")

# =========================
# 2. LOAD & BUFFER ROUTES (FIXED MULTI-LAYER)
# =========================
route_buffers = {}

print("🔄 Loading Route Geometries...")

for route_name, gpkg in ROUTE_FILES.items():
    try:
        # A. Load Main Route Layer
        main_gdf = gpd.read_file(gpkg, layer='main_route')
        
        # B. Load Extra Routes Layer (Handle if it doesn't exist/is empty)
        try:
            extra_gdf = gpd.read_file(gpkg, layer='extra_routes')
            # Combine both into one GeoDataFrame
            combined_gdf = pd.concat([main_gdf, extra_gdf], ignore_index=True)
            print(f"   ✅ {route_name}: Loaded Main + Extra routes.")
        except Exception:
            # If extra_routes layer is missing, just use main
            combined_gdf = main_gdf
            print(f"   ⚠️ {route_name}: Loaded Main route only (no extra layer found).")

        # C. Set CRS and Buffer
        if combined_gdf.crs is None:
            combined_gdf.set_crs(epsg=WGS_CRS, inplace=True)

        combined_gdf = combined_gdf.to_crs(METRIC_CRS)
        
        # Create a single merged polygon buffer for the whole route
        route_buffers[route_name] = combined_gdf.unary_union.buffer(BUFFER_METERS)
        
    except Exception as e:
        print(f"   ❌ Error loading route {route_name}: {e}")

# =========================
# 3. ROUTE ADHERENCE CALC
# =========================
results = []
missing_drivers = [] 

print("🔄 Calculating Adherence...")

for driver_id, route_name in DRIVER_ID_ROUTE_MAP.items():

    # Filter data by Driver ID
    driver_pts = drivers_gdf[drivers_gdf["driver_id"] == driver_id]

    # CHECK FOR MISSING DATA
    if driver_pts.empty:
        # Find name for reporting
        name_row = id_df[id_df['driver_id'] == driver_id]
        d_name = name_row.iloc[0]['name'] if not name_row.empty else "Unknown Name"
        
        missing_drivers.append({
            "Driver_ID": driver_id, 
            "Name": d_name, 
            "Assigned_Route": route_name
        })
        continue

    # Get the name for the CSV output
    driver_name_display = driver_pts.iloc[0]['Driver_Name']

    if route_name not in route_buffers:
        print(f"⚠️ Route {route_name} buffer not available, skipping {driver_id}")
        continue

    buffer_geom = route_buffers[route_name]
    
    # The Core Check: Are points within the combined buffer?
    on_route = driver_pts.geometry.within(buffer_geom)

    total = len(driver_pts)
    on_count = int(on_route.sum())

    results.append({
        "Driver_ID": driver_id,
        "Driver_Name": driver_name_display,
        "Assigned_Route": route_name,
        "Total_Points": total,
        "On_Route_Points": on_count,
        "Off_Route_Points": total - on_count,
        "Route_Adherence_%": round((on_count / total) * 100, 2)
    })

# =========================
# 4. EXPORT & REPORT
# =========================
result_df = pd.DataFrame(results)
output_csv = "route_adherence_by_id_fixed.csv"
result_df.to_csv(output_csv, index=False)

print("\n🎉 CALCULATION DONE")
print(f"Saved: {output_csv}")

# PRINT MISSING DRIVERS
if missing_drivers:
    print("\n" + "="*50)
    print("⚠️  DRIVERS WITH NO DATA (CALCULATION SKIPPED)")
    print("="*50)
    missing_df = pd.DataFrame(missing_drivers)
    print(missing_df.to_string(index=False))
    print("="*50)
else:
    print("\n✅ All assigned drivers had GPS data.")