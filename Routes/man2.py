# ============================================================
# IITB MANUAL ROUTE SYSTEM (FINAL FIXED VERSION)
# FIXED: METRIC CRS + ROBUST VISUALIZATION + BUFFERING + MULTI-INTERVAL
# ============================================================

import pandas as pd
import geopandas as gpd
import numpy as np
import networkx as nx
import folium
import json
import glob
import warnings
from shapely.geometry import Point, LineString
from shapely.ops import unary_union
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")

# ============================================================
# SETTINGS
# ============================================================

PROJECT_CRS = "EPSG:32643"  # UTM Zone 43N (Meters)
BUFFER_DISTANCE = 20        # 20 Meters tolerance
TOTAL_DRIVERS = 60

# ============================================================
# 1. LOAD ROADS & BUILD GRAPH
# ============================================================

print("Loading roads...")
roads = gpd.read_file("../map/IITB_Auto_POV_Shapefile.gpkg")
roads = roads.to_crs(PROJECT_CRS)

print("Building graph...")
G = nx.Graph()

for _, row in roads.iterrows():
    if row.geometry is None: 
        continue
    
    # Handle MultiLineString vs LineString
    lines = row.geometry.geoms if row.geometry.geom_type == "MultiLineString" else [row.geometry]

    for line in lines:
        coords = list(line.coords)
        for i in range(len(coords)-1):
            x1, y1 = coords[i]
            x2, y2 = coords[i+1]
            dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            G.add_edge((x1,y1), (x2,y2), weight=dist)

print("Graph built:", len(G.nodes), "nodes")

# Build KDTree for Snapping
nodes = np.array(list(G.nodes))
tree = cKDTree(nodes)

def snap_node(x, y):
    _, idx = tree.query([x, y])
    return tuple(nodes[idx])

# ============================================================
# 2. LOAD & PROCESS GPS
# ============================================================

print("Loading GPS files...")
files = glob.glob("./data/*.csv")
gps_list = []

for f in files:
    df = pd.read_csv(f)
    df.columns = df.columns.str.strip()
    gps_list.append(df)

gps_data = pd.concat(gps_list, ignore_index=True)

# Clean Coordinates
gps_data['mapmatched lat'] = pd.to_numeric(gps_data['mapmatched lat'], errors='coerce')
gps_data['mapmatched lon'] = pd.to_numeric(gps_data['mapmatched lon'], errors='coerce')
gps_data = gps_data.dropna(subset=['mapmatched lat', 'mapmatched lon'])

# Create Geometry & Convert to Metric
gps_data['geometry'] = gps_data.apply(
    lambda r: Point(r['mapmatched lon'], r['mapmatched lat']), 
    axis=1
)
gps_gdf = gpd.GeoDataFrame(gps_data, geometry='geometry', crs="EPSG:4326")
gps_gdf = gps_gdf.to_crs(PROJECT_CRS)

# Parse Timestamps
gps_gdf['timestamp'] = pd.to_datetime(
    gps_gdf['date'].astype(str) + " " + gps_gdf['ist time'].astype(str),
    dayfirst=True, 
    errors='coerce'
)
gps_gdf = gps_gdf.dropna(subset=['timestamp'])
gps_gdf = gps_gdf.sort_values(['driver id', 'timestamp'])
gps_gdf['hour'] = gps_gdf['timestamp'].dt.hour

print("Total GPS rows:", len(gps_gdf))

# ============================================================
# 3. LOAD DRAWN ROUTES (FROM GEOJSON)
# ============================================================

print("Loading drawn routes...")

with open("drawn_routes.geojson") as f:
    geojson = json.load(f)

manual_routes = {}

for i, feature in enumerate(geojson["features"]):
    coords = feature["geometry"]["coordinates"]
    
    # Snap the drawn points (Lat/Lon) to the Metric Graph
    snapped_nodes = []
    for lon, lat in coords:
        # Project single point to metric to snap correctly
        pt_metric = gpd.GeoSeries(
            [Point(lon, lat)], crs="EPSG:4326"
        ).to_crs(PROJECT_CRS)
        
        x, y = pt_metric.iloc[0].x, pt_metric.iloc[0].y
        snapped_nodes.append(snap_node(x, y))

    # Reconstruct path along the road network
    full_path_nodes = []
    for j in range(len(snapped_nodes)-1):
        try:
            path = nx.shortest_path(
                G, snapped_nodes[j], snapped_nodes[j+1], weight='weight'
            )
            full_path_nodes.extend(path)
        except:
            continue
            
    manual_routes[i] = list(set(full_path_nodes))

# ============================================================
# 4. COMBINE ROUTES (R0=0, R1=1+2+3, R2=4)
# ============================================================

final_routes = {
    0: set(manual_routes.get(0, [])) | set(manual_routes.get(1, [])),
    1: set(manual_routes.get(2, [])) | set(manual_routes.get(3, [])) | set(manual_routes.get(4, [])) | set(manual_routes.get(5, [])),
    2: set(manual_routes.get(6, [])) | set(manual_routes.get(7, [])) | set(manual_routes.get(8, [])) | set(manual_routes.get(9, []))
}

# ============================================================
# 5. SPATIAL COVERAGE (STRICT 20m BUFFER)
# ============================================================

print(f"\n===== CALCULATING SPATIAL COVERAGE ({BUFFER_DISTANCE}m) =====")

route_buffers = {}

for r_id, nodes in final_routes.items():
    subG = G.subgraph(nodes)
    
    # Create a list of LineStrings from edges
    lines = []
    for u, v in subG.edges():
        lines.append(LineString([u, v]))
        
    if lines:
        # Union all lines and buffer
        route_buffers[r_id] = unary_union(lines).buffer(BUFFER_DISTANCE)
    else:
        route_buffers[r_id] = None

# Assign Points
def assign_route(pt):
    for r_id, buffer in route_buffers.items():
        if buffer and buffer.contains(pt):
            return r_id
    return np.nan

gps_gdf['final_route'] = gps_gdf.geometry.apply(assign_route)

# Print Stats
total = len(gps_gdf)
counts = gps_gdf['final_route'].value_counts(dropna=False)
uncovered = gps_gdf['final_route'].isna().sum()

print(f"Total Points: {total}")
for r in [0, 1, 2]:
    c = counts.get(r, 0)
    print(f"Route {r}: {c} ({(c/total)*100:.2f}%)")

print(f"Uncovered: {uncovered} ({(uncovered/total)*100:.2f}%)")

# ============================================================
# 6. DRIVER ALLOCATION (MULTI-TIMEFRAME)
# ============================================================

print("\n========== DRIVER ALLOCATION ==========")

# Only allocate covered points
covered_gps = gps_gdf.dropna(subset=['final_route']).copy()
print(f"Allocating for {len(covered_gps)} covered points...")

# Function to generate time slot strings (HH:MM)
def get_time_slot(dt_series, interval_minutes):
    # Convert timestamp to "minute of the day" (0 - 1439)
    minutes_of_day = dt_series.dt.hour * 60 + dt_series.dt.minute
    
    # Floor to nearest interval
    slot_start_min = (minutes_of_day // interval_minutes) * interval_minutes
    
    # Format back to HH:MM
    h = (slot_start_min // 60).astype(str).str.zfill(2)
    m = (slot_start_min % 60).astype(str).str.zfill(2)
    return h + ":" + m

# List of intervals to process
time_intervals = [60, 30, 15, 5]

for mins in time_intervals:
    print(f"\nProcessing {mins}-minute intervals...")
    
    # 1. Create Time Slot Column
    slot_col = 'time_slot'
    covered_gps[slot_col] = get_time_slot(covered_gps['timestamp'], mins)
    
    # 2. Group by Time Slot + Route
    alloc_df = covered_gps.groupby([slot_col, 'final_route']).size().reset_index(name='demand')
    
    # 3. Calculate Total Demand per Slot (for Ratio)
    alloc_df['total_slot_demand'] = alloc_df.groupby(slot_col)['demand'].transform('sum')
    
    # 4. Allocate Drivers
    alloc_df['allocated_drivers'] = (
        TOTAL_DRIVERS * alloc_df['demand'] / 
        alloc_df['total_slot_demand']
    ).round()
    
    # 5. Save to Excel
    out_file = f"final_manual_allocation_{mins}min.xlsx"
    alloc_df.to_excel(out_file, index=False)
    print(f"Saved: {out_file}")

print("All allocations completed.")

# ============================================================
# 7. VISUALIZATION (ROBUST METHOD)
# ============================================================

print("\nCreating Visualization...")

# Convert Data to WGS84 for Map
gps_wgs = gps_gdf.to_crs(epsg=4326)
# Center map
center_lat = gps_wgs.geometry.y.mean()
center_lon = gps_wgs.geometry.x.mean()

m = folium.Map(location=[center_lat, center_lon], zoom_start=15)

route_colors = {0: "blue", 1: "orange", 2: "green"}

# --- DRAW ROUTES ---
for r_id, nodes in final_routes.items():
    subG = G.subgraph(nodes)
    
    # Collect edges as separate LineStrings
    edge_geoms = []
    for u, v in subG.edges():
        edge_geoms.append(LineString([u, v]))
    
    if not edge_geoms:
        continue
        
    # Create GeoDataFrame in Metric, convert to WGS84
    edges_gdf = gpd.GeoDataFrame(geometry=edge_geoms, crs=PROJECT_CRS).to_crs(epsg=4326)
    
    # Add each edge to map
    for _, row in edges_gdf.iterrows():
        coords = [(pt[1], pt[0]) for pt in row.geometry.coords]
        folium.PolyLine(
            coords, 
            color=route_colors.get(r_id, "gray"), 
            weight=5, 
            opacity=0.8
        ).add_to(m)

# --- DRAW GPS SAMPLE ---
sample = gps_wgs.sample(min(5000, len(gps_wgs)))

for _, row in sample.iterrows():
    r = row['final_route']
    
    if pd.isna(r):
        color = "black"
        opacity = 0.3
    else:
        color = route_colors.get(r, "black")
        opacity = 0.6
        
    folium.CircleMarker(
        location=[row.geometry.y, row.geometry.x],
        radius=2,
        color=color,
        fill=True,
        fill_opacity=opacity
    ).add_to(m)

m.save("final_manual_routes_vis.html")
print("Map saved: final_manual_routes_vis.html")
print("All Done.")