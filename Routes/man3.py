# ============================================================
# IITB MANUAL ROUTE SYSTEM (FINAL FIXED VERSION)
# FIXED: METRIC CRS + ROBUST VISUALIZATION + BUFFERING + 3D DEMAND PLOT
# ============================================================

import pandas as pd
import geopandas as gpd
import numpy as np
import networkx as nx
import folium
import json
import glob
import warnings
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from shapely.geometry import Point, LineString
from shapely.ops import unary_union
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")

# ============================================================
# SETTINGS
# ============================================================

PROJECT_CRS = "EPSG:32643"  # UTM Zone 43N (Meters)
BUFFER_DISTANCE = 20        # 20 Meters tolerance
CHECKPOINT_SNAP_DIST = 50   # Max dist to snap GPS to a stop (meters)
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

gps_data['mapmatched lat'] = pd.to_numeric(gps_data['mapmatched lat'], errors='coerce')
gps_data['mapmatched lon'] = pd.to_numeric(gps_data['mapmatched lon'], errors='coerce')
gps_data = gps_data.dropna(subset=['mapmatched lat', 'mapmatched lon'])

gps_data['geometry'] = gps_data.apply(
    lambda r: Point(r['mapmatched lon'], r['mapmatched lat']), 
    axis=1
)
gps_gdf = gpd.GeoDataFrame(gps_data, geometry='geometry', crs="EPSG:4326")
gps_gdf = gps_gdf.to_crs(PROJECT_CRS)

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
    snapped_nodes = []
    for lon, lat in coords:
        pt_metric = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(PROJECT_CRS)
        x, y = pt_metric.iloc[0].x, pt_metric.iloc[0].y
        snapped_nodes.append(snap_node(x, y))

    full_path_nodes = []
    for j in range(len(snapped_nodes)-1):
        try:
            path = nx.shortest_path(G, snapped_nodes[j], snapped_nodes[j+1], weight='weight')
            full_path_nodes.extend(path)
        except:
            continue
            
    manual_routes[i] = list(set(full_path_nodes))

# ============================================================
# 4. COMBINE ROUTES
# ============================================================

final_routes = {
    0: set(manual_routes.get(0, [])) | set(manual_routes.get(1, [])),
    1: set(manual_routes.get(2, [])) | set(manual_routes.get(3, [])) | set(manual_routes.get(4, [])) | set(manual_routes.get(5, [])),
    2: set(manual_routes.get(6, [])) | set(manual_routes.get(7, [])) | set(manual_routes.get(8, [])) | set(manual_routes.get(9, []))
}

# ============================================================
# 5. SPATIAL COVERAGE & ASSIGNMENT
# ============================================================

print(f"\n===== CALCULATING SPATIAL COVERAGE ({BUFFER_DISTANCE}m) =====")

route_buffers = {}

for r_id, nodes in final_routes.items():
    subG = G.subgraph(nodes)
    lines = [LineString([u, v]) for u, v in subG.edges()]
        
    if lines:
        route_buffers[r_id] = unary_union(lines).buffer(BUFFER_DISTANCE)
    else:
        route_buffers[r_id] = None

def assign_route(pt):
    for r_id, buffer in route_buffers.items():
        if buffer and buffer.contains(pt):
            return r_id
    return np.nan

gps_gdf['final_route'] = gps_gdf.geometry.apply(assign_route)

# ============================================================
# 6. ASSIGN CHECKPOINTS TO ROUTES
# ============================================================

print("\n===== ASSIGNING CHECKPOINTS TO ROUTES =====")

# Load Checkpoints
checkpoints_df = pd.read_csv("Checkpoints.csv")
checkpoints_df['geometry'] = checkpoints_df.apply(lambda r: Point(r['Lon'], r['Lat']), axis=1)
checkpoints_gdf = gpd.GeoDataFrame(checkpoints_df, geometry='geometry', crs="EPSG:4326").to_crs(PROJECT_CRS)

# Assign Checkpoint to nearest Route (if inside buffer)
def get_checkpoint_route(pt):
    for r_id, buffer in route_buffers.items():
        if buffer and buffer.contains(pt):
            return r_id
    return np.nan

checkpoints_gdf['route_id'] = checkpoints_gdf.geometry.apply(get_checkpoint_route)
print(checkpoints_gdf[['Checkpoint', 'route_id']].dropna())

# ============================================================
# 7. SNAP GPS TO NEAREST CHECKPOINT
# ============================================================

print("\n===== CALCULATING DEMAND PER CHECKPOINT =====")

gps_gdf['nearest_checkpoint'] = np.nan

# Process each route separately
for r_id in final_routes.keys():
    # 1. Get GPS points on this route
    route_gps_mask = gps_gdf['final_route'] == r_id
    if not route_gps_mask.any(): continue
    
    # 2. Get Checkpoints on this route
    route_checkpoints = checkpoints_gdf[checkpoints_gdf['route_id'] == r_id]
    if route_checkpoints.empty: continue
    
    # 3. Build Tree for Snap
    ckpt_coords = np.array([(p.x, p.y) for p in route_checkpoints.geometry])
    ckpt_names = route_checkpoints['Checkpoint'].values
    ckpt_tree = cKDTree(ckpt_coords)
    
    # 4. Query GPS points
    gps_coords = np.array([(p.x, p.y) for p in gps_gdf.loc[route_gps_mask, 'geometry']])
    dists, idxs = ckpt_tree.query(gps_coords)
    
    # 5. Assign (Only if within reasonable stop distance, e.g. 100m)
    # If it's too far from a checkpoint, it's "in-transit"
    valid_snaps = dists < CHECKPOINT_SNAP_DIST * 2 # Looser tolerance for "near a stop"
    
    # Map index back to name
    assigned_names = np.array([np.nan] * len(gps_coords), dtype=object)
    assigned_names[valid_snaps] = ckpt_names[idxs[valid_snaps]]
    
    gps_gdf.loc[route_gps_mask, 'nearest_checkpoint'] = assigned_names

# Aggregate Demand
demand_df = gps_gdf.groupby(['final_route', 'nearest_checkpoint', 'hour']).size().reset_index(name='demand')

# ============================================================
# 8. 3D VISUALIZATION
# ============================================================

print("\n===== GENERATING 3D PLOTS =====")

for r_id in final_routes.keys():
    subset = demand_df[demand_df['final_route'] == r_id]
    if subset.empty: continue
    
    # Get unique checkpoints and hours
    checkpoints = sorted(subset['nearest_checkpoint'].unique())
    hours = np.arange(24)
    
    # Create Grid
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Map checkpoint names to Y-index
    ckpt_map = {name: i for i, name in enumerate(checkpoints)}
    
    # Prepare Data for Bar3d
    x_data = [] # Time
    y_data = [] # Checkpoint Index
    z_data = [] # Base Z (0)
    dx = []     # Width
    dy = []     # Depth
    dz = []     # Height (Demand)
    
    for _, row in subset.iterrows():
        x_data.append(row['hour'])
        y_data.append(ckpt_map[row['nearest_checkpoint']])
        z_data.append(0)
        dx.append(0.6) # Bar width
        dy.append(0.6) # Bar depth
        dz.append(row['demand'])
        
    # Plot
    colors = plt.cm.viridis(np.array(dz) / max(dz)) # Color by height
    ax.bar3d(x_data, y_data, z_data, dx, dy, dz, color=colors, alpha=0.8)
    
    # Formatting
    ax.set_title(f"3D Demand Profile - Route {r_id}", fontsize=14, pad=20)
    ax.set_xlabel("Time (Hour of Day)")
    ax.set_ylabel("Checkpoint")
    ax.set_zlabel("Passenger Demand")
    
    # Set Y-ticks to Checkpoint Names
    ax.set_yticks(np.arange(len(checkpoints)) + 0.3)
    ax.set_yticklabels(checkpoints, rotation=-15, ha='left', fontsize=9)
    
    ax.set_xticks(range(0, 24, 2))
    
    plt.tight_layout()
    filename = f"3d_demand_route_{r_id}.png"
    plt.savefig(filename)
    print(f"Saved: {filename}")

print("\nAll Done.")