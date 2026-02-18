# ============================================================
# IITB INTERACTIVE 3D VISUALIZER (NAMED CHECKPOINTS EDITION)
# FEATURES: Checkpoint Names on Axis, No Distances, Clear Labels
# ============================================================

import pandas as pd
import geopandas as gpd
import numpy as np
import networkx as nx
import json
import glob
import warnings
import plotly.graph_objects as go
from shapely.geometry import Point, LineString, MultiLineString
from shapely.ops import linemerge
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter

warnings.filterwarnings("ignore")

# ============================================================
# SETTINGS
# ============================================================

PROJECT_CRS = "EPSG:32643"  # UTM Zone 43N (Meters)
BUFFER_DISTANCE = 30        # Buffer to catch GPS points
CHECKPOINT_SNAP = 80        # Tolerance to find stops
GRID_RES_METERS = 30        # Finer resolution for smoother bumps
GRID_RES_HOURS = 0.5        # 30-minute time slots

# ============================================================
# 1. SETUP & LOAD DATA
# ============================================================

print("Loading Data...")
roads = gpd.read_file("../map/IITB_Auto_POV_Shapefile.gpkg").to_crs(PROJECT_CRS)
checkpoints_df = pd.read_csv("Checkpoints.csv")

# Build Graph
G = nx.Graph()
for _, row in roads.iterrows():
    if row.geometry is None: continue
    lines = row.geometry.geoms if row.geometry.geom_type == "MultiLineString" else [row.geometry]
    for line in lines:
        coords = list(line.coords)
        for i in range(len(coords)-1):
            x1, y1 = coords[i]
            x2, y2 = coords[i+1]
            dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            G.add_edge((x1,y1), (x2,y2), weight=dist)

# Global Node Snapper
ALL_NODES = np.array(list(G.nodes))
TREE = cKDTree(ALL_NODES)

def snap(x, y):
    if np.isnan(x) or np.isnan(y): return ALL_NODES[0]
    _, idx = TREE.query([x, y])
    return tuple(ALL_NODES[idx])

# ============================================================
# 2. PROCESS GPS & ROUTES
# ============================================================

print("Processing GPS & Routes...")
files = glob.glob("./data/*.csv")
gps_df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

# Clean GPS
gps_df['mapmatched lat'] = pd.to_numeric(gps_df['mapmatched lat'], errors='coerce')
gps_df['mapmatched lon'] = pd.to_numeric(gps_df['mapmatched lon'], errors='coerce')
gps_df = gps_df.dropna(subset=['mapmatched lat', 'mapmatched lon'])

gps_gdf = gpd.GeoDataFrame(
    gps_df, 
    geometry=gpd.points_from_xy(gps_df['mapmatched lon'], gps_df['mapmatched lat']),
    crs="EPSG:4326"
).to_crs(PROJECT_CRS)

gps_gdf['timestamp'] = pd.to_datetime(
    gps_gdf['date'].astype(str) + " " + gps_gdf['ist time'].astype(str),
    dayfirst=True, errors='coerce'
)
gps_gdf['hour_float'] = gps_gdf['timestamp'].dt.hour + gps_gdf['timestamp'].dt.minute / 60.0

# Linearize Routes
with open("drawn_routes.geojson") as f:
    geojson = json.load(f)

def linearize(node_list):
    if len(node_list) < 2: return None
    subG = G.subgraph(node_list)
    edges = [LineString([u, v]) for u, v in subG.edges()]
    if not edges: return None
    merged = linemerge(edges)
    if isinstance(merged, MultiLineString):
        merged = max(merged.geoms, key=lambda x: x.length)
    return merged

# Reconstruct Routes
manual_routes = {}
for i, feat in enumerate(geojson["features"]):
    coords = feat["geometry"]["coordinates"]
    pts = [gpd.GeoSeries([Point(c)], crs="EPSG:4326").to_crs(PROJECT_CRS).iloc[0] for c in coords]
    snapped = [snap(p.x, p.y) for p in pts]
    path_nodes = []
    for j in range(len(snapped)-1):
        try:
            path_nodes.extend(nx.shortest_path(G, snapped[j], snapped[j+1], weight='weight'))
        except: continue
    manual_routes[i] = list(set(path_nodes))

# Combine
final_route_nodes = {
    0: set(manual_routes.get(0, [])) | set(manual_routes.get(1, [])),
    1: set(manual_routes.get(2, [])) | set(manual_routes.get(3, [])) | set(manual_routes.get(4, [])) | set(manual_routes.get(5, [])),
    2: set(manual_routes.get(6, [])) | set(manual_routes.get(7, [])) | set(manual_routes.get(8, [])) | set(manual_routes.get(9, []))
}

# ============================================================
# 3. GENERATE CUSTOM AXIS PLOTS
# ============================================================

print("\nGenerating Named 3D Plots...")

# Prepare Checkpoints
checkpoints_gdf = gpd.GeoDataFrame(
    checkpoints_df, 
    geometry=gpd.points_from_xy(checkpoints_df['Lon'], checkpoints_df['Lat']),
    crs="EPSG:4326"
).to_crs(PROJECT_CRS)

for r_id, nodes in final_route_nodes.items():
    line = linearize(list(nodes))
    if not line: continue
    
    print(f"  Route {r_id}...")
    
    # 1. Project Data
    route_buffer = line.buffer(BUFFER_DISTANCE)
    subset = gps_gdf[gps_gdf.geometry.within(route_buffer)].copy()
    if subset.empty: continue
    
    subset['dist_along'] = subset.geometry.apply(lambda p: line.project(p))
    
    # 2. Get Checkpoints & Sort by Distance
    ckpt_subset = checkpoints_gdf[checkpoints_gdf.geometry.distance(line) < CHECKPOINT_SNAP].copy()
    
    tick_vals = []
    tick_text = []
    
    if not ckpt_subset.empty:
        ckpt_subset['dist_along'] = ckpt_subset.geometry.apply(lambda p: line.project(p))
        ckpt_subset = ckpt_subset.sort_values('dist_along')
        
        # Deduplicate very close stops for labels (keep first)
        ckpt_subset['cluster'] = (ckpt_subset['dist_along'] / 150).round().astype(int)
        ckpt_unique = ckpt_subset.drop_duplicates('cluster')
        
        tick_vals = ckpt_unique['dist_along'].tolist()
        tick_text = ckpt_unique['Checkpoint'].tolist()

    # 3. Create Grid & Density
    x_grid = np.arange(0, 24, GRID_RES_HOURS)
    y_grid = np.arange(0, line.length, GRID_RES_METERS)
    
    H, _, _ = np.histogram2d(
        subset['hour_float'], 
        subset['dist_along'], 
        bins=[x_grid, y_grid]
    )
    
    # Smooth for "Mountain" look
    Z = gaussian_filter(H.T, sigma=1.2)
    
    # 4. Plotly Figure
    fig = go.Figure()

    # Add Surface
    fig.add_trace(go.Surface(
        z=Z,
        x=x_grid[:-1],
        y=y_grid[:-1],
        colorscale='Viridis',
        opacity=0.9,
        showscale=True,
        colorbar=dict(title='Passenger Density')
    ))

    # Add Checkpoint Lines (White lines across time)
    for pos, name in zip(tick_vals, tick_text):
        fig.add_trace(go.Scatter3d(
            x=[0, 24], y=[pos, pos], z=[0, 0],
            mode='lines',
            line=dict(color='white', width=3, dash='dot'),
            hoverinfo='text',
            text=f"Stop: {name}",
            showlegend=False
        ))

    # 5. CUSTOM AXIS LAYOUT (The Key Fix)
    fig.update_layout(
        title=f"Route {r_id} Demand Profile",
        scene=dict(
            xaxis=dict(title='Time (Hour)', tickmode='linear', dtick=2),
            
            # Y-AXIS: FORCE TEXT LABELS
            yaxis=dict(
                title='',  # Hide generic title
                tickmode='array',
                tickvals=tick_vals,
                ticktext=tick_text,
                tickfont=dict(size=11, color='black'),
                backgroundcolor="rgb(230, 230,230)"
            ),
            
            zaxis=dict(title='Demand Density'),
            camera=dict(eye=dict(x=-1.8, y=-0.5, z=0.8)) # Side view for readability
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    filename = f"named_demand_route_{r_id}.html"
    fig.write_html(filename)
    print(f"Saved: {filename}")

print("\nDone. Open HTML files to interact.")