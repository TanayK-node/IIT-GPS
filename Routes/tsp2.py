# ============================================================
# IITB AUTO ROUTE OPTIMIZATION V3 (FINAL FIXED VERSION)
# FIXED: METRIC CRS + STRICT BUFFERING + MULTI-INTERVAL ALLOCATION
# ============================================================

import pandas as pd
import geopandas as gpd
import numpy as np
import networkx as nx
import glob
import folium
import warnings
from shapely.geometry import Point, LineString
from shapely.ops import unary_union
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")

# ============================================================
# USER SETTINGS
# ============================================================

GPS_FOLDER = "./data/"     # Folder with 7 CSV files
TOTAL_DRIVERS = 60
EDGE_PERCENTILE = 65       # Top demand edges covered
BUFFER_DISTANCE = 20       # 20 meters tolerance
PROJECT_CRS = "EPSG:32643" # UTM Zone 43N (Meters)

# ============================================================
# 1. LOAD ROAD + CHECKPOINT DATA
# ============================================================

print("Loading roads and checkpoints...")

# Load Roads & Convert to Meters immediately
roads = gpd.read_file("../map/IITB_Auto_POV_Shapefile.gpkg")
roads = roads.to_crs(PROJECT_CRS)

# Load Checkpoints
checkpoints = pd.read_csv("Checkpoints.csv")
checkpoints.columns = checkpoints.columns.str.strip()

# ============================================================
# 2. BUILD GRAPH
# ============================================================

print("Building graph...")

roads = roads.explode(index_parts=False)

G = nx.Graph()

for _, row in roads.iterrows():
    geom = row.geometry
    if geom is None or geom.geom_type != "LineString":
        continue

    # Simplify geometry to edges
    coords = list(geom.coords)
    for i in range(len(coords)-1):
        x1,y1 = coords[i]
        x2,y2 = coords[i+1]
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        G.add_edge((x1,y1),(x2,y2), weight=length, demand=0)

print("Graph built:", len(G.nodes), "nodes")

# ============================================================
# 3. PREPARE CHECKPOINTS
# ============================================================

checkpoints['geometry'] = checkpoints.apply(
    lambda r: Point(r['Lon'], r['Lat']), axis=1
)

# Load as WGS84 then convert to Meters
checkpoints_gdf = gpd.GeoDataFrame(
    checkpoints, geometry='geometry', crs="EPSG:4326"
)
checkpoints_gdf = checkpoints_gdf.to_crs(PROJECT_CRS)

# Snap checkpoints to graph nodes
graph_nodes = np.array(list(G.nodes))
graph_tree = cKDTree(graph_nodes)

def snap_node(x,y):
    _, idx = graph_tree.query([x,y])
    return tuple(graph_nodes[idx])

checkpoints_gdf['nearest_node'] = checkpoints_gdf.apply(
    lambda r: snap_node(r.geometry.x, r.geometry.y),
    axis=1
)

print("Checkpoints snapped.")

# ============================================================
# 4. LOAD ALL 7 GPS FILES
# ============================================================

print("Loading GPS files...")

files = glob.glob(GPS_FOLDER + "*.csv")
gps_list = []

for file in files:
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()
    gps_list.append(df)

gps_data = pd.concat(gps_list, ignore_index=True)

print("Total GPS rows:", len(gps_data))

# ============================================================
# 5. PROCESS GPS
# ============================================================

print("\n========== GPS PROCESSING ==========")

# Coordinate Cleaning
gps_data['mapmatched lat'] = pd.to_numeric(gps_data['mapmatched lat'], errors='coerce')
gps_data['mapmatched lon'] = pd.to_numeric(gps_data['mapmatched lon'], errors='coerce')
gps_data = gps_data.dropna(subset=['mapmatched lat','mapmatched lon'])

# Geometry Creation (WGS84 -> Metric)
gps_data['geometry'] = gps_data.apply(
    lambda r: Point(r['mapmatched lon'], r['mapmatched lat']),
    axis=1
)

gps_gdf = gpd.GeoDataFrame(gps_data, geometry='geometry', crs="EPSG:4326")
gps_gdf = gps_gdf.to_crs(PROJECT_CRS) # Force Metric CRS

# Timestamp Parsing
gps_gdf['timestamp'] = pd.to_datetime(
    gps_gdf['date'].astype(str) + " " + gps_gdf['ist time'].astype(str),
    dayfirst=True,
    errors='coerce'
)
gps_gdf = gps_gdf.dropna(subset=['timestamp'])
gps_gdf = gps_gdf.sort_values(['driver id','timestamp'])
gps_gdf['hour'] = gps_gdf['timestamp'].dt.hour

# Snap for demand calculation (Graph Traversal only)
gps_gdf['graph_node'] = gps_gdf.apply(
    lambda r: snap_node(r.geometry.x, r.geometry.y),
    axis=1
)

print("GPS processing complete. Rows:", len(gps_gdf))

# ============================================================
# 6. COMPUTE EDGE DEMAND (Raw Graph Association)
# ============================================================

print("Computing edge demand...")

for driver, df in gps_gdf.groupby('driver id'):
    nodes = df['graph_node'].tolist()
    for i in range(len(nodes)-1):
        try:
            path = nx.shortest_path(G, nodes[i], nodes[i+1], weight='weight')
            for j in range(len(path)-1):
                u, v = path[j], path[j+1]
                if G.has_edge(u,v):
                    G[u][v]['demand'] += 1
        except:
            continue

print("Edge demand complete.")

# ============================================================
# 7. DEFINE ROUTES
# ============================================================

print("Constructing Routes...")

# Backbone
mg_node = checkpoints_gdf[checkpoints_gdf['Checkpoint']=='Main Gate']['nearest_node'].values[0]
h12_node = checkpoints_gdf[checkpoints_gdf['Checkpoint']=='Main road-h12']['nearest_node'].values[0]
backbone_path = nx.shortest_path(G, mg_node, h12_node, weight='weight')
backbone_edges = set((backbone_path[i], backbone_path[i+1]) for i in range(len(backbone_path)-1))

# High Demand
edge_demands = [G[u][v]['demand'] for u,v in G.edges()]
threshold = np.percentile(edge_demands, EDGE_PERCENTILE)
high_edges = set((u,v) for u,v in G.edges() if G[u][v]['demand'] >= threshold)

# Route 1 construction
route1_edges = backbone_edges.copy()
for u,v in high_edges:
    if u in backbone_path or v in backbone_path:
        route1_edges.add((u,v))

route1_nodes = set()
for u,v in route1_edges:
    route1_nodes.update([u,v])

# Routes 2 & 3
remaining_edges = high_edges - route1_edges
remaining_nodes = set()
for u,v in remaining_edges:
    remaining_nodes.update([u,v])

remaining_graph = G.subgraph(remaining_nodes)
components = list(nx.connected_components(remaining_graph))

routes = {}
routes[0] = list(route1_nodes)

if len(components) >= 2:
    routes[1] = list(components[0])
    routes[2] = list(components[1])
else:
    nodes_list = list(remaining_nodes)
    midpoint = len(nodes_list)//2
    routes[1] = nodes_list[:midpoint]
    routes[2] = nodes_list[midpoint:]

print("Routes constructed.")

# ============================================================
# 8. SPATIAL COVERAGE (STRICT BUFFER)
# ============================================================

print("\n========== CALCULATING SPATIAL COVERAGE ==========")
print(f"Using buffer distance: {BUFFER_DISTANCE} meters")

# Build precise route geometries
route_buffers = {}
for r_id, nodes in routes.items():
    subG = G.subgraph(nodes)
    lines = []
    for u, v in subG.edges():
        lines.append(LineString([u, v]))
    
    if lines:
        # Create metric buffer
        route_buffers[r_id] = unary_union(lines).buffer(BUFFER_DISTANCE)
    else:
        route_buffers[r_id] = None

# Strict assignment
def assign_spatial(point):
    for r_id, buffer in route_buffers.items():
        if buffer and buffer.contains(point):
            return r_id
    return np.nan # Use NaN for easy counting

gps_gdf['spatial_route'] = gps_gdf.geometry.apply(assign_spatial)

# Correct Statistics
total = len(gps_gdf)
counts = gps_gdf['spatial_route'].value_counts(dropna=False)
uncovered = gps_gdf['spatial_route'].isna().sum()

print(f"Total GPS points: {total}")
for r_id in routes.keys():
    c = counts.get(r_id, 0)
    print(f"Route {r_id+1}: {c} ({(c/total)*100:.2f}%)")

print(f"Uncovered: {uncovered} ({(uncovered/total)*100:.2f}%)")


# ============================================================
# 9. DRIVER ALLOCATION (MULTI-TIMEFRAME)
# ============================================================

print("\n========== DRIVER ALLOCATION ==========")

# Filter only covered points for allocation
covered_gps = gps_gdf.dropna(subset=['spatial_route']).copy()
print(f"Allocating for {len(covered_gps)} covered points...")

# Function to generate time slot strings (HH:MM) for any interval
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
    
    # 1. Create Time Slot Column (Aggregates across all days)
    slot_col = 'time_slot'
    covered_gps[slot_col] = get_time_slot(covered_gps['timestamp'], mins)
    
    # 2. Group by Time Slot + Route
    alloc_df = covered_gps.groupby([slot_col, 'spatial_route']).size().reset_index(name='demand')
    
    # 3. Calculate Total Demand per Slot (for Ratio)
    # This sums the demand of all routes in that specific time slot
    alloc_df['total_slot_demand'] = alloc_df.groupby(slot_col)['demand'].transform('sum')
    
    # 4. Allocate Drivers
    # Formula: Total_Fleet * (Route_Demand / Total_Slot_Demand)
    alloc_df['allocated_drivers'] = (
        TOTAL_DRIVERS * alloc_df['demand'] / 
        alloc_df['total_slot_demand']
    ).round()
    
    # 5. Save to Excel
    out_file = f"iitb_driver_allocation_{mins}min.xlsx"
    alloc_df.to_excel(out_file, index=False)
    print(f"Saved: {out_file}")

print("\nAll allocations completed.")

# ============================================================
# 10. VISUALIZATION (FIXED)
# ============================================================

print("\nCreating Visualization...")

# Back to WGS84 for Folium
gps_wgs = gps_gdf.to_crs(epsg=4326)
checkpoints_wgs = checkpoints_gdf.to_crs(epsg=4326)

center = [checkpoints_wgs.geometry.y.mean(), checkpoints_wgs.geometry.x.mean()]
m = folium.Map(location=center, zoom_start=15)

route_colors = {0: "red", 1: "blue", 2: "green"}

# ------------------------------------------------------------
# DRAW ROUTES (FIXED LOGIC)
# ------------------------------------------------------------
for r_id, nodes in routes.items():
    subG = G.subgraph(nodes)
    
    # Collect all edges as individual LineStrings
    edge_geoms = []
    for u, v in subG.edges():
        edge_geoms.append(LineString([u, v]))
    
    if not edge_geoms: 
        continue
    
    # Create a GeoDataFrame of all edges in this route
    route_edges_gdf = gpd.GeoDataFrame(
        geometry=edge_geoms, 
        crs=PROJECT_CRS
    ).to_crs(epsg=4326)
    
    # Plot each edge
    for _, row in route_edges_gdf.iterrows():
        coords = [(pt[1], pt[0]) for pt in row.geometry.coords]
        
        folium.PolyLine(
            coords, 
            color=route_colors.get(r_id, "gray"), 
            weight=5, 
            opacity=0.8
        ).add_to(m)

# ------------------------------------------------------------
# DRAW GPS SAMPLE
# ------------------------------------------------------------
sample = gps_wgs.sample(min(5000, len(gps_wgs)))

for _, row in sample.iterrows():
    r = row['spatial_route']
    
    # Color logic
    if pd.isna(r):
        color = "black"
        opacity = 0.3
    else:
        color = route_colors.get(r, "black")
        opacity = 0.6
    
    folium.CircleMarker(
        [row.geometry.y, row.geometry.x], 
        radius=2, 
        color=color, 
        fill=True,
        fill_opacity=opacity
    ).add_to(m)

m.save("iitb_routes_spatial_final.html")
print("Map saved: iitb_routes_spatial_final.html")
print("Done.")