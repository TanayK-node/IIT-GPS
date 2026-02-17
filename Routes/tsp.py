# ============================================================
# IITB AUTO ROUTE OPTIMIZATION V3 (FINAL FULL VERSION)
# 7-DAY DEMAND + BACKBONE + CONNECTED ROUTES
# + INTERACTIVE MAP + EXCEL DRIVER OUTPUT
# ============================================================

import pandas as pd
import geopandas as gpd
import numpy as np
import networkx as nx
from shapely.geometry import Point, LineString
from scipy.spatial import cKDTree
import glob
import folium
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# USER SETTINGS
# ============================================================

GPS_FOLDER = "./data/"     # Folder with 7 CSV files
TOTAL_DRIVERS = 60
EDGE_PERCENTILE = 65       # Top 30% demand edges covered

# ============================================================
# 1. LOAD ROAD + CHECKPOINT DATA
# ============================================================

print("Loading roads and checkpoints...")

roads = gpd.read_file("../map/IITB_Auto_POV_Shapefile.gpkg")
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

checkpoints_gdf = gpd.GeoDataFrame(
    checkpoints, geometry='geometry', crs="EPSG:4326"
)

checkpoints_gdf = checkpoints_gdf.to_crs(roads.crs)

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
# 5. PROCESS GPS  (FIXED VERSION – NO DATA LOSS)
# ============================================================

print("\n========== GPS PROCESSING ==========")
print("After concat:", len(gps_data))

# ------------------------------------------------------------
# Ensure coordinates are numeric
# ------------------------------------------------------------
gps_data['mapmatched lat'] = pd.to_numeric(
    gps_data['mapmatched lat'], errors='coerce'
)

gps_data['mapmatched lon'] = pd.to_numeric(
    gps_data['mapmatched lon'], errors='coerce'
)

# Drop ONLY rows where coordinates are truly missing
gps_data = gps_data.dropna(subset=['mapmatched lat','mapmatched lon'])

print("After coordinate cleaning:", len(gps_data))

# ------------------------------------------------------------
# Create geometry
# ------------------------------------------------------------
gps_data['geometry'] = gps_data.apply(
    lambda r: Point(r['mapmatched lon'], r['mapmatched lat']),
    axis=1
)

gps_gdf = gpd.GeoDataFrame(
    gps_data,
    geometry='geometry',
    crs="EPSG:4326"
)

print("After GeoDataFrame creation:", len(gps_gdf))

# ------------------------------------------------------------
# Convert CRS
# ------------------------------------------------------------
gps_gdf = gps_gdf.to_crs(roads.crs)

print("After CRS transform:", len(gps_gdf))

# ------------------------------------------------------------
# Timestamp Parsing (SAFE + EXPLICIT FORMAT)
# ------------------------------------------------------------
print(gps_data[['date','ist time']].head(20))

# NEW CODE (Fix)
gps_gdf['timestamp'] = pd.to_datetime(
    gps_gdf['date'].astype(str) + " " + gps_gdf['ist time'].astype(str),
    # Removed the strict format argument
    dayfirst=True,   # Helps resolve ambiguity (e.g. 02-07 vs 07-02)
    errors='coerce'
)
nat_count = gps_gdf['timestamp'].isna().sum()

print("NaT timestamps:", nat_count)
print("Rows BEFORE dropping invalid timestamps:", len(gps_gdf))

# ------------------------------------------------------------
# Drop ONLY if actually invalid
# ------------------------------------------------------------
if nat_count > 0:
    gps_gdf = gps_gdf.dropna(subset=['timestamp'])
    print("Rows AFTER dropping invalid timestamps:", len(gps_gdf))
else:
    print("No timestamp drops required.")

# ------------------------------------------------------------
# Sort + Hour extraction
# ------------------------------------------------------------
gps_gdf = gps_gdf.sort_values(['driver id','timestamp'])

gps_gdf['hour'] = gps_gdf['timestamp'].dt.hour

print("Final GPS rows after processing:", len(gps_gdf))

# ------------------------------------------------------------
# Snap to Graph
# ------------------------------------------------------------
gps_gdf['graph_node'] = gps_gdf.apply(
    lambda r: snap_node(r.geometry.x, r.geometry.y),
    axis=1
)

print("GPS processing complete.")
print("====================================\n")

   
# ============================================================
# 6. COMPUTE EDGE DEMAND (WEEK AGGREGATED)
# ============================================================

print("Computing edge demand...")

for driver, df in gps_gdf.groupby('driver id'):

    nodes = df['graph_node'].tolist()

    for i in range(len(nodes)-1):
        try:
            path = nx.shortest_path(G, nodes[i], nodes[i+1], weight='weight')

            for j in range(len(path)-1):
                u = path[j]
                v = path[j+1]
                if G.has_edge(u,v):
                    G[u][v]['demand'] += 1

        except:
            continue

print("Edge demand complete.")

# ============================================================
# 7. DEFINE BACKBONE
# ============================================================

mg_node = checkpoints_gdf[
    checkpoints_gdf['Checkpoint']=='Main Gate'
]['nearest_node'].values[0]

h12_node = checkpoints_gdf[
    checkpoints_gdf['Checkpoint']=='Main road-h12'
]['nearest_node'].values[0]

backbone_path = nx.shortest_path(G, mg_node, h12_node, weight='weight')

backbone_edges = set(
    (backbone_path[i], backbone_path[i+1])
    for i in range(len(backbone_path)-1)
)

print("Backbone created.")

# ============================================================
# 8. HIGH DEMAND EDGES
# ============================================================

edge_demands = [G[u][v]['demand'] for u,v in G.edges()]
threshold = np.percentile(edge_demands, EDGE_PERCENTILE)

high_edges = set(
    (u,v) for u,v in G.edges()
    if G[u][v]['demand'] >= threshold
)

print("High demand edges selected.")

# ============================================================
# 9. BUILD ROUTES
# ============================================================

route1_edges = backbone_edges.copy()

for u,v in high_edges:
    if u in backbone_path or v in backbone_path:
        route1_edges.add((u,v))

route1_nodes = set()
for u,v in route1_edges:
    route1_nodes.update([u,v])

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
# 10. DRIVER ALLOCATION (AVERAGE PER HOUR)
# ============================================================

cp_coords = np.array(
    list(zip(checkpoints_gdf.geometry.x,
             checkpoints_gdf.geometry.y))
)

cp_tree = cKDTree(cp_coords)

gps_coords = np.array(
    list(zip(gps_gdf.geometry.x,
             gps_gdf.geometry.y))
)

_, idx = cp_tree.query(gps_coords, k=1)

gps_gdf['assigned_checkpoint'] = checkpoints_gdf.iloc[idx]['Checkpoint'].values

cp_route_map = {}

for r_id, nodes in routes.items():
    for node in nodes:
        match = checkpoints_gdf[
            checkpoints_gdf['nearest_node']==node
        ]
        if len(match)>0:
            cp_route_map[match['Checkpoint'].values[0]] = r_id

gps_gdf['route'] = gps_gdf['assigned_checkpoint'].map(cp_route_map)

hourly = gps_gdf.groupby(['hour','route']).size().reset_index(name='demand')

hourly['avg_demand'] = hourly['demand'] / 7
hourly['total_hour'] = hourly.groupby('hour')['avg_demand'].transform('sum')

hourly['allocated_drivers'] = (
    TOTAL_DRIVERS *
    hourly['avg_demand'] /
    hourly['total_hour']
).round()

hourly[['hour','route','avg_demand','allocated_drivers']].to_excel(
    "iitb_driver_allocation.xlsx", index=False
)

print("Excel saved: iitb_driver_allocation.xlsx")

# ============================================================
# ROUTE COVERAGE STATISTICS
# ============================================================

print("\n========== ROUTE COVERAGE STATS ==========")

# Collect route nodes
route_node_sets = {
    r_id: set(nodes)
    for r_id, nodes in routes.items()
}

# Map checkpoint -> route
checkpoint_route_map = {}

for r_id, node_set in route_node_sets.items():
    for node in node_set:
        matches = checkpoints_gdf[
            checkpoints_gdf['nearest_node'] == node
        ]
        for cp in matches['Checkpoint']:
            checkpoint_route_map[cp] = r_id

# Compute stats
total_checkpoints = len(checkpoints_gdf)

route_counts = {}
for r_id in routes.keys():
    route_counts[r_id] = list(checkpoint_route_map.values()).count(r_id)

covered_checkpoints = set(checkpoint_route_map.keys())
all_checkpoints = set(checkpoints_gdf['Checkpoint'])

uncovered_checkpoints = all_checkpoints - covered_checkpoints

# Print stats
print(f"Total checkpoints: {total_checkpoints}\n")

for r_id in routes.keys():
    print(f"Route {r_id+1} checkpoint count: {route_counts[r_id]}")

print(f"\nUncovered checkpoints: {len(uncovered_checkpoints)}")

if len(uncovered_checkpoints) > 0:
    print("\nList of uncovered checkpoints:")
    for cp in uncovered_checkpoints:
        print("-", cp)

print("\n===========================================")
# ============================================================
# SIMPLE GPS POINT COVERAGE STATS (7-DAY REAL DATA)
# ============================================================

print("\n========== GPS POINT COVERAGE ==========")

# Convert route node lists to sets
route_node_sets = {
    r_id: set(nodes)
    for r_id, nodes in routes.items()
}

# Assign each GPS point to route if its graph node is in route
def assign_route(node):
    for r_id, node_set in route_node_sets.items():
        if node in node_set:
            return r_id
    return None

gps_gdf['route_assigned'] = gps_gdf['graph_node'].apply(assign_route)

total_points = len(gps_gdf)

route_counts = gps_gdf['route_assigned'].value_counts(dropna=False)

print(f"Total GPS points (7 days): {total_points}\n")

for r_id in routes.keys():
    count = route_counts.get(r_id, 0)
    percent = (count / total_points) * 100
    print(f"Route {r_id} points: {count} ({percent:.2f}%)")

uncovered = route_counts.get(None, 0)
uncovered_percent = (uncovered / total_points) * 100

print(f"\nUncovered GPS points: {uncovered} ({uncovered_percent:.2f}%)")

print("========================================")

# ============================================================
# VISUALIZE REAL GPS (ALL FILES) + ROUTES
# ============================================================

print("Creating full GPS visualization (real data)...")

from folium.plugins import MarkerCluster

# Convert everything to WGS84
roads_wgs = roads.to_crs(epsg=4326)
checkpoints_wgs = checkpoints_gdf.to_crs(epsg=4326)
gps_wgs = gps_gdf.to_crs(epsg=4326)

# Center map
center = [
    checkpoints_wgs.geometry.y.mean(),
    checkpoints_wgs.geometry.x.mean()
]

m = folium.Map(location=center, zoom_start=15)

route_colors = {
    0: "red",
    1: "blue",
    2: "green"
}

# -----------------------------
# DRAW ROUTES
# -----------------------------
for route_id, route_nodes in routes.items():

    subG = G.subgraph(route_nodes)
    full_path = []

    for comp in nx.connected_components(subG):
        comp = list(comp)
        for i in range(len(comp)-1):
            try:
                path = nx.shortest_path(G, comp[i], comp[i+1], weight='weight')
                full_path.extend(path)
            except:
                continue

    if len(full_path) < 2:
        continue

    route_line = gpd.GeoDataFrame(
        geometry=[LineString(full_path)],
        crs=roads.crs
    ).to_crs(epsg=4326)

    coords_latlon = [(pt[1], pt[0]) for pt in route_line.geometry.iloc[0].coords]

    folium.PolyLine(
        coords_latlon,
        color=route_colors.get(route_id, "gray"),
        weight=6,
        tooltip=f"Route {route_id+1}"
    ).add_to(m)

# -----------------------------
# DRAW GPS POINTS (ALL REAL DATA)
# -----------------------------

marker_cluster = MarkerCluster().add_to(m)

for _, row in gps_wgs.iterrows():

    route_id = row.get("route_assigned")

    if pd.isna(route_id):
        color = "black"  # Uncovered
    else:
        color = route_colors.get(route_id, "gray")

    folium.CircleMarker(
        location=[row.geometry.y, row.geometry.x],
        radius=2,
        color=color,
        fill=True,
        fill_opacity=0.7
    ).add_to(marker_cluster)

# -----------------------------
# DRAW CHECKPOINTS
# -----------------------------
for _, row in checkpoints_wgs.iterrows():
    folium.CircleMarker(
        location=[row.geometry.y, row.geometry.x],
        radius=5,
        color="white",
        fill=True,
        fill_opacity=1,
        popup=row['Checkpoint']
    ).add_to(m)

m.save("iitb_routes_with_real_gps_v3.html")

print("Map saved as: iitb_routes_with_real_gps_v3.html")

print("All Done.")

