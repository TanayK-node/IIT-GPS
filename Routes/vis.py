# ============================================================
# IITB ROUTE VISUALIZATION (FULLY SELF-CONTAINED)
# ============================================================

import pandas as pd
import geopandas as gpd
import numpy as np
import networkx as nx
from shapely.geometry import Point, LineString
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree
import folium

print("Loading data...")

# Adjust paths if needed
roads = gpd.read_file("../map/IITB_Auto_POV_Shapefile.gpkg")
checkpoints = pd.read_csv("Checkpoints.csv")
gps_data = pd.read_csv("./data/matched_driver_locations-20250702-110402.csv")

checkpoints.columns = checkpoints.columns.str.strip()
gps_data.columns = gps_data.columns.str.strip()

# ============================================================
# BUILD GRAPH
# ============================================================

print("Building graph...")

roads = roads.explode(index_parts=False)

G = nx.Graph()

for _, row in roads.iterrows():

    geom = row.geometry
    if geom is None or geom.geom_type != "LineString":
        continue

    coords = list(geom.coords)

    for i in range(len(coords) - 1):
        x1, y1 = coords[i]
        x2, y2 = coords[i + 1]

        node1 = (x1, y1)
        node2 = (x2, y2)

        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        G.add_edge(node1, node2, weight=length)

print("Graph built.")
print("Nodes:", len(G.nodes))

# ============================================================
# PREPARE CHECKPOINTS
# ============================================================

checkpoints['geometry'] = checkpoints.apply(
    lambda row: Point(row['Lon'], row['Lat']),
    axis=1
)

checkpoints_gdf = gpd.GeoDataFrame(
    checkpoints,
    geometry='geometry',
    crs="EPSG:4326"
)

# Project to road CRS
checkpoints_gdf = checkpoints_gdf.to_crs(roads.crs)

# Snap to nearest graph node
graph_nodes = np.array(list(G.nodes))
graph_tree = cKDTree(graph_nodes)

def nearest_node(x, y):
    _, idx = graph_tree.query([x, y])
    return tuple(graph_nodes[idx])

checkpoints_gdf['nearest_node'] = checkpoints_gdf.apply(
    lambda row: nearest_node(row.geometry.x, row.geometry.y),
    axis=1
)

# ============================================================
# CLUSTER INTO 3 ROUTES
# ============================================================

coords = np.array(
    list(zip(checkpoints_gdf.geometry.x, checkpoints_gdf.geometry.y))
)

kmeans = KMeans(n_clusters=3, random_state=42)
checkpoints_gdf['cluster'] = kmeans.fit_predict(coords)

# ============================================================
# BUILD ROUTES
# ============================================================

def nearest_neighbor_tsp(nodes):
    if len(nodes) <= 1:
        return nodes

    unvisited = set(nodes)
    route = []
    current = nodes[0]

    route.append(current)
    unvisited.remove(current)

    while unvisited:
        nearest = min(
            unvisited,
            key=lambda node: nx.shortest_path_length(
                G, current, node, weight='weight'
            )
        )
        route.append(nearest)
        unvisited.remove(nearest)
        current = nearest

    return route

routes = {}

for cluster_id in checkpoints_gdf['cluster'].unique():
    cluster_nodes = checkpoints_gdf[
        checkpoints_gdf['cluster'] == cluster_id
    ]['nearest_node'].tolist()

    routes[cluster_id] = nearest_neighbor_tsp(cluster_nodes)

# ============================================================
# INTERACTIVE MAP
# ============================================================

print("Creating interactive map...")

roads_wgs = roads.to_crs(epsg=4326)
checkpoints_wgs = checkpoints_gdf.to_crs(epsg=4326)

center = [
    checkpoints_wgs.geometry.y.mean(),
    checkpoints_wgs.geometry.x.mean()
]

m = folium.Map(location=center, zoom_start=15)

colors = ['red', 'blue', 'green']

for cluster_id, route_nodes in routes.items():

    full_path = []

    for i in range(len(route_nodes) - 1):

        try:
            path = nx.shortest_path(
                G,
                route_nodes[i],
                route_nodes[i+1],
                weight='weight'
            )
            full_path.extend(path)
        except:
            continue

    if len(full_path) < 2:
        continue

    route_line = gpd.GeoDataFrame(
        geometry=[LineString(full_path)],
        crs=roads.crs
    ).to_crs(epsg=4326)

    coords_latlon = [
        (pt[1], pt[0]) for pt in route_line.geometry.iloc[0].coords
    ]

    folium.PolyLine(
        coords_latlon,
        color=colors[cluster_id],
        weight=5,
        tooltip=f"Route {cluster_id+1}"
    ).add_to(m)

# Add checkpoints
for _, row in checkpoints_wgs.iterrows():
    folium.CircleMarker(
        location=[row.geometry.y, row.geometry.x],
        radius=4,
        color="black",
        fill=True
    ).add_to(m)

m.save("iitb_routes_map.html")

print("Map saved as iitb_routes_map.html")
print("Open this file in your browser.")
