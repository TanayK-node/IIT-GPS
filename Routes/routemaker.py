import pandas as pd
import geopandas as gpd
import networkx as nx
from shapely.geometry import Point, LineString
import folium

# =========================
# CONFIG
# =========================
# Updated to point to the file in the current directory
ROADS_FILE = "../map/IITB_Auto_POV_Shapefile.gpkg"
POINTS_FILE = "./paths/market-h12.csv" # Ensure this path is correct!

# 0-based indices of anchor points
EXTRA_CONNECTIONS = [
    (1,5),(5,3)
]

METRIC_CRS = 32643   # UTM (meters)
WGS_CRS = 4326       # lat/lon

# =========================
# 1. LOAD ROAD NETWORK
# =========================
print(f"⏳ Loading road network from {ROADS_FILE}...")
roads = gpd.read_file(ROADS_FILE)

# Ensure CRS is set
if roads.crs is None:
    roads.set_crs(epsg=WGS_CRS, inplace=True)
roads = roads.to_crs(METRIC_CRS)

# =========================
# 2. BUILD ROAD GRAPH
# =========================
G = nx.Graph()

def add_linestring(line):
    coords = list(line.coords)
    for i in range(len(coords) - 1):
        p1, p2 = coords[i], coords[i + 1]
        G.add_edge(p1, p2, weight=Point(p1).distance(Point(p2)))

for geom in roads.geometry:
    if geom.geom_type == "LineString":
        add_linestring(geom)
    elif geom.geom_type == "MultiLineString":
        for g in geom.geoms:
            add_linestring(g)

print(f"✅ Road graph built: {len(G.nodes)} nodes, {len(G.edges)} edges")

# =========================
# 3. LOAD ROUTE POINTS
# =========================
try:
    df = pd.read_csv(POINTS_FILE)
    df.columns = df.columns.str.strip().str.lower()
    
    points_gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.lon, df.lat),
        crs=WGS_CRS
    ).to_crs(METRIC_CRS)

    n_points = len(points_gdf)
    print(f"✅ Loaded {n_points} anchor points from {POINTS_FILE}")
except FileNotFoundError:
    print(f"❌ ERROR: Could not find points file at {POINTS_FILE}")
    print("   Please check the path or upload the file.")
    exit()
# ============================================================
# V3: BUILD EDGE DEMAND FROM GPS MOVEMENT
# ============================================================

print("Computing edge demand...")

# Snap GPS to nearest graph node
graph_nodes = np.array(list(G.nodes))
graph_tree = cKDTree(graph_nodes)

def snap_to_graph(x,y):
    _, idx = graph_tree.query([x,y])
    return tuple(graph_nodes[idx])

gps_gdf['graph_node'] = gps_gdf.apply(
    lambda r: snap_to_graph(r.geometry.x, r.geometry.y),
    axis=1
)

# Sort by driver and time
gps_gdf = gps_gdf.sort_values(['driver id','timestamp'])

# Initialize demand on edges
for u,v in G.edges():
    G[u][v]['demand'] = 0

# Build edge demand
for driver, df in gps_gdf.groupby('driver id'):

    nodes = df['graph_node'].tolist()

    for i in range(len(nodes)-1):

        try:
            path = nx.shortest_path(
                G, nodes[i], nodes[i+1], weight='weight'
            )

            for j in range(len(path)-1):
                u = path[j]
                v = path[j+1]
                if G.has_edge(u,v):
                    G[u][v]['demand'] += 1

        except:
            continue

print("Edge demand computed.")

# =========================
# 4. SNAP POINTS TO NEAREST GRAPH NODE
# =========================
graph_nodes = list(G.nodes)

def nearest_node(pt):
    return min(graph_nodes, key=lambda n: Point(n).distance(pt))

print("⏳ Snapping points to road network...")
snapped_nodes = [nearest_node(pt) for pt in points_gdf.geometry]

# =========================
# 5. ROUTE MAIN PATH
# =========================
main_route_lines = []
print("⏳ Calculating main route...")

for i in range(n_points - 1):
    src, tgt = snapped_nodes[i], snapped_nodes[i + 1]
    if src == tgt:
        continue
    try:
        path = nx.shortest_path(G, src, tgt, weight="weight")
        main_route_lines.append(LineString(path))
    except nx.NetworkXNoPath:
        print(f"⚠️ No main path between point {i} and {i+1}")

# =========================
# 6. ROUTE EXTRA CONNECTIONS
# =========================
extra_route_lines = []
extra_metadata = [] # Store metadata for successful paths

print("⏳ Calculating extra connections...")
for i, j in EXTRA_CONNECTIONS:
    if i >= n_points or j >= n_points:
        print(f"⚠️ Skipping invalid extra connection indices ({i}, {j})")
        continue

    src, tgt = snapped_nodes[i], snapped_nodes[j]
    if src == tgt:
        continue

    try:
        path = nx.shortest_path(G, src, tgt, weight="weight")
        line = LineString(path)
        extra_route_lines.append(line)
        
        # Capture metadata for the GeoPackage
        extra_metadata.append({"from_pt": i, "to_pt": j, "length_m": line.length})
        
        print(f"   ✅ Extra path added: {i} ↔ {j}, length = {line.length:.1f} m")
    except nx.NetworkXNoPath:
        print(f"   ⚠️ No extra path found between {i} and {j}")

# =========================
# 7. CONVERT EVERYTHING TO WGS84
# =========================
roads_wgs = roads.to_crs(WGS_CRS)
points_wgs = points_gdf.to_crs(WGS_CRS)

main_wgs = gpd.GeoSeries(main_route_lines, crs=METRIC_CRS).to_crs(WGS_CRS)
extra_wgs = gpd.GeoSeries(extra_route_lines, crs=METRIC_CRS).to_crs(WGS_CRS)

# =========================
# 8. VISUALIZATION
# =========================
if not points_wgs.empty:
    center = points_wgs.geometry.iloc[0]
else:
    center = roads_wgs.geometry.centroid.iloc[0]

m = folium.Map(
    location=[center.y, center.x],
    zoom_start=16,
    tiles="cartodbpositron"
)

# --- A. Road Network (Background) ---
# Made darker and thicker to verify snapping alignment
folium.GeoJson(
    roads_wgs,
    name="IITB Road Network",
    style_function=lambda x: {
        "color": "#444444",  # Dark grey (high contrast against light map)
        "weight": 2,         # Thicker for visibility
        "opacity": 0.5       # Semi-transparent to not block the route totally
    },
    tooltip="Road Segment"
).add_to(m)

# --- B. Main Route ---
for line in main_wgs:
    coords = [(y, x) for x, y in line.coords]
    folium.PolyLine(
        coords,
        color="#FFFF00", # market-h12
        weight=7,
        opacity=0.8,
        tooltip="Main Route Path"
    ).add_to(m)

# --- C. Extra Routes ---
for line in extra_wgs:
    coords = [(y, x) for x, y in line.coords]
    folium.PolyLine(
        coords,
        color="#00FFFF", # Cyan
        weight=6,
        dash_array="10,6",
        opacity=1.0,
        tooltip="Extra Connection"
    ).add_to(m)

# --- D. Anchor Points ---
for idx, p in enumerate(points_wgs.geometry):
    folium.CircleMarker(
        [p.y, p.x],
        radius=5,
        color="red",
        fill=True,
        fill_color="red",
        fill_opacity=1,
        tooltip=f"Pt {idx}"
    ).add_to(m)

# Add layer control to toggle roads on/off
folium.LayerControl().add_to(m)

OUTPUT_HTML = "market-h12.html"
m.save(OUTPUT_HTML)
print(f"🎉 Visualization saved: {OUTPUT_HTML}")
print("👉 Open this file to verify the route lines up with the road network.")

# =========================
# 9. SAVE TO GPKG
# =========================
GPKG_FILE = "market-h12.gpkg"

# Main route layer
if len(main_wgs) > 0:
    main_route_gdf = gpd.GeoDataFrame(
        {"route_type": ["main"] * len(main_wgs), "segment_id": range(len(main_wgs))},
        geometry=main_wgs,
        crs=WGS_CRS
    )
    main_route_gdf.to_file(GPKG_FILE, layer="main_route", driver="GPKG")

# Extra route layer
if len(extra_wgs) > 0:
    extra_route_gdf = gpd.GeoDataFrame(
        extra_metadata, # Contains from_pt, to_pt, length_m
        geometry=extra_wgs,
        crs=WGS_CRS
    )
    extra_route_gdf.to_file(GPKG_FILE, layer="extra_routes", driver="GPKG")

print(f"✅ GeoPackage saved: {GPKG_FILE}")