import pandas as pd
import geopandas as gpd
import numpy as np
import math
from shapely.geometry import Point, LineString, MultiLineString, box
from shapely.strtree import STRtree

# =========================
# CONFIG (REFERENCE MODE)
# =========================
ROADS_FILE = "../map/IITB_Auto_POV_Shapefile.gpkg"
GREEN_PATH_FILE = "../data/Path/green-Path.csv"

EPSG_WGS = 4326
EPSG_METRIC = 32643

# Tight constraints (reference snapping)
CANDIDATE_RADIUS_M = 30
MAX_EMISSION_DIST = 50
TOP_K = 3

SIGMA_DIST = 10.0
W_DIST = 1.0
W_HEAD = 0.0   # ⛔ Disable heading for anchors

# =========================
# HELPERS
# =========================
def gaussian_nll(x, s):
    return 0.5 * (x / s) ** 2

def fast_bearing(p0, p1):
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    if dx == 0 and dy == 0:
        return 0.0
    return (math.degrees(math.atan2(dx, dy)) + 360) % 360

# =========================
# LOAD ROAD NETWORK
# =========================
roads = gpd.read_file(ROADS_FILE)
roads = roads.to_crs(EPSG_METRIC)

geoms = []
for g in roads.geometry:
    if isinstance(g, MultiLineString):
        geoms.extend(g.geoms)
    elif isinstance(g, LineString):
        geoms.append(g)

tree = STRtree(geoms)

# =========================
# LOAD GREEN PATH POINTS
# =========================
df = pd.read_csv(GREEN_PATH_FILE)

gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df["Lon"], df["Lat"]),
    crs=EPSG_WGS
).to_crs(EPSG_METRIC)

points = list(gdf.geometry)

# =========================
# MAP MATCH (VITERBI – SIMPLIFIED)
# =========================
layers = []

for i, pt in enumerate(points):
    bbox = box(
        pt.x - CANDIDATE_RADIUS_M,
        pt.y - CANDIDATE_RADIUS_M,
        pt.x + CANDIDATE_RADIUS_M,
        pt.y + CANDIDATE_RADIUS_M
    )

    candidates = []
    for idx in tree.query(bbox):
        geom = geoms[idx]
        d = geom.distance(pt)
        if d <= MAX_EMISSION_DIST:
            proj = geom.interpolate(geom.project(pt))
            cost = gaussian_nll(d, SIGMA_DIST)
            candidates.append((cost, proj, idx))

    candidates.sort(key=lambda x: x[0])
    layers.append(candidates[:TOP_K])

# =========================
# VITERBI BACKTRACK
# =========================
T = len(layers)
costs = [[1e12]*len(l) for l in layers]
parents = [[-1]*len(l) for l in layers]

for k in range(len(layers[0])):
    costs[0][k] = layers[0][k][0]

for t in range(1, T):
    for j, (c_cost, c_pt, _) in enumerate(layers[t]):
        for i, (_, p_pt, _) in enumerate(layers[t-1]):
            d = p_pt.distance(c_pt)
            total = costs[t-1][i] + c_cost + d * 0.1
            if total < costs[t][j]:
                costs[t][j] = total
                parents[t][j] = i

idx = int(np.argmin(costs[-1]))
path = [idx]
for t in range(T-1, 0, -1):
    idx = parents[t][idx]
    path.append(idx)
path.reverse()

# =========================
# BUILD REFERENCE ROUTE
# =========================
matched_points = [
    layers[t][k][1]
    for t, k in enumerate(path)
]

route_line = LineString(matched_points)

# =========================
# EXPORT
# =========================
ref_route = gpd.GeoDataFrame(
    geometry=[route_line],
    crs=EPSG_METRIC
).to_crs(EPSG_WGS)

ref_route.to_file("green_reference_route.gpkg", driver="GPKG")

line = ref_route.geometry.iloc[0]

gps_points = [(y, x) for x, y in line.coords]

pd.DataFrame(
    gps_points,
    columns=["lat", "lon"]
).to_csv("green_reference_route_points.csv", index=False)


print("✅ green reference route generated")
