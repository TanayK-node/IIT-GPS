import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import math
import os
import json
import time
import joblib
from joblib import Parallel, delayed
from shapely.geometry import Point, LineString, MultiLineString, box
from shapely.ops import substring
from shapely.strtree import STRtree
import folium
from streamlit_folium import st_folium

# =========================
# 1. CONFIGURATION
# =========================
st.set_page_config(page_title="Multicore Route Adherence", layout="wide")

# Paths
DEFAULT_ROADS_PATH = "../map/IITB_Auto_POV_Shapefile.gpkg"
DEFAULT_CAMPUS_PATH = "../map/IITB_Boundary_Poly.gpkg"
WEIGHTS_FILENAME = "../road_weights.json"

# Viterbi Constants
EPSG_METRIC = 32643
CANDIDATE_RADIUS_M = 100.0 
TOP_K = 6 
MAX_EMISSION_DIST = 200.0 
SIGMA_DIST = 12.0  
SIGMA_HEAD = 25.0
W_DIST = 1.0
W_HEAD = 0.6
V_SOFT = 20.0
V_MAX = 40.0
TURN_PENALTY_90 = 1.5
IDLE_SPEED_MS = 0.8
STAY_SAME_EDGE_BONUS = -1.0
SMALL_DT_S = 0.5
GAP_THRESHOLD_S = 80.0 
LEARNING_RATE = 2.0 

# =========================
# 2. CORE LOGIC (PICKLABLE HELPERS)
# =========================

def fast_bearing(p0, p1):
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    if dx == 0 and dy == 0: return 0.0
    deg = math.degrees(math.atan2(dx, dy))
    return (deg + 360.0) % 360.0

def angle_diff(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)

def gaussian_nll(x, s): return 0.5 * (x / s)**2

def get_candidates_bbox(pt, tree, geoms):
    x, y = pt.x, pt.y
    r = CANDIDATE_RADIUS_M
    indices = tree.query(box(x - r, y - r, x + r, y + r))
    
    candidates = []
    for i in indices:
        geom = geoms[i]
        d = geom.distance(pt)
        if d <= MAX_EMISSION_DIST:
            candidates.append((d, geom, i))
    
    candidates.sort(key=lambda x: x[0])
    return candidates[:TOP_K]

def timestamp_from_oid(oid):
    try: return int(str(oid)[:8], 16)
    except: return 0

def run_viterbi_logic(points_metric, times, tree, geoms, edge_weights):
    """
    Pure Logic Function for Viterbi. Returns dense_path (List of Points).
    """
    n_pts = len(points_metric)
    if n_pts < 2: return None

    layers = []
    
    # --- 1. Emission ---
    for i, pt in enumerate(points_metric):
        course_bearing = None
        if i < n_pts - 1:
            p_next = points_metric[i+1]
            course_bearing = fast_bearing((pt.x, pt.y), (p_next.x, p_next.y))
        elif i > 0:
            p_prev = points_metric[i-1]
            course_bearing = fast_bearing((p_prev.x, p_prev.y), (pt.x, pt.y))
            
        cands = get_candidates_bbox(pt, tree, geoms)
        if not cands: return None 

        layer_nodes = []
        for d, geom, geom_index in cands:
            proj_s = geom.project(pt)
            p_curr = geom.interpolate(proj_s)
            
            p_next_step = geom.interpolate(min(proj_s + 1.0, geom.length))
            seg_bearing = fast_bearing((p_curr.x, p_curr.y), (p_next_step.x, p_next_step.y))
            
            head_diff = angle_diff(course_bearing, seg_bearing) if course_bearing is not None else 0.0
            emis_cost = W_DIST * gaussian_nll(d, SIGMA_DIST) + W_HEAD * gaussian_nll(head_diff, SIGMA_HEAD)
            usage_count = edge_weights.get(geom_index, 0)
            bonus = LEARNING_RATE * np.log1p(usage_count)
            
            layer_nodes.append({
                "cost": emis_cost - bonus, 
                "snap_point": p_curr, 
                "dist_along": proj_s, 
                "seg_bearing": seg_bearing, 
                "geom_index": geom_index,
                "geom_obj": geom
            })
        layers.append(layer_nodes)

    # --- 2. Transition ---
    T = len(layers)
    costs = [[1e12] * len(l) for l in layers]
    parents = [[-1] * len(l) for l in layers]
    
    for k in range(len(layers[0])):
        costs[0][k] = layers[0][k]['cost']
        
    for t in range(1, T):
        prev_nodes = layers[t-1]
        curr_nodes = layers[t]
        dt = max(times[t] - times[t-1], SMALL_DT_S)
        if not curr_nodes or not prev_nodes: return None
        
        for k_curr, node_curr in enumerate(curr_nodes):
            best_val = 1e12
            best_prev = -1
            p_curr_snap = node_curr['snap_point']
            
            for k_prev, node_prev in enumerate(prev_nodes):
                prev_cost = costs[t-1][k_prev]
                if prev_cost >= 1e11: continue
                
                d_move = node_prev['snap_point'].distance(p_curr_snap)
                v = d_move / dt
                
                if v > V_MAX: trans_cost = 1e6
                else:
                    v_pen = ((v - V_SOFT) / (V_MAX - V_SOFT + 1e-6))**2 * 5.0 if v > V_SOFT else 0.0
                    turn_angle = angle_diff(node_prev['seg_bearing'], node_curr['seg_bearing'])
                    stay_bonus = STAY_SAME_EDGE_BONUS if (d_move/dt <= IDLE_SPEED_MS and node_prev['geom_index'] == node_curr['geom_index']) else 0.0
                    trans_cost = v_pen + ((turn_angle/90.0) * TURN_PENALTY_90) + stay_bonus

                total = prev_cost + trans_cost + node_curr['cost']
                if total < best_val:
                    best_val = total
                    best_prev = k_prev
            
            costs[t][k_curr] = best_val
            parents[t][k_curr] = best_prev

    # --- 3. Backtrack ---
    if not costs[-1]: return None
    last_idx = np.argmin(costs[-1])
    if costs[-1][last_idx] >= 1e11: return None

    path_nodes = []
    curr_idx = last_idx
    for t in range(T-1, -1, -1):
        path_nodes.append(layers[t][curr_idx])
        curr_idx = parents[t][curr_idx]
        if curr_idx == -1 and t > 0: return None
    path_nodes.reverse()

    # --- 4. Densify (Geometry Slicing) ---
    dense_path = []
    for i in range(len(path_nodes) - 1):
        curr = path_nodes[i]
        next_n = path_nodes[i+1]
        
        if curr['geom_index'] == next_n['geom_index']:
            try:
                line_slice = substring(curr['geom_obj'], curr['dist_along'], next_n['dist_along'])
                coords = list(line_slice.coords)
                segment_points = [Point(c) for c in coords]
                dense_path.extend(segment_points[:-1])
            except:
                dense_path.append(curr['snap_point'])
        else:
            dense_path.append(curr['snap_point'])
            
    dense_path.append(path_nodes[-1]['snap_point'])
    return dense_path

def calculate_single_trip_metrics(matched_points, prescribed_json):
    if not matched_points or len(matched_points) < 2: return None

    matched_line = LineString(matched_points)
    
    prescribed_pts = [Point(p['longitude'], p['latitude']) for p in prescribed_json]
    gs_prescribed = gpd.GeoSeries(prescribed_pts, crs="EPSG:4326").to_crs(epsg=EPSG_METRIC)
    prescribed_line = LineString(gs_prescribed.tolist())

    hausdorff_dist = matched_line.hausdorff_distance(prescribed_line)
    
    buffer_dist = 15.0
    prescribed_buffer = prescribed_line.buffer(buffer_dist)
    intersection = matched_line.intersection(prescribed_buffer)
    overlap_pct = (intersection.length / matched_line.length) * 100 if matched_line.length > 0 else 0

    return {
        "Overlap_Pct": round(overlap_pct, 2),
        "Hausdorff_Dist": round(hausdorff_dist, 2),
        "Matched_Length_m": round(matched_line.length, 2),
        "Prescribed_Length_m": round(prescribed_line.length, 2)
    }

# --- WORKER FUNCTION ---
def process_trip_task(row, tree, geoms, edge_weights):
    try:
        trip_id = row.get('_id', 'unknown')
        raw_path_json = json.loads(row['ridePath'])
        prescribed_path_json = json.loads(row['ridePathPrescribed'])
        
        if not raw_path_json: return None

        raw_lats = [p['latitude'] for p in raw_path_json]
        raw_lons = [p['longitude'] for p in raw_path_json]
        raw_times = [timestamp_from_oid(p.get('_id', '')) for p in raw_path_json]
        
        gs_wgs = gpd.GeoSeries([Point(x, y) for x, y in zip(raw_lons, raw_lats)], crs="EPSG:4326")
        gs_metric = gs_wgs.to_crs(epsg=EPSG_METRIC)
        points_metric = gs_metric.tolist()
        
        matched_points = run_viterbi_logic(points_metric, raw_times, tree, geoms, edge_weights)
        
        if not matched_points:
            return {"_id": trip_id, "status": "failed"}
            
        metrics = calculate_single_trip_metrics(matched_points, prescribed_path_json)
        
        matched_line_metric = LineString(matched_points)
        matched_gdf = gpd.GeoSeries([matched_line_metric], crs=EPSG_METRIC).to_crs(epsg=4326)
        matched_coords = [(p[1], p[0]) for p in matched_gdf.iloc[0].coords]
        prescribed_coords = [(p['latitude'], p['longitude']) for p in prescribed_path_json]
        raw_coords = list(zip(raw_lats, raw_lons))
        
        return {
            "_id": trip_id,
            "status": "success",
            "metrics": metrics,
            "raw_coords": raw_coords,
            "matched_coords": matched_coords,
            "prescribed_coords": prescribed_coords
        }
    except Exception as e:
        return {"_id": row.get('_id'), "status": "error", "msg": str(e)}

# =========================
# 3. MAP LOADING
# =========================
@st.cache_resource
def load_maps_resources(roads_path, campus_path):
    if not os.path.exists(roads_path): return None, None, None, None
    
    roads = gpd.read_file(roads_path)
    if roads.crs is None: roads.set_crs(epsg=4326, inplace=True)
    roads_metric = roads.to_crs(epsg=EPSG_METRIC)
    
    geoms = []
    for g in roads_metric.geometry:
        if g is None: continue
        if isinstance(g, MultiLineString): geoms.extend(g.geoms)
        elif isinstance(g, LineString): geoms.append(g)
    
    tree = STRtree(geoms)
    return roads, tree, geoms, roads_metric

def load_edge_weights():
    if os.path.exists(WEIGHTS_FILENAME):
        try:
            with open(WEIGHTS_FILENAME, "r") as f:
                data = json.load(f)
                return {int(k): v for k, v in data.items()}
        except:
            return {}
    return {}

# =========================
# 4. MAIN UI
# =========================

st.title("🚀 High-Performance Adherence Dashboard")

st.sidebar.header("Setup")
map_path = st.sidebar.text_input("Road Network (.gpkg)", DEFAULT_ROADS_PATH)
campus_path = st.sidebar.text_input("Campus (.gpkg)", DEFAULT_CAMPUS_PATH)
cpu_count = joblib.cpu_count()
n_jobs = st.sidebar.slider("CPU Cores", 1, cpu_count, max(1, cpu_count - 1))

roads, tree, geoms, roads_metric = load_maps_resources(map_path, campus_path)
if roads is None:
    st.error("Map not found.")
    st.stop()
else:
    st.sidebar.success(f"Map Loaded ({len(geoms)} segments)")

st.sidebar.divider()
st.sidebar.header("Layers")
show_roads = st.sidebar.toggle("Roads (Background)", value=True)
show_planned = st.sidebar.toggle("Planned Route (Blue)", value=True)
show_mapped = st.sidebar.toggle("Mapped Route (Red Line)", value=True)
show_raw = st.sidebar.toggle("Raw GPS (Grey)", value=True)
show_matched_pts = st.sidebar.toggle("Matched Points (Red Dots)", value=True)

uploaded_file = st.file_uploader("Upload Trip CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    if 'batch_results' not in st.session_state:
        st.session_state['batch_results'] = {}
        
    if st.sidebar.button(f"⚡ Batch Process ({len(df)} Trips)"):
        edge_weights = load_edge_weights()
        st.write("Processing...")
        bar = st.progress(0)
        
        results_list = Parallel(n_jobs=n_jobs)(
            delayed(process_trip_task)(row, tree, geoms, edge_weights) 
            for _, row in df.iterrows()
        )
        
        for res in results_list:
            if res and res['status'] == 'success':
                st.session_state['batch_results'][res['_id']] = res
        
        bar.progress(100)
        st.success("Done!")
    
    processed_ids = list(st.session_state['batch_results'].keys())
    
    if processed_ids:
        st.subheader("Results")
        selected_trip_id = st.selectbox("Select Trip", processed_ids)
        res = st.session_state['batch_results'][selected_trip_id]
        met = res['metrics']
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Adherence", f"{met['Overlap_Pct']}%")
        col2.metric("Deviation", f"{met['Hausdorff_Dist']} m")
        col3.metric("Actual Dist", f"{met['Matched_Length_m']} m")
        col4.metric("Planned Dist", f"{met['Prescribed_Length_m']} m")
        
        # --- OPTIMIZED MAP RENDERING ---
        center = res['raw_coords'][0] if res['raw_coords'] else [19.13, 72.91]
        
        # 'prefer_canvas=True' forces Leaflet to use Canvas rendering (Smoother for many points)
        m = folium.Map(location=center, zoom_start=15, tiles="cartodbpositron", prefer_canvas=True)
        
        # 1. Road Network (Simplified)
        if show_roads:
            folium.GeoJson(
                roads, 
                style_function=lambda x: {'color':'#000', 'weight':0.5, 'opacity':0.8},
                smooth_factor=2.0 # Optimizes rendering speed during zoom
            ).add_to(m)
            
        # 2. Raw Points (CircleMarkers can be heavy, use carefully)
        if show_raw:
            for pt in res['raw_coords']:
                folium.CircleMarker(pt, radius=1.5, color="gray", fill=True, fill_opacity=0.6, stroke=False).add_to(m)
        
        # 3. Planned Route
        if show_planned:
            folium.PolyLine(res['prescribed_coords'], color="blue", weight=3, dash_array="5, 10", opacity=0.7).add_to(m)
            
        # 4. Mapped Route
        if show_mapped:
            folium.PolyLine(res['matched_coords'], color="red", weight=3, opacity=0.8).add_to(m)
            
        # 5. Matched Points (The Red Dots - OPTIMIZED)
        if show_matched_pts:
            # Only draw points if there aren't too many to avoid lag
            if len(res['matched_coords']) < 1000:
                for pt in res['matched_coords']:
                    folium.CircleMarker(pt, radius=2, color="yellow", fill=True, fill_opacity=1.0, stroke=False).add_to(m)

        # 6. Start/End Markers (Restored!)
        folium.Marker(
            res['raw_coords'][0], popup="Start", 
            icon=folium.Icon(color="green", icon="play", prefix='fa')
        ).add_to(m)
        
        folium.Marker(
            res['raw_coords'][-1], popup="End", 
            icon=folium.Icon(color="black", icon="stop", prefix='fa')
        ).add_to(m)
            
        st_folium(m, width=1200, height=600)
        
    else:
        st.info("Select 'Batch Process' in sidebar to start.")