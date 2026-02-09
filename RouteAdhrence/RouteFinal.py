import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import math
import os
import json
from shapely.geometry import Point, LineString, MultiLineString, box
from shapely.strtree import STRtree
import folium
from streamlit_folium import st_folium

# =========================
# 1. CONFIGURATION & CONSTANTS
# =========================
st.set_page_config(page_title="Trip Adherence Dashboard", layout="wide")

# Default Paths
DEFAULT_ROADS_PATH = "../map/IITB_Auto_POV_Shapefile.gpkg"
DEFAULT_CAMPUS_PATH = "../map/IITB_Boundary_Poly.gpkg"
WEIGHTS_FILENAME = "../road_weights.json"

# Physics / Viterbi Constants
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
# 2. LOGIC: MAP LOADING & VITERBI
# =========================

@st.cache_resource
def load_maps(roads_path, campus_path):
    """Loads map data and builds the spatial index. Cached for performance."""
    if not os.path.exists(roads_path):
        return None, None, None, None, "Road file not found."

    try:
        # Load Roads (Keep original WGS84 for visualization)
        roads = gpd.read_file(roads_path)
        if roads.crs is None: roads.set_crs(epsg=4326, inplace=True)
        
        # Project to Metric for Math
        roads_metric = roads.to_crs(epsg=EPSG_METRIC)
        
        # Load Campus (Optional)
        campus = None
        if os.path.exists(campus_path):
            campus = gpd.read_file(campus_path)
            if campus.crs is None: campus.set_crs(epsg=4326, inplace=True)

        # Build Index
        geoms = []
        for g in roads_metric.geometry:
            if g is None: continue
            if isinstance(g, MultiLineString): 
                geoms.extend(g.geoms)
            elif isinstance(g, LineString): 
                geoms.append(g)
        
        tree = STRtree(geoms)
        return roads, roads_metric, tree, geoms, "Success"
    except Exception as e:
        return None, None, None, None, str(e)

def load_edge_weights():
    if os.path.exists(WEIGHTS_FILENAME):
        try:
            with open(WEIGHTS_FILENAME, "r") as f:
                data = json.load(f)
                return {int(k): v for k, v in data.items()}
        except:
            return {}
    return {}

# --- Viterbi Algorithm Helpers ---
def fast_bearing(p0, p1):
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    if dx == 0 and dy == 0: return 0.0
    deg = math.degrees(math.atan2(dx, dy))
    return (deg + 360.0) % 360.0

def angle_diff(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)

def gaussian_nll(x, s): return 0.5 * (x / s)**2

def get_geometry_slice(p1, p2, geom):
    """
    Returns the list of coordinates along the geometry 'geom' 
    between points p1 and p2.
    """
    # Project points to get distance along the line
    d1 = geom.project(p1)
    d2 = geom.project(p2)
    
    # 1. Identify start and end (handle driving backwards along the digitized line)
    if d1 == d2:
        return [p1]
    
    is_forward = d1 < d2
    start_dist, end_dist = (d1, d2) if is_forward else (d2, d1)
    
    # 2. Extract intermediate vertices from the LineString
    coords = list(geom.coords)
    intermediate_points = []
    
    current_dist = 0.0
    for i in range(len(coords) - 1):
        # Segment length
        pt_a = Point(coords[i])
        pt_b = Point(coords[i+1])
        seg_len = pt_a.distance(pt_b)
        
        # Check if this segment overlaps with our slice [start_dist, end_dist]
        # We need the vertex "coords[i+1]" if it lies strictly inside the range
        vertex_dist = current_dist + seg_len
        
        if start_dist < vertex_dist < end_dist:
            intermediate_points.append(Point(coords[i+1]))
            
        current_dist += seg_len
        if current_dist > end_dist:
            break
            
    # 3. Construct Final Sequence
    # Start -> Intermediates -> End
    
    slice_points = [p1]
    if is_forward:
        slice_points.extend(intermediate_points)
    else:
        # If moving backward, reverse the intermediate points
        slice_points.extend(intermediate_points[::-1])
    slice_points.append(p2)
    
    return slice_points
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

def run_viterbi(points_metric, times, tree, geoms, edge_weights):
    n_pts = len(points_metric)
    if n_pts < 2: return []

    layers = []
    
    # --- STEP 1: Build Layers (No changes here) ---
    for i, pt in enumerate(points_metric):
        course_bearing = None
        if i < n_pts - 1:
            p_next = points_metric[i+1]
            course_bearing = fast_bearing((pt.x, pt.y), (p_next.x, p_next.y))
        elif i > 0:
            p_prev = points_metric[i-1]
            course_bearing = fast_bearing((p_prev.x, p_prev.y), (pt.x, pt.y))
            
        cands = get_candidates_bbox(pt, tree, geoms)
        if not cands: return []

        layer_nodes = []
        for d, geom, geom_index in cands:
            proj_s = geom.project(pt)
            p_curr = geom.interpolate(proj_s)
            
            # Look ahead for bearing
            p_next_step = geom.interpolate(min(proj_s + 1.0, geom.length))
            seg_bearing = fast_bearing((p_curr.x, p_curr.y), (p_next_step.x, p_next_step.y))
            
            head_diff = angle_diff(course_bearing, seg_bearing) if course_bearing is not None else 0.0
            emis_cost = W_DIST * gaussian_nll(d, SIGMA_DIST) + W_HEAD * gaussian_nll(head_diff, SIGMA_HEAD)
            usage_count = edge_weights.get(geom_index, 0)
            popularity_bonus = LEARNING_RATE * np.log1p(usage_count)
            final_cost = emis_cost - popularity_bonus
            
            # STORE GEOMETRY INFO for reconstruction later
            layer_nodes.append({
                "cost": final_cost, 
                "snap_point": p_curr, 
                "seg_bearing": seg_bearing, 
                "geom_index": geom_index,
                "geom_obj": geom # Keep reference to actual LineString
            })
        layers.append(layer_nodes)

    # --- STEP 2: Forward Pass (No changes here) ---
    T = len(layers)
    costs = [[1e12] * len(l) for l in layers]
    parents = [[-1] * len(l) for l in layers]
    
    for k in range(len(layers[0])):
        costs[0][k] = layers[0][k]['cost']
        
    for t in range(1, T):
        prev_nodes = layers[t-1]
        curr_nodes = layers[t]
        dt = max(times[t] - times[t-1], SMALL_DT_S)
        if not curr_nodes or not prev_nodes: return []
        
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
                    turn_pen = (turn_angle/90.0) * TURN_PENALTY_90
                    stay_bonus = STAY_SAME_EDGE_BONUS if (d_move/dt <= IDLE_SPEED_MS and node_prev['geom_index'] == node_curr['geom_index']) else 0.0
                    trans_cost = v_pen + turn_pen + stay_bonus

                total = prev_cost + trans_cost + node_curr['cost']
                if total < best_val:
                    best_val = total
                    best_prev = k_prev
            
            costs[t][k_curr] = best_val
            parents[t][k_curr] = best_prev

    # --- STEP 3: Backtrack & RECONSTRUCT GEOMETRY ---
    if not costs[-1]: return []
    last_idx = np.argmin(costs[-1])
    if costs[-1][last_idx] >= 1e11: return []

    path_nodes = []
    curr_idx = last_idx
    for t in range(T-1, -1, -1):
        node = layers[t][curr_idx]
        path_nodes.append(node)
        curr_idx = parents[t][curr_idx]
        if curr_idx == -1 and t > 0: return []
    path_nodes.reverse()

    # Create the detailed path
    dense_path = []
    
    for i in range(len(path_nodes) - 1):
        curr_node = path_nodes[i]
        next_node = path_nodes[i+1]
        
        p1 = curr_node['snap_point']
        p2 = next_node['snap_point']
        
        # KEY CHANGE: If on the same road, follow the curve!
        if curr_node['geom_index'] == next_node['geom_index']:
            # Get intermediate road vertices
            segment_points = get_geometry_slice(p1, p2, curr_node['geom_obj'])
            # Avoid duplicating points (the end of this is the start of next)
            dense_path.extend(segment_points[:-1]) 
        else:
            # Different roads: Just connect them (Intersection handling is complex)
            dense_path.append(p1)
            
    # Add the very last point
    dense_path.append(path_nodes[-1]['snap_point'])

    return dense_path
# =========================
# 3. LOGIC: ADHERENCE METRICS
# =========================
def calculate_metrics(matched_points_metric, prescribed_json):
    """
    Compares Matched Path (Metric Shapely Points) with Prescribed Path (JSON Lat/Lon).
    """
    if not matched_points_metric or len(matched_points_metric) < 2:
        return None

    # 1. Create LineStrings in Metric
    matched_line = LineString(matched_points_metric)
    
    prescribed_coords = [(p['longitude'], p['latitude']) for p in prescribed_json]
    if len(prescribed_coords) < 2: return None
    
    # Project prescribed to metric
    prescribed_line_wgs = LineString(prescribed_coords)
    prescribed_gdf = gpd.GeoSeries([prescribed_line_wgs], crs="EPSG:4326").to_crs(epsg=EPSG_METRIC)
    prescribed_line = prescribed_gdf.iloc[0]

    # 2. Calculate Metrics
    # Hausdorff (Max deviation)
    hausdorff_dist = matched_line.hausdorff_distance(prescribed_line)

    # Buffer Overlap (Adherence %)
    buffer_dist = 15.0 # meters
    prescribed_buffer = prescribed_line.buffer(buffer_dist)
    intersection = matched_line.intersection(prescribed_buffer)
    overlap_pct = (intersection.length / matched_line.length) * 100 if matched_line.length > 0 else 0

    return {
        "Overlap_Pct": round(overlap_pct, 2),
        "Hausdorff_Dist": round(hausdorff_dist, 2),
        "Matched_Length_m": round(matched_line.length, 2),
        "Prescribed_Length_m": round(prescribed_line.length, 2)
    }

def timestamp_from_oid(oid):
    try: return int(str(oid)[:8], 16)
    except: return 0

# =========================
# 4. STREAMLIT APP UI
# =========================

st.title("🗺️ Route Adherence Dashboard")

# --- SIDEBAR: Settings & Layers ---
st.sidebar.header("Configuration")
map_path = st.sidebar.text_input("Road Network File (.gpkg)", DEFAULT_ROADS_PATH)
campus_path = st.sidebar.text_input("Campus Boundary File (.gpkg)", DEFAULT_CAMPUS_PATH)

# Load Maps
roads, roads_metric, tree, geoms, msg = load_maps(map_path, campus_path)
if roads is None:
    st.error(f"Map Load Failed: {msg}")
    st.stop()
else:
    st.sidebar.success("Map Loaded Successfully!")

st.sidebar.divider()
st.sidebar.header("Layer Controls")
show_roads = st.sidebar.toggle("Road Network", value=True)
show_planned = st.sidebar.toggle("Planned Route (Blue)", value=True)
show_mapped_line = st.sidebar.toggle("Mapped Route (Red Line)", value=True)
show_raw_pts = st.sidebar.toggle("Raw Points (Grey)", value=True)
show_matched_pts = st.sidebar.toggle("Matched Points (Red Dots)", value=False)


# --- MAIN: Data Upload ---
uploaded_file = st.file_uploader("Upload Trip Data (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write(f"Loaded {len(df)} trips.")
    
    # --- SELECT TRIP ---
    trip_options = df['_id'].astype(str).tolist()
    selected_trip_id = st.selectbox("Select a Trip ID to Analyze", trip_options)
    
    # --- INITIALIZE SESSION STATE ---
    if 'analyzed_trip_id' not in st.session_state:
        st.session_state['analyzed_trip_id'] = None
    if 'analysis_results' not in st.session_state:
        st.session_state['analysis_results'] = None
        
    # --- BUTTON TO TRIGGER ANALYSIS ---
    if st.button("Analyze Trip"):
        with st.spinner("Processing Map Matching & Adherence..."):
            # Get Trip Data
            row = df[df['_id'] == selected_trip_id].iloc[0]
            
            # Parse JSONs
            try:
                raw_path_json = json.loads(row['ridePath'])
                prescribed_path_json = json.loads(row['ridePathPrescribed'])
                
                # Prepare Data for Viterbi
                points_metric = []
                times = []
                raw_points_wgs = [] # For visualization
                
                for p in raw_path_json:
                    lat, lon = p['latitude'], p['longitude']
                    raw_points_wgs.append((lat, lon))
                    
                    # Project to Metric
                    pt_wgs = Point(lon, lat)
                    s_wgs = gpd.GeoSeries([pt_wgs], crs="EPSG:4326")
                    s_metric = s_wgs.to_crs(epsg=EPSG_METRIC)
                    points_metric.append(s_metric.iloc[0])
                    times.append(timestamp_from_oid(p.get('_id', '')))

                # Run Map Matching
                edge_weights = load_edge_weights()
                matched_pts_metric = run_viterbi(points_metric, times, tree, geoms, edge_weights)
                
                if not matched_pts_metric:
                    st.error("Map Matching Failed (Could not find valid path on road network).")
                    st.session_state['analysis_results'] = None
                else:
                    # Calculate Metrics
                    metrics = calculate_metrics(matched_pts_metric, prescribed_path_json)
                    
                    # Convert Matched Points back to WGS84 for Folium
                    matched_line_metric = LineString(matched_pts_metric)
                    matched_gdf = gpd.GeoSeries([matched_line_metric], crs=EPSG_METRIC).to_crs(epsg=4326)
                    matched_line_wgs = matched_gdf.iloc[0]
                    matched_coords_wgs = [(p[1], p[0]) for p in matched_line_wgs.coords]
                    prescribed_coords_wgs = [(p['latitude'], p['longitude']) for p in prescribed_path_json]

                    # Save to Session State
                    st.session_state['analysis_results'] = {
                        'metrics': metrics,
                        'raw_points': raw_points_wgs,
                        'matched_coords': matched_coords_wgs,
                        'prescribed_coords': prescribed_coords_wgs
                    }
                    st.session_state['analyzed_trip_id'] = selected_trip_id
                    
            except Exception as e:
                st.error(f"Error parsing or processing data: {e}")

    # --- DISPLAY RESULTS (Persist across reloads) ---
    if st.session_state['analysis_results'] and st.session_state['analyzed_trip_id'] == selected_trip_id:
        
        results = st.session_state['analysis_results']
        metrics = results['metrics']
        
        st.divider()
        st.subheader(f"Results for Trip: {selected_trip_id}")

        # 1. Metrics Columns
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Adherence (Overlap %)", f"{metrics['Overlap_Pct']}%")
        col2.metric("Deviation (Hausdorff)", f"{metrics['Hausdorff_Dist']} m")
        col3.metric("Matched Length", f"{metrics['Matched_Length_m']} m")
        col4.metric("Prescribed Length", f"{metrics['Prescribed_Length_m']} m")

        # 2. Map
        center = results['raw_points'][0] if results['raw_points'] else [19.13, 72.91]
        m = folium.Map(location=center, zoom_start=15, tiles="cartodbpositron")
        
        # --- LAYERS ---
        
        # 1. Road Network (Background)
        if show_roads and roads is not None:
             folium.GeoJson(
                roads,
                name="Road Network",
                style_function=lambda x: {'color': '#999999', 'weight': 1, 'opacity': 1},
                tooltip="Road Segment"
            ).add_to(m)

        # 2. Raw Points (Grey dots)
        if show_raw_pts:
            for pt in results['raw_points']:
                folium.CircleMarker(
                    pt, radius=2, color="gray", fill=True, fill_opacity=0.6, tooltip="Raw GPS"
                ).add_to(m)

        # 3. Planned Route (Blue Dashed)
        if show_planned:
            folium.PolyLine(
                results['prescribed_coords'], 
                color="blue", weight=4, dash_array="5, 10", opacity=0.7, 
                tooltip="Planned Route (Prescribed)"
            ).add_to(m)
        
        # 4. Mapped Route (Red Solid Line)
        if show_mapped_line:
            folium.PolyLine(
                results['matched_coords'], 
                color="red", weight=4, opacity=0.8, 
                tooltip="Mapped Route (Actual)"
            ).add_to(m)

        # 5. Matched Points (Red Dots) - The actual vertices on the road
        if show_matched_pts:
            for pt in results['matched_coords']:
                folium.CircleMarker(
                    pt, radius=3, color="red", fill=True, color_opacity=1.0, 
                    tooltip="Matched Vertex"
                ).add_to(m)

        # Start/End Markers (Always ON)
        folium.Marker(results['raw_points'][0], popup="Start", icon=folium.Icon(color="green", icon="play")).add_to(m)
        folium.Marker(results['raw_points'][-1], popup="End", icon=folium.Icon(color="black", icon="stop")).add_to(m)

        st_folium(m, width=1200, height=600)
        
    elif st.session_state['analyzed_trip_id'] and st.session_state['analyzed_trip_id'] != selected_trip_id:
        st.warning("Trip selection changed. Click 'Analyze Trip' to process the new selection.")

else:
    st.info("Please upload a Trip Data CSV file to begin.")