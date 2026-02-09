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
from joblib import Parallel, delayed

# =========================
# 1. CONFIGURATION
# =========================
st.set_page_config(page_title="Fast Map Matcher (Self-Learning)", layout="wide")

# --- HARDCODED FILE PATHS ---
ROADS_FILENAME = "map/IITB_Auto_POV_Shapefile.gpkg"
CAMPUS_FILENAME = "map/IITB_Boundary_Poly.gpkg"
WEIGHTS_FILENAME = "road_weights.json" 

# Constants
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

# Learning Hyperparameter
LEARNING_RATE = 2.0 

# =========================
# 2. CACHED RESOURCES & MEMORY
# =========================

def load_edge_weights():
    """Loads the usage count for each road segment index."""
    if os.path.exists(WEIGHTS_FILENAME):
        try:
            with open(WEIGHTS_FILENAME, "r") as f:
                data = json.load(f)
                # JSON keys are strings, convert back to int indices
                return {int(k): v for k, v in data.items()}
        except:
            return {}
    return {}

def save_edge_weights(weights):
    """Saves the usage counts back to disk."""
    try:
        # Convert all keys and values to standard Python types (int) to avoid JSON errors
        clean_weights = {int(k): int(v) for k, v in weights.items()}
        with open(WEIGHTS_FILENAME, "w") as f:
            json.dump(clean_weights, f)
    except Exception as e:
        # Show the error in the app so we know if it fails
        st.error(f"⚠️ Saving Failed: {e}")

@st.cache_resource(show_spinner="Loading Map Network...")
def load_maps_and_tree():
    if not os.path.exists(ROADS_FILENAME) or not os.path.exists(CAMPUS_FILENAME):
        return None, None, None, None
        
    roads = gpd.read_file(ROADS_FILENAME)
    if roads.crs is None: roads.set_crs(epsg=4326, inplace=True)
    roads_metric = roads.to_crs(epsg=EPSG_METRIC)
    
    campus = gpd.read_file(CAMPUS_FILENAME)
    if campus.crs is None: campus.set_crs(epsg=4326, inplace=True)
    
    geoms = []
    # FIX: We do not attach .idx_ref anymore. We just build the list.
    # The index in this list 'geoms' will be the unique ID.
    for g in roads_metric.geometry:
        if g is None: continue
        if isinstance(g, MultiLineString): 
            geoms.extend(g.geoms)
        elif isinstance(g, LineString): 
            geoms.append(g)
    
    tree = STRtree(geoms)
    return roads.to_crs(4326), campus.to_crs(4326), tree, geoms

# =========================
# 3. MATH HELPERS
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
    
    # Query returns indices of geometries in the 'geoms' list
    indices = tree.query(box(x - r, y - r, x + r, y + r))
    
    candidates = []
    for i in indices:
        geom = geoms[i]
        d = geom.distance(pt)
        if d <= MAX_EMISSION_DIST:
            # FIX: Return the index 'i' along with the geometry
            candidates.append((d, geom, i))
    
    candidates.sort(key=lambda x: x[0])
    return candidates[:TOP_K]

# =========================
# 4. VITERBI WORKER (Self-Learning)
# =========================

def run_viterbi_on_segment(segment_data, tree, geoms, edge_weights):
    points = segment_data['points']
    times = segment_data['times']
    original_indices = segment_data['indices']
    
    n_pts = len(points)
    if n_pts < 2: return []

    layers = []
    
    for i, pt in enumerate(points):
        course_bearing = None
        if i < n_pts - 1:
            p_next = points[i+1]
            course_bearing = fast_bearing((pt.x, pt.y), (p_next.x, p_next.y))
        elif i > 0:
            p_prev = points[i-1]
            course_bearing = fast_bearing((p_prev.x, p_prev.y), (pt.x, pt.y))
            
        cands = get_candidates_bbox(pt, tree, geoms)
        if not cands: return [] 

        layer_nodes = []
        # FIX: Unpack the tuple to get geom_index (i)
        for d, geom, geom_index in cands:
            proj_s = geom.project(pt)
            p_curr = geom.interpolate(proj_s)
            p_next = geom.interpolate(min(proj_s + 1.0, geom.length))
            seg_bearing = fast_bearing((p_curr.x, p_curr.y), (p_next.x, p_next.y))
            
            head_diff = angle_diff(course_bearing, seg_bearing) if course_bearing is not None else 0.0
            
            # 1. Base Emission Cost
            emis_cost = W_DIST * gaussian_nll(d, SIGMA_DIST) + W_HEAD * gaussian_nll(head_diff, SIGMA_HEAD)
            
            # 2. Self-Learning Popularity Bonus
            # Use geom_index directly to look up weights
            usage_count = edge_weights.get(geom_index, 0)
            
            # Logarithmic bonus
            popularity_bonus = LEARNING_RATE * np.log1p(usage_count)
            
            final_cost = emis_cost - popularity_bonus
            
            layer_nodes.append({
                "geom": geom,
                "geom_index": geom_index, # Store the index
                "cost": final_cost, 
                "snap_point": p_curr,
                "seg_bearing": seg_bearing,
                "dist": d
            })
        layers.append(layer_nodes)

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
                if prev_cost >= 1e12: continue
                
                d_move = node_prev['snap_point'].distance(p_curr_snap)
                v = d_move / dt
                
                if v > V_MAX: trans_cost = 1e6
                else:
                    v_pen = ((v - V_SOFT) / (V_MAX - V_SOFT + 1e-6))**2 * 5.0 if v > V_SOFT else 0.0
                    turn_angle = angle_diff(node_prev['seg_bearing'], node_curr['seg_bearing'])
                    turn_pen = (turn_angle/90.0) * TURN_PENALTY_90
                    stay_bonus = 0.0
                    
                    # FIX: Compare geom_index instead of the geometry object itself for speed/safety
                    if (d_move / dt) <= IDLE_SPEED_MS and (node_prev['geom_index'] == node_curr['geom_index']):
                        stay_bonus = STAY_SAME_EDGE_BONUS
                    trans_cost = v_pen + turn_pen + stay_bonus

                total = prev_cost + trans_cost + node_curr['cost']
                if total < best_val:
                    best_val = total
                    best_prev = k_prev
            
            costs[t][k_curr] = best_val
            parents[t][k_curr] = best_prev

    if not costs[-1]: return []
    last_layer_idx = np.argmin(costs[-1])
    if costs[-1][last_layer_idx] >= 1e11: return []

    path_indices = [last_layer_idx]
    for t in range(T-1, 0, -1):
        prev_idx = parents[t][path_indices[-1]]
        if prev_idx == -1: return []
        path_indices.append(prev_idx)
    path_indices.reverse()

    results = []
    for t, k in enumerate(path_indices):
        node = layers[t][k]
        results.append({
            "idx": original_indices[t],
            "Matched_X": node['snap_point'].x,
            "Matched_Y": node['snap_point'].y,
            "Distance_Diff_m": node['dist'],
            "geom_index": node['geom_index'] # Pass out the index
        })
        
    return results

# =========================
# 5. MAIN PROCESSING
# =========================

def process_matching_parallel(_df_seg, _tree, _geoms, lat_col, lon_col):
    # 1. Load Brain (Weights)
    edge_weights = load_edge_weights()
    
    gdf = gpd.GeoDataFrame(
        _df_seg,
        geometry=gpd.points_from_xy(_df_seg[lon_col], _df_seg[lat_col]),
        crs="EPSG:4326"
    ).to_crs(epsg=EPSG_METRIC)
    
    tasks = []
    unique_segments = gdf["segment_id"].unique()
    
    for seg_id in unique_segments:
        group = gdf[gdf["segment_id"] == seg_id]
        if len(group) < 2: continue
        tasks.append({
            'points': list(group.geometry.values),
            'times': list(group["timestamp"].astype('int64') // 10**9),
            'indices': list(group.index)
        })
    
    st.write(f"Processing {len(tasks)} segments in parallel...")
    
    # 2. Pass weights to workers
    results_lists = Parallel(n_jobs=-1, prefer="threads")(
        delayed(run_viterbi_on_segment)(task, _tree, _geoms, edge_weights) for task in tasks
    )
    
    flat_results = [item for sublist in results_lists for item in sublist]
    
    if not flat_results: return pd.DataFrame()
    
    # 3. Update Brain (Learning Step)
    # 3. Update Brain (Learning Step)
    updated_count = 0
    for res in flat_results:
        g_idx = res.get('geom_index')
        # Ensure g_idx is valid and convert to standard python int
        if g_idx is not None and g_idx != -1:
            safe_idx = int(g_idx) 
            edge_weights[safe_idx] = edge_weights.get(safe_idx, 0) + 1
            updated_count += 1
            
    if updated_count > 0:
        save_edge_weights(edge_weights)
        st.toast(f"Brain updated with {updated_count} new data points!", icon="🧠")
    else:
        st.warning("Analysis finished, but no road usage was recorded for learning.")
    
    res_df = pd.DataFrame(flat_results).set_index("idx")
    final_df = _df_seg.join(res_df, how="inner")
    
    matched_gdf = gpd.GeoDataFrame(
        final_df,
        geometry=gpd.points_from_xy(final_df["Matched_X"], final_df["Matched_Y"]),
        crs=EPSG_METRIC
    ).to_crs(epsg=4326)
    
    final_df["Matched_Lat"] = matched_gdf.geometry.y
    final_df["Matched_Lon"] = matched_gdf.geometry.x
    
    final_df["_Raw_Lat_Disp"] = final_df[lat_col]
    final_df["_Raw_Lon_Disp"] = final_df[lon_col]
    
    return final_df

def segment_data(df):
    time_col = next((c for c in df.columns if "time" in c.lower()), None)
    if not time_col: return pd.DataFrame()
    try:
        df["timestamp"] = pd.to_datetime(df[time_col])
    except:
        if "Date" in df.columns and "IST_time" in df.columns:
             df["timestamp"] = pd.to_datetime(df["Date"] + " " + df["IST_time"], dayfirst=True)
    df = df.sort_values(by=["Driver_Name", "timestamp"])
    df["dt"] = df.groupby("Driver_Name")["timestamp"].diff().dt.total_seconds().fillna(0)
    df["new_seg"] = (df["dt"] > GAP_THRESHOLD_S) | (df["Driver_Name"] != df["Driver_Name"].shift(1))
    df["segment_id"] = df["new_seg"].cumsum()
    return df

# =========================
# 6. UI
# =========================

st.title("📍 IITB Map Matcher (Self-Learning)")

if 'matched_df' not in st.session_state:
    st.session_state['matched_df'] = None

with st.sidebar:
    st.header("Upload")
    traj_file = st.file_uploader("CSV File", type=["csv"])
    
    st.header("Display Options")
    display_limit = st.slider("Max Points on Map", 500, 5000, 1000)
    show_raw = st.checkbox("Show Raw Points (Blue)", value=False)
    
    # Load map and weights status
    roads_wgs, campus_wgs, tree, geoms = load_maps_and_tree()
    
    if roads_wgs is not None: 
        st.success("Maps Loaded")
        
    # Show learning stats
    weights = load_edge_weights()
    learned_segments = len(weights)
    total_matches = sum(weights.values())
    st.metric("Learned Segments", learned_segments)
    st.metric("Total Experience (Pts)", total_matches)

    st.markdown("---")
    # --- DOWNLOAD BUTTON ---
    if st.session_state['matched_df'] is not None:
        csv_data = st.session_state['matched_df'].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="💾 Download Results (CSV)",
            data=csv_data,
            file_name="matched_output.csv",
            mime="text/csv"
        )

if traj_file and roads_wgs is not None:
    if st.button("🚀 Run Matching"):
        df_raw = pd.read_csv(traj_file)
        lat_col = next((c for c in df_raw.columns if c.lower() in ['lat', 'latitude']), None)
        lon_col = next((c for c in df_raw.columns if c.lower() in ['lon', 'longitude', 'lng']), None)
        
        if lat_col:
            df_seg = segment_data(df_raw)
            if not df_seg.empty:
                result = process_matching_parallel(df_seg, tree, geoms, lat_col, lon_col)
                if not result.empty:
                    st.session_state['matched_df'] = result
                    st.success(f"Matched {len(result)} points.")
                else:
                    st.warning("No matches found.")
            else:
                st.error("Time column parsing failed.")
        else:
            st.error("Lat/Lon columns not found.")

    if st.session_state['matched_df'] is not None:
        df = st.session_state['matched_df']
        st.subheader("Results")
        
        plot_df = df.sample(min(len(df), display_limit))
        
        center_lat = plot_df["Matched_Lat"].mean()
        center_lon = plot_df["Matched_Lon"].mean()
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles="cartodbpositron")
        
        folium.GeoJson(campus_wgs, style_function=lambda x: {"color": "#333", "fill": False}).add_to(m)
        folium.GeoJson(roads_wgs, style_function=lambda x: {"color": "#ccc", "weight": 1}).add_to(m)
        
        # Matched Points (Red)
        for _, row in plot_df.iterrows():
            folium.CircleMarker(
                [row["Matched_Lat"], row["Matched_Lon"]],
                radius=3, color="red", fill=True, fill_opacity=0.8,
                tooltip="Matched"
            ).add_to(m)
            
        # Raw Points (Lighter Blue)
        if show_raw:
            raw_sample = plot_df.sample(frac=0.5, random_state=42)
            for _, row in raw_sample.iterrows():
                if "_Raw_Lat_Disp" in row and not pd.isna(row["_Raw_Lat_Disp"]):
                    folium.CircleMarker(
                        [row["_Raw_Lat_Disp"], row["_Raw_Lon_Disp"]],
                        radius=1, color="blue", weight=0, fill=True, fill_opacity=0.3,
                        tooltip="Raw GPS"
                    ).add_to(m)
            
        st_folium(m, width=1000, height=600, returned_objects=[], key="result_map")
# --- EXPORT SECTION ---
st.markdown("---")
st.header("📥 Export Data")

if st.session_state['matched_df'] is not None:
    # Prepare the full dataframe for export
    full_export_df = st.session_state['matched_df'].copy()
    
    # Optional: Clean up internal calculation columns before export
    cols_to_exclude = ["new_seg", "dt", "Matched_X", "Matched_Y"]
    export_ready = full_export_df.drop(columns=[c for c in cols_to_exclude if c in full_export_df.columns])
    
    # Convert to CSV
    csv_bytes = export_ready.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="Download Full Matched Data (CSV)",
        data=csv_bytes,
        file_name=f"IITB_Matched_Points_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
        help="Click to download the complete dataset with matched Latitude and Longitude."
    )
    
    st.info(f"Ready to export {len(export_ready)} points.")
else:
    st.info("Run the matching process to enable export.")