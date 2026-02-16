import pandas as pd
import geopandas as gpd
import numpy as np
import math
import os
import glob
import json
from shapely.geometry import Point, LineString, MultiLineString, box
from shapely.strtree import STRtree
from joblib import Parallel, delayed

# =========================
# 1. CONFIGURATION
# =========================
ROADS_FILENAME = "../map/IITB_Auto_POV_Shapefile.gpkg"
CAMPUS_FILENAME = "../map/IITB_Boundary_Poly.gpkg"
WEIGHTS_FILENAME = "../road_weights.json" 
INPUT_FOLDER = "data"
OUTPUT_FOLDER = "matched_output_status"
ID_MAPPING_FILE = "0_ID_name.csv"

# Map Matching Constants
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

# Speed Threshold
STATIONARY_THRESHOLD_MPS = 5.0

# =========================
# 2. HELPER FUNCTIONS
# =========================

def load_edge_weights():
    if os.path.exists(WEIGHTS_FILENAME):
        try:
            with open(WEIGHTS_FILENAME, "r") as f:
                data = json.load(f)
                return {int(k): v for k, v in data.items()}
        except:
            return {}
    return {}

def save_edge_weights(weights):
    try:
        clean_weights = {int(k): int(v) for k, v in weights.items()}
        with open(WEIGHTS_FILENAME, "w") as f:
            json.dump(clean_weights, f)
    except Exception as e:
        print(f"⚠️ Saving Weights Failed: {e}")

def load_maps_and_tree():
    print("Loading Map Network...")
    if not os.path.exists(ROADS_FILENAME):
        print(f"Error: Road file not found at {ROADS_FILENAME}")
        return None, None
        
    roads = gpd.read_file(ROADS_FILENAME)
    if roads.crs is None: roads.set_crs(epsg=4326, inplace=True)
    roads_metric = roads.to_crs(epsg=EPSG_METRIC)
    
    geoms = []
    for g in roads_metric.geometry:
        if g is None: continue
        if isinstance(g, MultiLineString): 
            geoms.extend(g.geoms)
        elif isinstance(g, LineString): 
            geoms.append(g)
    
    tree = STRtree(geoms)
    return tree, geoms

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
        for d, geom, geom_index in cands:
            proj_s = geom.project(pt)
            p_curr = geom.interpolate(proj_s)
            p_next = geom.interpolate(min(proj_s + 1.0, geom.length))
            seg_bearing = fast_bearing((p_curr.x, p_curr.y), (p_next.x, p_next.y))
            
            head_diff = angle_diff(course_bearing, seg_bearing) if course_bearing is not None else 0.0
            
            emis_cost = W_DIST * gaussian_nll(d, SIGMA_DIST) + W_HEAD * gaussian_nll(head_diff, SIGMA_HEAD)
            usage_count = edge_weights.get(geom_index, 0)
            popularity_bonus = LEARNING_RATE * np.log1p(usage_count)
            final_cost = emis_cost - popularity_bonus
            
            layer_nodes.append({
                "geom": geom,
                "geom_index": geom_index,
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
            "geom_index": node['geom_index']
        })
        
    return results

def process_matching_parallel(_df_seg, _tree, _geoms, lat_col, lon_col):
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
    
    print(f"Running Viterbi on {len(tasks)} segments...")
    
    results_lists = Parallel(n_jobs=-1, prefer="threads")(
        delayed(run_viterbi_on_segment)(task, _tree, _geoms, edge_weights) for task in tasks
    )
    
    flat_results = [item for sublist in results_lists for item in sublist]
    
    if not flat_results: return pd.DataFrame()
    
    # Update Weights
    updated_count = 0
    for res in flat_results:
        g_idx = res.get('geom_index')
        if g_idx is not None and g_idx != -1:
            safe_idx = int(g_idx) 
            edge_weights[safe_idx] = edge_weights.get(safe_idx, 0) + 1
            updated_count += 1
            
    if updated_count > 0:
        save_edge_weights(edge_weights)
        print(f"Updated weights with {updated_count} points.")
    
    res_df = pd.DataFrame(flat_results).set_index("idx")
    final_df = _df_seg.join(res_df, how="inner")
    
    matched_gdf = gpd.GeoDataFrame(
        final_df,
        geometry=gpd.points_from_xy(final_df["Matched_X"], final_df["Matched_Y"]),
        crs=EPSG_METRIC
    ).to_crs(epsg=4326)
    
    final_df["Matched_Lat"] = matched_gdf.geometry.y
    final_df["Matched_Lon"] = matched_gdf.geometry.x
    
    return final_df

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

# =========================
# 3. PIPELINE LOGIC
# =========================

def run_pipeline():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        
    try:
        id_map = pd.read_csv(ID_MAPPING_FILE)
        id_map = id_map.rename(columns={"driver_id": "driverId", "name": "Driver_Name"})
    except Exception as e:
        print(f"Error loading ID Mapping file: {e}")
        return

    tree, geoms = load_maps_and_tree()
    if tree is None:
        print("Map files not found. Aborting.")
        return

    csv_files = glob.glob(os.path.join(INPUT_FOLDER, "*.csv"))
    print(f"Found {len(csv_files)} files to process.")

    for file_path in csv_files:
        try:
            print(f"\nProcessing {file_path}...")
            df = pd.read_csv(file_path)
            
            # --- STEP 0: FIX SWAPPED COORDINATES ---
            lat_col = 'latitude'
            lon_col = 'longitude'
            if lat_col in df.columns and lon_col in df.columns:
                mean_lat = df[lat_col].mean()
                mean_lon = df[lon_col].mean()
                if mean_lat > 50 and mean_lon < 50:
                    print(f"⚠️ Swapping coordinates for {file_path}...")
                    df.rename(columns={lat_col: 'temp_lon', lon_col: 'temp_lat'}, inplace=True)
                    df.rename(columns={'temp_lon': lon_col, 'temp_lat': lat_col}, inplace=True)

            # --- STEP 1: CLEANING ---
            if 'driverId' in df.columns:
                df = df.merge(id_map, on='driverId', how='left')
                df['Driver_Name'] = df['Driver_Name'].fillna("Unknown")
            else:
                print(f"Skipping {file_path}: 'driverId' column missing.")
                continue

            # Timestamp Processing
            df['timestamp_utc'] = pd.to_datetime(df['timestamp'], format='mixed')
            df['timestamp_ist'] = df['timestamp_utc'].dt.tz_convert('Asia/Kolkata') if df['timestamp_utc'].dt.tz is not None else (df['timestamp_utc'] + pd.Timedelta(hours=5, minutes=30))
            
            df['Date'] = df['timestamp_ist'].dt.date
            
            # --- FIX FOR 0-24 HOUR FORMAT ---
            # Explicitly format as string to avoid Excel Duration (24+) bugs
            df['IST_time'] = df['timestamp_ist'].dt.strftime('%H:%M:%S')
            
            df['timestamp'] = df['timestamp_ist']
            
            df = df.drop_duplicates(subset=['driverId', 'timestamp'], keep='first')
            df = df.sort_values(by=['driverId', 'timestamp'])

            # --- PREPARE FOR MATCHING ---
            df["dt_internal"] = df.groupby("driverId")["timestamp"].diff().dt.total_seconds().fillna(0)
            df["new_seg"] = (df["dt_internal"] > GAP_THRESHOLD_S) | (df["driverId"] != df["driverId"].shift(1))
            df["segment_id"] = df["new_seg"].cumsum()
            
            # --- STEP 2: MAP MATCHING ---
            if lat_col not in df.columns:
                print("Latitude column not found.")
                continue

            matched_df = process_matching_parallel(df, tree, geoms, lat_col, lon_col)
            
            if matched_df.empty:
                print(f"No matches found for {file_path}.")
                continue

            # --- STEP 3: CALCULATE STATUS, DT, DS, SPEED ---
            print("Calculating Step Metrics & Status...")
            matched_df = matched_df.sort_values(by=['driverId', 'timestamp'])
            
            matched_df['prev_time'] = matched_df.groupby('driverId')['timestamp'].shift(1)
            matched_df['dt'] = (matched_df['timestamp'] - matched_df['prev_time']).dt.total_seconds().fillna(0)
            
            matched_df['prev_matched_lat'] = matched_df.groupby('driverId')['Matched_Lat'].shift(1)
            matched_df['prev_matched_lon'] = matched_df.groupby('driverId')['Matched_Lon'].shift(1)
            
            matched_df['ds'] = haversine(
                matched_df['prev_matched_lat'], matched_df['prev_matched_lon'],
                matched_df['Matched_Lat'], matched_df['Matched_Lon']
            ).fillna(0)
            
            matched_df['speed'] = np.where(
                matched_df['dt'] > 0,
                matched_df['ds'] / matched_df['dt'],
                0
            )
            
            matched_df['status'] = np.where(matched_df['speed'] < STATIONARY_THRESHOLD_MPS, 'stationary', 'moving')

            # --- FINAL OUTPUT FORMATTING ---
            final_output = matched_df[[
                'Date', 'IST_time', 'driverId', 'Driver_Name', 
                'latitude', 'longitude', 
                'Matched_Lat', 'Matched_Lon', 
                'accuracy', 
                'dt', 'ds', 'speed', 
                'Distance_Diff_m',
                'status'
            ]].copy()
            
            final_output.columns = [
                'data', 'ist time', 'driver id', 'drivername', 
                'original lat', 'original lon', 
                'mapmatched lat', 'mapmatched lon', 
                'accuracy', 
                'dt', 'ds', 'speed', 
                'distnaece bw raw an og point',
                'status'
            ]
            
            base_name = os.path.basename(file_path)
            output_path = os.path.join(OUTPUT_FOLDER, f"matched_{base_name}")
            final_output.to_csv(output_path, index=False)
            print(f"Saved: {output_path}")
            
        except Exception as e:
            print(f"Failed to process {file_path}: {e}")

if __name__ == "__main__":
    run_pipeline()