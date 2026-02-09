import pandas as pd
import geopandas as gpd
import numpy as np
import os
import math
from shapely.geometry import Point, LineString, MultiLineString, box
from shapely.strtree import STRtree
from joblib import Parallel, delayed

# =========================
# 1. CONFIGURATION
# =========================
TRAIN_FILE = "data/driver_locations-20250913-073001.csv"
TEST_FILE = "data/driver_locations-20250913-073001.csv"

# FILES
ROADS_FILENAME = "map/IITB_Auto_POV_Shapefile.gpkg"
CAMPUS_FILENAME = "map/IITB_Boundary_Poly.gpkg"

# PARAMETERS (Relaxed for debugging)
EPSG_METRIC = 32643
CANDIDATE_RADIUS_M = 300.0   # INCREASED from 100 to 300 to find distant roads
MAX_EMISSION_DIST = 400.0    # INCREASED
TOP_K = 6
SIGMA_DIST = 20.0            # Relaxed standard deviation
SIGMA_HEAD = 30.0
LEARNING_RATE = 2.0 

# =========================
# 2. CORE LOGIC
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

def load_maps_and_tree():
    if not os.path.exists(ROADS_FILENAME):
        print(f"❌ Map file not found: {ROADS_FILENAME}")
        return None, None
    
    print("Loading map network...")
    roads = gpd.read_file(ROADS_FILENAME)
    if roads.crs is None: roads.set_crs(epsg=4326, inplace=True)
    
    # DEBUG: Print Map Bounds
    bounds = roads.total_bounds # minx, miny, maxx, maxy
    print(f"🗺️  Map Coverage (Lat/Lon):")
    print(f"   Lat: {bounds[1]:.4f} to {bounds[3]:.4f}")
    print(f"   Lon: {bounds[0]:.4f} to {bounds[2]:.4f}")

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

def run_viterbi(segment_data, tree, geoms, edge_weights):
    points = segment_data['points']
    times = segment_data['times']
    original_indices = segment_data['indices']
    
    n_pts = len(points)
    if n_pts < 2: return []

    layers = []
    
    for i, pt in enumerate(points):
        # 1. Bearing Calculation
        course_bearing = None
        if i < n_pts - 1:
            p_next = points[i+1]
            course_bearing = fast_bearing((pt.x, pt.y), (p_next.x, p_next.y))
        elif i > 0:
            p_prev = points[i-1]
            course_bearing = fast_bearing((p_prev.x, p_prev.y), (pt.x, pt.y))
            
        # 2. Get Candidates
        cands = get_candidates_bbox(pt, tree, geoms)
        
        # FAIL SAFE: If no candidates, allow a "dummy" node so we don't drop the whole chain?
        # For now, strict viterbi returns empty if broken.
        if not cands: 
            # print(f"DEBUG: Point {i} has 0 candidates. Broken chain.") 
            return [] 

        layer_nodes = []
        for d, geom, geom_index in cands:
            proj_s = geom.project(pt)
            p_curr = geom.interpolate(proj_s)
            p_next = geom.interpolate(min(proj_s + 1.0, geom.length))
            seg_bearing = fast_bearing((p_curr.x, p_curr.y), (p_next.x, p_next.y))
            
            head_diff = angle_diff(course_bearing, seg_bearing) if course_bearing is not None else 0.0
            
            emis_cost = gaussian_nll(d, SIGMA_DIST) + 0.5 * gaussian_nll(head_diff, SIGMA_HEAD)
            
            # Brain Bonus
            usage_count = edge_weights.get(geom_index, 0)
            popularity_bonus = LEARNING_RATE * np.log1p(usage_count)
            
            final_cost = emis_cost - popularity_bonus
            
            layer_nodes.append({
                "geom_index": geom_index,
                "cost": final_cost, 
                "snap_point": p_curr,
                "seg_bearing": seg_bearing,
                "dist": d
            })
        layers.append(layer_nodes)

    # 3. Viterbi Pass
    T = len(layers)
    costs = [[1e12] * len(l) for l in layers]
    parents = [[-1] * len(l) for l in layers]
    
    for k in range(len(layers[0])):
        costs[0][k] = layers[0][k]['cost']
        
    for t in range(1, T):
        prev_nodes = layers[t-1]
        curr_nodes = layers[t]
        dt = max(times[t] - times[t-1], 0.1)
        
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
                
                # Simplified Transition Cost
                if v > 45.0: trans = 1e6
                else: trans = 0
                
                # Stay Bonus
                if (d_move/dt) < 1.0 and node_prev['geom_index'] == node_curr['geom_index']:
                    trans -= 1.0

                total = prev_cost + trans + node_curr['cost']
                if total < best_val:
                    best_val = total
                    best_prev = k_prev
            
            costs[t][k_curr] = best_val
            parents[t][k_curr] = best_prev

    # 4. Backtrack
    if not costs[-1]: return []
    last_layer_idx = np.argmin(costs[-1])
    if costs[-1][last_layer_idx] >= 1e10: return []

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
            "geom_index": node['geom_index'],
            "dist": node['dist'],
            "cost": node['cost']
        })
    return results

def process_pipeline(csv_path, tree, geoms, edge_weights, learn_mode=False):
    try:
        df = pd.read_csv(csv_path)
    except:
        return pd.DataFrame(columns=["idx"]).set_index("idx")

    # --- FIX: Explicit Date Parsing ---
    # Handles "2025-09-13" + "07:30:11" safely
    df["timestamp"] = pd.to_datetime(df["Date"].astype(str) + " " + df["IST_time"].astype(str), format='mixed', dayfirst=True)
    
    df = df.sort_values(by="timestamp")
    df["dt"] = df["timestamp"].diff().dt.total_seconds().fillna(0)
    df["segment_id"] = (df["dt"] > 120.0).cumsum()

    # DEBUG: Check GPS Bounds
    print(f"📍 GPS Data Coverage (Lat/Lon):")
    print(f"   Lat: {df['latitude'].min():.4f} to {df['latitude'].max():.4f}")
    print(f"   Lon: {df['longitude'].min():.4f} to {df['longitude'].max():.4f}")

    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["longitude"], df["latitude"]), crs="EPSG:4326"
    ).to_crs(epsg=EPSG_METRIC)

    tasks = []
    for seg_id in df["segment_id"].unique():
        group = gdf[gdf["segment_id"] == seg_id]
        # Keep segments shorter to increase chance of success for testing
        if len(group) > 5:
            # Slice giant segments into chunks of 50 to avoid one bad point killing 1000 points
            pts = list(group.geometry.values)
            tms = list(group["timestamp"].astype('int64') // 10**9)
            idxs = list(group.index)
            
            chunk_size = 50
            for i in range(0, len(pts), chunk_size):
                tasks.append({
                    'points': pts[i:i+chunk_size],
                    'times': tms[i:i+chunk_size],
                    'indices': idxs[i:i+chunk_size]
                })

    print(f"   > Processing {len(tasks)} segment chunks...")

    results_lists = Parallel(n_jobs=-1)(
        delayed(run_viterbi)(task, tree, geoms, edge_weights) for task in tasks
    )
    
    flat_results = [item for sublist in results_lists for item in sublist]
    
    if not flat_results:
        print("⚠️  Still 0 matches. Your GPS data is definitely NOT on the map.")
        # Return empty DF to prevent crash
        return pd.DataFrame(columns=["idx", "geom_index", "dist", "cost"]).set_index("idx")
    
    if learn_mode:
        for res in flat_results:
            idx = int(res['geom_index'])
            edge_weights[idx] = edge_weights.get(idx, 0) + 1
        return edge_weights
    else:
        return pd.DataFrame(flat_results).set_index("idx")

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    tree, geoms = load_maps_and_tree()
    
    if tree:
        print("\n❄️  COLD START...")
        cold_weights = {} 
        df_cold = process_pipeline(TEST_FILE, tree, geoms, cold_weights, learn_mode=False)
        
        print("\n🧠  TRAINING...")
        warm_weights = {} 
        warm_weights = process_pipeline(TRAIN_FILE, tree, geoms, warm_weights, learn_mode=True)
        
        print("\n🔥  WARM START...")
        df_warm = process_pipeline(TEST_FILE, tree, geoms, warm_weights, learn_mode=False)
        
        # COMPARE
        print("\n" + "="*40)
        comparison = df_cold.join(df_warm, lsuffix='_cold', rsuffix='_warm', how='inner')
        
        if comparison.empty:
            print("❌ Result: 0 Matches. Please check the Map Lat/Lon printed above.")
        else:
            changed = comparison[comparison['geom_index_cold'] != comparison['geom_index_warm']]
            print(f"✅ Success! Matched {len(comparison)} points.")
            print(f"path changed: {(len(changed)/len(comparison))*100:.2f}%")
            # ... inside the "if not comparison.empty:" block ...
        
        changed = comparison[comparison['geom_index_cold'] != comparison['geom_index_warm']].copy()
        
        print(f"✅ Success! Matched {len(comparison)} points.")
        print(f"path changed: {(len(changed)/len(comparison))*100:.2f}%")
        
        # --- NEW: Save Disagreements for Visualization ---
        if not changed.empty:
            # We need to recover lat/lon from the original dataframe to plot them
            # Join back to original data to get Lat/Lon
            original_data = pd.read_csv(TEST_FILE)
            # Assuming 'idx' in results corresponds to index in original CSV (which is how the script works)
            
            # Extract just the rows that changed
            # We need to know WHICH road index corresponds to which geometry, but for simple viz,
            # let's just save the GPS points where the algorithms disagreed.
            
            changed_indices = changed.index
            disagreement_df = original_data.loc[changed_indices].copy()
            
            # Add the decisions
            disagreement_df['Cold_Edge_ID'] = changed['geom_index_cold']
            disagreement_df['Warm_Edge_ID'] = changed['geom_index_warm']
            
            out_file = "diff_analysis.csv"
            disagreement_df.to_csv(out_file, index=False)
            print(f"💾 Saved {len(disagreement_df)} disagreement points to {out_file}")
            print("   -> Upload this to your map to see where the logic differs!")