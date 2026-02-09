import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, LineString, MultiLineString
from shapely.ops import substring
from scipy.stats import gaussian_kde
import folium

# =========================
# CONFIGURATION
# =========================
INPUT_MAP_FILE = "../map/IITB_Auto_POV_Shapefile.gpkg"
MATCHED_CSV_FILE = "../data/matched_output (3).csv"  # Output from Viterbi Matcher
OUTPUT_STATS_FILE = "segment_stats.csv"
OUTPUT_PROOF_FILE = "point_migration_proof.csv"
OUTPUT_VIZ_FILE = "final_segmentation_map.html"

EPSG_METRIC = 32643
LEARNING_RATE = 0.6  # Updated to recommended safer value

# =========================
# 1. LOAD MAP & ALIGN INDICES
# =========================
def load_map_geoms():
    print("Loading map geometry...")
    roads = gpd.read_file(INPUT_MAP_FILE)
    if roads.crs is None: roads.set_crs(epsg=4326, inplace=True)
    roads_metric = roads.to_crs(epsg=EPSG_METRIC)
    
    geoms = []
    original_ids = []
    
    # CRITICAL: We must flatten exactly the same way the matcher did
    for idx, row in roads_metric.iterrows():
        g = row.geometry
        if g is None: continue
        if isinstance(g, MultiLineString):
            for sub_g in g.geoms:
                geoms.append(sub_g)
                original_ids.append(idx)
        elif isinstance(g, LineString):
            geoms.append(g)
            original_ids.append(idx)
            
    print(f"Loaded {len(geoms)} road segments.")
    return geoms, original_ids

# =========================
# 2. CALCULATE PROJECTIONS (The "Mini-Match")
# =========================
def calculate_projections(df, geoms):
    print("Calculating projected distances (proj_s)...")
    
    geom_map = {} # { geom_index: [ {row_data}, ... ] }
    
    # Convert dataframe to metric CRS for accurate distance measurement
    gdf = gpd.GeoDataFrame(
        df, 
        geometry=gpd.points_from_xy(df["Matched_Lon"], df["Matched_Lat"]),
        crs="EPSG:4326"
    ).to_crs(EPSG_METRIC)
    
    for idx, row in gdf.iterrows():
        g_idx = int(row["geom_index"])
        
        if g_idx >= len(geoms): continue
        
        target_geom = geoms[g_idx]
        pt = row.geometry
        
        proj_s = target_geom.project(pt)
        
        if g_idx not in geom_map: geom_map[g_idx] = []
        
        geom_map[g_idx].append({
            "CSV_Index": idx,
            "proj_s": proj_s,
            "latitude": row["Matched_Lat"],
            "longitude": row["Matched_Lon"]
        })
        
    return geom_map

# =========================
# 3. HOTSPOT SEGMENTATION LOGIC
# =========================
def analyze_and_segment(geoms, original_ids, geom_map):
    print("Running Hotspot Analysis & Weightage Calculation...")
    
    segment_stats = []
    point_mappings = []
    
    temp_stats = [] # Store items temporarily to calculate global min/max

    for g_idx, points_data in geom_map.items():
        if not points_data: continue
        
        geom = geoms[g_idx]
        projections = np.array([p['proj_s'] for p in points_data])
        
        # --- A. Detect Split Points ---
        split_points = []
        
        if len(projections) > 20 and geom.length > 50:
            try:
                kde = gaussian_kde(projections, bw_method=0.15)
                x_grid = np.linspace(0, geom.length, 100)
                density = kde(x_grid)
                
                threshold = density.max() * 0.20
                is_active = density > threshold
                
                transitions = np.where(np.diff(is_active.astype(int)) != 0)[0]
                
                for t in transitions:
                    loc = x_grid[t]
                    if 15 < loc < (geom.length - 15):
                        split_points.append(loc)
            except:
                pass 

        # --- B. Define Segments ---
        boundaries = [0.0] + sorted(split_points) + [geom.length]
        
        for i in range(len(boundaries)-1):
            start = boundaries[i]
            end = boundaries[i+1]
            
            sub_points = [
                p for p in points_data 
                if start <= p['proj_s'] < end
            ]
            
            count = len(sub_points)
            
            is_hotspot = (count > 100) or (count > len(projections) * 0.2)
            status = "Hotspot" if is_hotspot else "Inactive"
            
            new_id = f"{original_ids[g_idx]}_{i}"
            
            # --- C. Calculate Weightage ---
            weightage = LEARNING_RATE * np.log(1 + count)

            temp_stats.append({
                "Geom_List_Index": g_idx,
                "Old_ID": original_ids[g_idx],
                "New_ID": new_id,
                "Start_m": start,
                "End_m": end,
                "Point_Count": count,
                "Status": status,
                "Weightage": weightage
            })
            
            for p in sub_points:
                point_mappings.append({
                    "CSV_Index": p["CSV_Index"],
                    "Old_ID": original_ids[g_idx],
                    "New_ID": new_id,
                    "proj_s": round(p["proj_s"], 2),
                    "latitude": p["latitude"],
                    "longitude": p["longitude"]
                })

    # --- D. Classify Weights (6 Categories) ---
    if temp_stats:
        weights = [x["Weightage"] for x in temp_stats]
        min_w = min(weights)
        max_w = max(weights)
        rng = max_w - min_w
        
        # === PRINTING STATS TO TERMINAL ===
        print("-" * 40)
        print(f"📊 Weightage Statistics:")
        print(f"   Min Weightage : {min_w:.4f}")
        print(f"   Max Weightage : {max_w:.4f}")
        print(f"   Range         : {rng:.4f}")
        print("-" * 40)
        # ==================================
        
        diff = rng
        if diff == 0: diff = 1 
        
        bin_size = diff / 6.0
        
        for item in temp_stats:
            w = item["Weightage"]
            cat = int((w - min_w) // bin_size) + 1
            if cat > 6: cat = 6 
            if w == 0: cat = 0 
            
            item["Weight_Category"] = cat
            segment_stats.append(item)
    
    return segment_stats, point_mappings

# =========================
# 4. VISUALIZATION
# =========================
def generate_map(geoms, stats_df):
    print("Generating Map...")
    
    active = stats_df[stats_df["Point_Count"] > 0]["Point_Count"]
    if not active.empty:
        low, high = active.quantile(0.33), active.quantile(0.66)
    else:
        low, high = 10, 50

    def get_color(cnt):
        if cnt == 0: return "#dddddd"
        if cnt < low: return "green"
        if cnt < high: return "orange"
        return "red"

    m = folium.Map(location=[19.133, 72.913], zoom_start=15, tiles="cartodbpositron")
    
    fg = folium.FeatureGroup(name="Segments")
    
    for _, row in stats_df.iterrows():
        g_idx = int(row["Geom_List_Index"])
        geom = geoms[g_idx]
        
        start, end = row["Start_m"], row["End_m"]
        if start >= geom.length: continue
        
        sub_geom = substring(geom, start, min(end, geom.length))
        
        seg_wgs = gpd.GeoSeries([sub_geom], crs=EPSG_METRIC).to_crs(4326).iloc[0]
        coords = [(p[1], p[0]) for p in seg_wgs.coords]
        
        color = get_color(row["Point_Count"])
        weight = 4 if row["Point_Count"] > 0 else 2
        opacity = 0.8 if row["Point_Count"] > 0 else 0.3
        
        popup_txt = (f"ID: {row['New_ID']}<br>"
                     f"Pts: {row['Point_Count']}<br>"
                     f"Wgt: {row['Weightage']:.4f}<br>"
                     f"Cat: {row['Weight_Category']}")
        
        folium.PolyLine(
            coords, 
            color=color, 
            weight=weight, 
            opacity=opacity,
            popup=popup_txt,
            tooltip=f"{row['New_ID']} (Cat: {row['Weight_Category']})"
        ).add_to(fg)

    fg.add_to(m)
    m.save(OUTPUT_VIZ_FILE)
    print(f"✅ Map saved to {OUTPUT_VIZ_FILE}")

# =========================
# MAIN PIPELINE
# =========================
def main():
    geoms, original_ids = load_map_geoms()
    df = pd.read_csv(MATCHED_CSV_FILE)
    
    geom_map = calculate_projections(df, geoms)
    
    stats, mappings = analyze_and_segment(geoms, original_ids, geom_map)
    
    df_stats = pd.DataFrame(stats)
    df_stats.to_csv(OUTPUT_STATS_FILE, index=False)
    
    df_map = pd.DataFrame(mappings)
    df_map.to_csv(OUTPUT_PROOF_FILE, index=False)
    
    print(f"✅ Saved stats to {OUTPUT_STATS_FILE}")
    print(f"✅ Saved proof to {OUTPUT_PROOF_FILE}")
    
    generate_map(geoms, df_stats)

if __name__ == "__main__":
    main()