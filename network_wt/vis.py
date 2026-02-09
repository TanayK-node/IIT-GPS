import pandas as pd
import geopandas as gpd
import folium
from shapely.ops import substring
from shapely.geometry import LineString, MultiLineString

# --- CONFIG ---
MAP_FILE = "../map/IITB_Auto_POV_Shapefile.gpkg"
STATS_FILE = "segment_stats.csv"
POINTS_FILE = "point_migration_proof.csv"
OUTPUT_HTML = "final_traffic_heatmap.html"

def load_map_geoms():
    print("Loading map geometry...")
    roads = gpd.read_file(MAP_FILE)
    if roads.crs is None: roads.set_crs(epsg=4326, inplace=True)
    roads_metric = roads.to_crs(epsg=32643) 
    
    geoms = []
    # Use exact same flattening logic as Phase 1
    for idx, row in roads_metric.iterrows():
        g = row.geometry
        if g is None: continue
        if isinstance(g, MultiLineString):
            for sub_g in g.geoms: geoms.append(sub_g)
        elif isinstance(g, LineString):
            geoms.append(g)
    return geoms

def get_color(count, low, high):
    if count == 0: return "#dddddd"
    if count < low: return "green"
    if count < high: return "orange"
    return "red"

def main():
    # 1. Load Data
    try:
        df_stats = pd.read_csv(STATS_FILE)
        df_points = pd.read_csv(POINTS_FILE)
        geoms = load_map_geoms()
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        return

    print(f"Loaded {len(df_stats)} segments from stats file.")
    print(f"Loaded {len(geoms)} geometries from map file.")

    # 2. Setup Thresholds
    active = df_stats[df_stats["Point_Count"] > 0]["Point_Count"]
    if len(active) > 0:
        low, high = active.quantile(0.33), active.quantile(0.66)
    else:
        low, high = 5, 100
    
    # 3. Create Map
    center_lat = df_points["latitude"].mean() if not df_points.empty else 19.133
    center_lon = df_points["longitude"].mean() if not df_points.empty else 72.913
    m = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles="cartodbpositron")

    # 4. Plot Segments
    count_plotted = 0
    fg_seg = folium.FeatureGroup(name="Segments")
    
    for i, row in df_stats.iterrows():
        # --- THE FIX: USE LIST INDEX ---
        g_idx = int(row["Geom_List_Index"]) 
        
        if g_idx >= len(geoms):
            print(f"Warning: Index {g_idx} out of bounds (Max {len(geoms)})")
            continue
            
        geom = geoms[g_idx]
        
        # Cut
        start, end = row["Start_m"], row["End_m"]
        if start >= geom.length: continue
        
        segment = substring(geom, start, min(end, geom.length))
        
        # Style
        cnt = row["Point_Count"]
        color = get_color(cnt, low, high)
        weight = 4 if cnt > 0 else 2
        opacity = 0.8 if cnt > 0 else 0.2
        
        # Convert to WGS84 for plotting
        seg_wgs = gpd.GeoSeries([segment], crs=32643).to_crs(4326).iloc[0]
        coords = [(p[1], p[0]) for p in seg_wgs.coords]
        
        folium.PolyLine(
            coords, color=color, weight=weight, opacity=opacity,
            popup=f"Pts: {cnt}", tooltip=f"ID: {row['New_ID']}"
        ).add_to(fg_seg)
        count_plotted += 1

    fg_seg.add_to(m)
    print(f"Successfully plotted {count_plotted} segments.")

    # 5. Plot Points
    fg_pts = folium.FeatureGroup(name="Raw Points", show=False)
    if not df_points.empty:
        # Downsample for speed
        sample = df_points.sample(frac=0.5) if len(df_points) > 5000 else df_points
        for _, row in sample.iterrows():
            folium.CircleMarker(
                [row["latitude"], row["longitude"]],
                radius=1, color="black", fill=True, fill_opacity=0.5
            ).add_to(fg_pts)
    fg_pts.add_to(m)

    folium.LayerControl().add_to(m)
    m.save(OUTPUT_HTML)
    print(f"✅ Map Saved to {OUTPUT_HTML}")

if __name__ == "__main__":
    main()