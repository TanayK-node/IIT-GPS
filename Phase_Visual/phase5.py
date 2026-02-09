import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import box

# =========================
# CONFIGURATION
# =========================
ROADS_FILENAME = "../map/IITB_Auto_POV_Shapefile.gpkg"
CSV_FILENAME = "../data/Discrete points.csv"
START_INDEX = 4     
WINDOW_SIZE = 5      
SEARCH_RADIUS_M = 100

def main():
    print(f"--- 🔄 Generating CLEAN Phase 5 (Learning Loop) ---")

    # 1. Load Data
    try:
        gdf_roads = gpd.read_file(ROADS_FILENAME)
        if not gdf_roads.crs.is_projected:
            gdf_roads = gdf_roads.to_crs(epsg=32643)

        df = pd.read_csv(CSV_FILENAME)
        lat_col = next(c for c in df.columns if c.lower() in ['lat', 'latitude'])
        lon_col = next(c for c in df.columns if c.lower() in ['lon', 'longitude', 'lng'])
        subset = df.iloc[START_INDEX : START_INDEX + WINDOW_SIZE].copy()
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # 2. Get Matched Path
    matched_roads = []
    for t, (idx, row) in enumerate(subset.iterrows()):
        p_t = gpd.points_from_xy([row[lon_col]], [row[lat_col]], crs="EPSG:4326")[0]
        p_t = gpd.GeoSeries([p_t], crs="EPSG:4326").to_crs(gdf_roads.crs).iloc[0]
        
        # Robust search
        radius = SEARCH_RADIUS_M
        candidates = gpd.GeoDataFrame()
        while radius <= 500:
            box_search = box(p_t.x - radius, p_t.y - radius, p_t.x + radius, p_t.y + radius)
            candidates = gdf_roads[gdf_roads.intersects(box_search)].copy()
            if not candidates.empty: break
            radius += 100
            
        if not candidates.empty:
            candidates['dist'] = candidates.geometry.distance(p_t)
            best = candidates.sort_values('dist').iloc[0]
            matched_roads.append(best)

    if not matched_roads:
        print("❌ No matches found to plot.")
        return

    demo_road = matched_roads[0]
    seg_id = demo_road.name 
    
    # =========================
    # PLOTTING
    # =========================
    fig, ax = plt.subplots(figsize=(16, 8)) 
    ax.set_facecolor('white')
    
    # 1. Calculate Bounds
    min_x = min(r.geometry.centroid.x for r in matched_roads)
    max_x = max(r.geometry.centroid.x for r in matched_roads)
    min_y = min(r.geometry.centroid.y for r in matched_roads)
    max_y = max(r.geometry.centroid.y for r in matched_roads)
    
    # Define distinct zones
    map_center_x = (min_x + max_x) / 2
    
    # Set plot limits
    ax.set_xlim(min_x - 100, max_x + 400) # Big offset to the right
    ax.set_ylim(min_y - 100, max_y + 100)
    
    # 2. Draw Map (Very Clean)
    view_box = box(min_x - 200, min_y - 200, max_x + 200, max_y + 200)
    visible = gdf_roads[gdf_roads.intersects(view_box)]
    visible.plot(ax=ax, color='#bdc3c7', linewidth=2, alpha=0.15, zorder=1)
    
    for r in matched_roads:
        gpd.GeoSeries([r.geometry]).plot(ax=ax, color='#2ecc71', linewidth=8, alpha=0.9, zorder=2)

    ax.text(map_center_x, max_y + 20, "Matched Trip", 
            ha='center', fontsize=14, fontweight='bold', color='#2ecc71')

    # 3. Draw "Memory Update" Box (Far Right)
    box_x = max_x + 150
    box_y = (min_y + max_y) / 2
    box_w = 220
    box_h = 120
    
    # --- FIX: Use FancyBboxPatch instead of Rectangle ---
    rect = patches.FancyBboxPatch((box_x, box_y - box_h/2), box_w, box_h, 
                                  boxstyle="round,pad=0.1", 
                                  facecolor='white', edgecolor='#34495e', linewidth=3, 
                                  zorder=5, mutation_scale=1.0)
    ax.add_patch(rect)
    
    # Header
    ax.text(box_x + box_w/2, box_y + 30, "System Memory", 
            ha='center', fontsize=14, fontweight='bold', color='#34495e', zorder=6)
    
    # The Update Logic
    ax.text(box_x + 20, box_y - 10, f"Segment ID: {seg_id}", fontsize=12, color='grey', zorder=6)
    
    # The Counter
    ax.text(box_x + 20, box_y - 40, "13,766", fontsize=14, color='#95a5a6', fontweight='bold', zorder=6)
    ax.text(box_x + 90, box_y - 40, "➔", fontsize=16, color='#f39c12', zorder=6)
    ax.text(box_x + 130, box_y - 40, "13,767", fontsize=16, color='#27ae60', fontweight='bold', zorder=6)
    
    ax.text(box_x + box_w/2, box_y - 70, "(Usage Count Updated)", 
            ha='center', fontsize=10, style='italic', color='grey', zorder=6)

    # 4. Connection Arrow
    arrow_start = (max_x, (min_y+max_y)/2)
    arrow_end = (box_x, box_y)
    
    ax.annotate("", xy=arrow_end, xytext=arrow_start,
                arrowprops=dict(arrowstyle="->", color='#f39c12', lw=4, connectionstyle="arc3,rad=-0.1"),
                zorder=4)
    
    ax.text((max_x + box_x)/2, box_y + 20, "Feedback", 
            ha='center', fontsize=12, fontweight='bold', color='#e67e22', 
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

    # 5. Title
    plt.title("Phase 5: The Loop (Learning from Experience)", fontsize=18, pad=20)
    plt.axis('off')
    
    output_file = "Phase5_Learning_Clean_Fixed.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"✅ Saved Clean Phase 5: {output_file}")
    plt.show()

if __name__ == "__main__":
    main()