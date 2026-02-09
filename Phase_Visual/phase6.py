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
POINT_INDEX = 4  # <--- Specifically targeting Point #4
INITIAL_RADIUS = 100

def main():
    print(f"--- 🎯 Visualizing Map Match for Point #{POINT_INDEX} ---")

    # 1. Load Data
    try:
        gdf_roads = gpd.read_file(ROADS_FILENAME)
        if not gdf_roads.crs.is_projected:
            gdf_roads = gdf_roads.to_crs(epsg=32643)

        df = pd.read_csv(CSV_FILENAME)
        lat_col = next(c for c in df.columns if c.lower() in ['lat', 'latitude'])
        lon_col = next(c for c in df.columns if c.lower() in ['lon', 'longitude', 'lng'])
        
        # Get the specific point
        row = df.iloc[POINT_INDEX]
        p_t = gpd.points_from_xy([row[lon_col]], [row[lat_col]], crs="EPSG:4326")[0]
        p_t = gpd.GeoSeries([p_t], crs="EPSG:4326").to_crs(gdf_roads.crs).iloc[0]
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # 2. Find Candidates (Robust Search)
    # If point 4 is far away, we expand the search radius until we find something
    radius = INITIAL_RADIUS
    candidates = gpd.GeoDataFrame()
    
    print(f"Searching for roads near Point {POINT_INDEX}...")
    while radius <= 1000: # Limit to 1km to prevent infinite loop
        search_box = box(p_t.x - radius, p_t.y - radius, p_t.x + radius, p_t.y + radius)
        candidates = gdf_roads[gdf_roads.intersects(search_box)].copy()
        
        if not candidates.empty:
            print(f"✅ Found {len(candidates)} candidate(s) at {radius}m radius.")
            break
        
        print(f"⚠️ No roads at {radius}m... expanding search.")
        radius += 100 # Expand search
        
    if candidates.empty:
        print("❌ Point is completely isolated (no roads within 1km).")
        return

    # 3. Calculate Distances & Identify Best Match
    candidates['dist'] = candidates.geometry.distance(p_t)
    candidates = candidates.sort_values('dist')
    
    # Best Match is the first one
    best_road = candidates.iloc[0]
    best_proj = best_road.geometry.project(p_t)
    best_snap = best_road.geometry.interpolate(best_proj)

    # =========================
    # PLOTTING
    # =========================
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    ax.set_facecolor('white')

    # Zoom Logic (Dynamic based on how far the road is)
    dist_to_road = best_road['dist']
    zoom_margin = max(dist_to_road * 1.5, 60) # Ensure at least 60m zoom
    
    ax.set_xlim(p_t.x - zoom_margin, p_t.x + zoom_margin)
    ax.set_ylim(p_t.y - zoom_margin, p_t.y + zoom_margin)

    # 1. Draw All Nearby Roads (Grey)
    candidates.plot(ax=ax, color='lightgrey', linewidth=4, alpha=0.5, zorder=1)

    # 2. Draw Best Match Road (Green)
    gpd.GeoSeries([best_road.geometry]).plot(ax=ax, color='#2ecc71', linewidth=6, zorder=2, label="Best Match")

    # 3. Draw GPS Point (Blue)
    ax.scatter([p_t.x], [p_t.y], color='#2980b9', s=200, zorder=5, edgecolors='white', linewidth=2, label="GPS Point #4")
    ax.text(p_t.x + 2, p_t.y + 2, "Pt #4", fontsize=12, fontweight='bold', color='#2980b9')

    # 4. Draw Snapped Point (Green Dot)
    ax.scatter([best_snap.x], [best_snap.y], color='#27ae60', s=150, zorder=5, edgecolors='white', linewidth=2)

    # 5. Draw Projection Line (The "Error" Distance)
    ax.plot([p_t.x, best_snap.x], [p_t.y, best_snap.y], color='black', linestyle=':', linewidth=2, zorder=4)
    
    # Label the distance
    mid_x, mid_y = (p_t.x + best_snap.x)/2, (p_t.y + best_snap.y)/2
    ax.text(mid_x, mid_y, f"{dist_to_road:.1f}m", fontsize=10, fontweight='bold', 
            bbox=dict(facecolor='white', edgecolor='black', alpha=0.7, boxstyle='round,pad=0.2'))

    # 6. Title & Legend
    plt.title(f"Visualizing Match for Point #{POINT_INDEX}", fontsize=16)
    
    # Custom Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2980b9', markersize=10, label='GPS Raw'),
        plt.Line2D([0], [0], color='#2ecc71', lw=4, label='Matched Road'),
        plt.Line2D([0], [0], color='black', linestyle=':', lw=2, label=f'Error: {dist_to_road:.1f}m')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.axis('off')
    output_file = f"Point_{POINT_INDEX}_Detail.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"✅ Saved visualization: {output_file}")
    plt.show()

if __name__ == "__main__":
    main()