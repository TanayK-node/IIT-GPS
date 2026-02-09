import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import box, LineString

# =========================
# CONFIGURATION
# =========================
ROADS_FILENAME = "../map/IITB_Auto_POV_Shapefile.gpkg"
CSV_FILENAME = "../data/Discrete points.csv"
START_INDEX = 10     
WINDOW_SIZE = 5      
SEARCH_RADIUS_M = 100

def main():
    print(f"--- 🗺️ Generating Map Overlay for Points {START_INDEX}-{START_INDEX+WINDOW_SIZE-1} ---")

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

    # 2. Calculate Matches (Greedy/Best Match for Visualization)
    matched_points = [] # Stores (GPS Point, Snapped Point, Road Geom)
    
    print("Calculating matches...")
    for t, (idx, row) in enumerate(subset.iterrows()):
        p_t = gpd.points_from_xy([row[lon_col]], [row[lat_col]], crs="EPSG:4326")[0]
        p_t = gpd.GeoSeries([p_t], crs="EPSG:4326").to_crs(gdf_roads.crs).iloc[0]
        
        # Robust Search (Dynamic Radius)
        current_radius = SEARCH_RADIUS_M
        candidates = gpd.GeoDataFrame()
        while current_radius <= 500:
            search_box = box(p_t.x - current_radius, p_t.y - current_radius,
                             p_t.x + current_radius, p_t.y + current_radius)
            candidates = gdf_roads[gdf_roads.intersects(search_box)].copy()
            if not candidates.empty: break
            current_radius += 100

        if candidates.empty:
            print(f"⚠️ Point {idx} skipped (no roads).")
            continue
            
        # Find Best Match (Closest)
        candidates['dist'] = candidates.geometry.distance(p_t)
        best_match = candidates.sort_values('dist').iloc[0]
        
        # Calculate Snapped Point
        proj = best_match.geometry.project(p_t)
        snap_pt = best_match.geometry.interpolate(proj)
        
        matched_points.append({
            'gps': p_t,
            'snap': snap_pt,
            'road': best_match.geometry,
            'index': idx
        })

    if not matched_points:
        print("❌ No matches found to plot.")
        return

    # =========================
    # PLOTTING
    # =========================
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_aspect('equal')
    ax.set_facecolor('white')
    
    # 1. Determine Zoom Extents (Cover all matched points + margin)
    min_x = min(m['gps'].x for m in matched_points)
    max_x = max(m['gps'].x for m in matched_points)
    min_y = min(m['gps'].y for m in matched_points)
    max_y = max(m['gps'].y for m in matched_points)
    
    margin = 80
    ax.set_xlim(min_x - margin, max_x + margin)
    ax.set_ylim(min_y - margin, max_y + margin)

    # 2. Draw Background Roads
    # Only draw roads visible in this window to keep it clean
    view_box = box(min_x - margin, min_y - margin, max_x + margin, max_y + margin)
    visible_roads = gdf_roads[gdf_roads.intersects(view_box)]
    visible_roads.plot(ax=ax, color='#bdc3c7', linewidth=4, alpha=0.4, zorder=1)

    # 3. Draw The "Matched Path" (Connect the Green Snapped Points)
    snap_coords = [(m['snap'].x, m['snap'].y) for m in matched_points]
    if len(snap_coords) > 1:
        # Draw Arrows connecting snaps
        for i in range(len(snap_coords) - 1):
            start = snap_coords[i]
            end = snap_coords[i+1]
            ax.annotate("", xy=end, xytext=start,
                        arrowprops=dict(facecolor='#2ecc71', edgecolor='#27ae60', 
                                        width=4, headwidth=10, shrink=0.0), zorder=2)

    # 4. Draw Points & Projections
    for i, m in enumerate(matched_points):
        # A. Raw GPS (Blue)
        ax.scatter([m['gps'].x], [m['gps'].y], color='#2980b9', s=150, zorder=5, label='Raw GPS' if i==0 else "")
        ax.text(m['gps'].x + 2, m['gps'].y + 2, f"T{START_INDEX+i}", color='#2980b9', fontsize=10, fontweight='bold')
        
        # B. Snapped Point (Green)
        ax.scatter([m['snap'].x], [m['snap'].y], color='#27ae60', s=150, zorder=5, label='Matched' if i==0 else "")
        
        # C. Projection Line (Dotted)
        ax.plot([m['gps'].x, m['snap'].x], [m['gps'].y, m['snap'].y], 
                color='black', linestyle=':', linewidth=1.5, zorder=3)

    # 5. Legend & Title
    # Manual Legend to avoid duplicate labels
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2980b9', markersize=10, label='Raw GPS Point'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#27ae60', markersize=10, label='Matched Road Point'),
        plt.Line2D([0], [0], color='#2ecc71', lw=4, label='Inferred Path'),
        plt.Line2D([0], [0], color='black', linestyle=':', lw=2, label='Projection Error')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.title(f"Phase 4 Overlay: Map Matching Result (Points {START_INDEX}-{START_INDEX+WINDOW_SIZE-1})", fontsize=16)
    plt.axis('off')
    
    output_file = f"Phase4_Map_Overlay_Pts_{START_INDEX}.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"✅ Saved Map Overlay: {output_file}")
    plt.show()

if __name__ == "__main__":
    main()