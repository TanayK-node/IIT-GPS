import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from shapely.geometry import box, Point

# =========================
# CONFIGURATION
# =========================
ROADS_FILENAME = "map/IITB_Auto_POV_Shapefile.gpkg"
CSV_FILENAME = "data/Discrete points.csv"  # <--- Make sure this matches your file
SEARCH_RADIUS_M = 50             # Same as your main code
MIN_CANDIDATES = 3                # We look for a point with at least this many roads

def main():
    print("--- 🕵️ Searching for a point with multiple candidates ---")
    
    # 1. Load Map
    try:
        gdf_roads = gpd.read_file(ROADS_FILENAME)
        # Ensure Metric Projection (Mumbai UTM)
        if gdf_roads.crs is None or not gdf_roads.crs.is_projected:
            gdf_roads = gdf_roads.to_crs(epsg=32643) 
    except Exception as e:
        print(f"❌ Error loading map: {e}")
        return

    # 2. Load CSV
    try:
        df = pd.read_csv(CSV_FILENAME)
        lat_col = next((c for c in df.columns if c.lower() in ['lat', 'latitude']), None)
        lon_col = next((c for c in df.columns if c.lower() in ['lon', 'longitude', 'lng']), None)
        if not lat_col: 
            print("❌ No Lat/Lon columns found.")
            return
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return

    # 3. Create Geometry for ALL points (Fast Batch Processing)
    print(f"Scanning {len(df)} points...")
    gdf_points = gpd.GeoDataFrame(
        df, 
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs="EPSG:4326"
    ).to_crs(gdf_roads.crs)

    # 4. Find the Perfect Point
    # We create a buffer around all points and check intersection counts
    # (Doing this one by one is safer for memory than a giant matrix)
    
    found_index = -1
    candidate_count = 0
    
    # Create spatial index for speed
    sindex = gdf_roads.sindex

    for idx, point in gdf_points.geometry.items():
        # Create the search box for this point
        search_box = box(point.x - SEARCH_RADIUS_M, point.y - SEARCH_RADIUS_M,
                         point.x + SEARCH_RADIUS_M, point.y + SEARCH_RADIUS_M)
        
        # Fast query: indices of roads that intersect the box
        possible_matches_index = list(sindex.intersection(search_box.bounds))
        possible_matches = gdf_roads.iloc[possible_matches_index]
        
        # Precise check (intersects the box geometry)
        precise_matches = possible_matches[possible_matches.intersects(search_box)]
        
        count = len(precise_matches)
        
        if count >= MIN_CANDIDATES:
            found_index = idx
            candidate_count = count
            print(f"✅ FOUND MATCH! Point Index {idx} has {count} candidate roads.")
            break # Stop at the first good one

    if found_index == -1:
        print(f"⚠️ No points found with >= {MIN_CANDIDATES} candidates. Try lowering the number.")
        return

    # =========================
    # PLOT THE FOUND POINT
    # =========================
    visualize_point(found_index, gdf_points.loc[found_index].geometry, gdf_roads, candidate_count)

def visualize_point(idx, point_geom, gdf_roads, count):
    print(f"🎨 Generating Visualization for Point #{idx}...")
    
    # Define Search Box
    search_box = box(point_geom.x - SEARCH_RADIUS_M, point_geom.y - SEARCH_RADIUS_M,
                     point_geom.x + SEARCH_RADIUS_M, point_geom.y + SEARCH_RADIUS_M)
    
    # Get Intersection
    highlighted_roads = gdf_roads[gdf_roads.intersects(search_box)]

    # Setup Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    ax.set_facecolor('white')

    # Zoom Limits
    zoom = 400
    ax.set_xlim(point_geom.x - zoom, point_geom.x + zoom)
    ax.set_ylim(point_geom.y - zoom, point_geom.y + zoom)

    # 1. Draw Grid (Visual only)
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    grid_size = 200
    for x in np.arange(x_min, x_max, grid_size):
        ax.plot([x, x], [y_min, y_max], color='#e0e0e0', linestyle='--', linewidth=0.8, zorder=0)
    for y in np.arange(y_min, y_max, grid_size):
        ax.plot([x_min, x_max], [y, y], color='#e0e0e0', linestyle='--', linewidth=0.8, zorder=0)

    # 2. Draw Background Roads
    gdf_roads.cx[x_min:x_max, y_min:y_max].plot(ax=ax, color='grey', linewidth=1.5, alpha=0.4, zorder=1)

    # 3. Draw Candidate Roads (Red)
    highlighted_roads.plot(ax=ax, color='red', linewidth=3, zorder=2, label='Candidate Roads')

    # 4. Draw Search Box
    rect = patches.Rectangle(
        (point_geom.x - SEARCH_RADIUS_M, point_geom.y - SEARCH_RADIUS_M), 
        SEARCH_RADIUS_M * 2, SEARCH_RADIUS_M * 2, 
        linewidth=2, edgecolor='blue', facecolor='blue', alpha=0.1, linestyle='--', zorder=3
    )
    ax.add_patch(rect)

    # 5. Draw Point
    ax.scatter([point_geom.x], [point_geom.y], color='blue', s=120, zorder=5, label='GPS Point')

    # Annotations
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    info_text = (
        f"Point Index: {idx}\n"
        f"Candidates Found: {count}\n"
        f"Search Radius: {SEARCH_RADIUS_M}m"
    )
    ax.text(0.95, 0.95, info_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', horizontalalignment='right', bbox=props)
    
    # Label the box
    ax.annotate("Spatial Index Box", 
                xy=(point_geom.x - SEARCH_RADIUS_M, point_geom.y + SEARCH_RADIUS_M), 
                xytext=(point_geom.x - 300, point_geom.y + 300),
                arrowprops=dict(facecolor='blue', shrink=0.05),
                fontsize=11, color='blue', fontweight='bold')

    plt.title(f"Phase 1: Candidate Filtering (Point #{idx})", fontsize=15)
    plt.axis('off')
    
    filename = f"Phase1_Complex_Point_{idx}.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"📸 Saved image: {filename}")
    plt.show()

if __name__ == "__main__":
    main()