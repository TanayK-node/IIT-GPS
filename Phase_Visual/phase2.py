import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import math
from shapely.geometry import box, Point, LineString

# =========================
# CONFIGURATION
# =========================
ROADS_FILENAME = "map/IITB_Auto_POV_Shapefile.gpkg"
CSV_FILENAME = "data/Discrete points.csv"
POINT_INDEX = 4  # <--- REPLACE THIS with the index found in Phase 1
SEARCH_RADIUS_M = 50
MAX_VALID_DIST = 200  # Threshold for Green Check vs Red X

def fast_bearing(p1, p2):
    """Calculates angle in degrees between two points (0=North, 90=East)"""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    if dx == 0 and dy == 0: return 0.0
    rads = math.atan2(dx, dy) # atan2(dx, dy) gives 0 at North in some conventions
    degs = math.degrees(rads)
    return (degs + 360) % 360

def main():
    print(f"--- 📐 Generating Phase 2 Diagram for Point {POINT_INDEX} ---")

    # 1. Load Map & CSV
    try:
        gdf_roads = gpd.read_file(ROADS_FILENAME)
        if not gdf_roads.crs.is_projected:
            gdf_roads = gdf_roads.to_crs(epsg=32643)

        df = pd.read_csv(CSV_FILENAME)
        lat_col = next(c for c in df.columns if c.lower() in ['lat', 'latitude'])
        lon_col = next(c for c in df.columns if c.lower() in ['lon', 'longitude', 'lng'])
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # 2. Get The Specific Point (P_t) and Heading
    # We need P_prev or P_next to calculate the GPS arrow direction
    current_row = df.iloc[POINT_INDEX]
    p_t = gpd.points_from_xy([current_row[lon_col]], [current_row[lat_col]], crs="EPSG:4326")[0]
    p_t = gpd.GeoSeries([p_t], crs="EPSG:4326").to_crs(gdf_roads.crs).iloc[0]

    # Calculate GPS Heading (Course)
    gps_bearing = 0
    if POINT_INDEX + 1 < len(df):
        next_row = df.iloc[POINT_INDEX + 1]
        p_next = gpd.points_from_xy([next_row[lon_col]], [next_row[lat_col]], crs="EPSG:4326")[0]
        p_next = gpd.GeoSeries([p_next], crs="EPSG:4326").to_crs(gdf_roads.crs).iloc[0]
        gps_bearing = fast_bearing((p_t.x, p_t.y), (p_next.x, p_next.y))
    elif POINT_INDEX > 0:
        prev_row = df.iloc[POINT_INDEX - 1]
        p_prev = gpd.points_from_xy([prev_row[lon_col]], [prev_row[lat_col]], crs="EPSG:4326")[0]
        p_prev = gpd.GeoSeries([p_prev], crs="EPSG:4326").to_crs(gdf_roads.crs).iloc[0]
        gps_bearing = fast_bearing((p_prev.x, p_prev.y), (p_t.x, p_t.y))

    # 3. Find Candidates
    search_box = box(p_t.x - SEARCH_RADIUS_M, p_t.y - SEARCH_RADIUS_M,
                     p_t.x + SEARCH_RADIUS_M, p_t.y + SEARCH_RADIUS_M)
    candidates = gdf_roads[gdf_roads.intersects(search_box)].copy()

    if candidates.empty:
        print("❌ No candidates found for this point.")
        return

    # =========================
    # PLOTTING
    # =========================
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_aspect('equal')
    ax.set_facecolor('white')

    # Zoom Setup
    zoom = 60  # Zoom in very close (60m)
    ax.set_xlim(p_t.x - zoom, p_t.x + zoom)
    ax.set_ylim(p_t.y - zoom, p_t.y + zoom)

    # A. Draw Roads (Grey Background)
    gdf_roads.cx[p_t.x-zoom:p_t.x+zoom, p_t.y-zoom:p_t.y+zoom].plot(
        ax=ax, color='lightgrey', linewidth=5, alpha=0.5, zorder=1
    )

    # Loop through candidates to draw projections
    for i, (idx, row) in enumerate(candidates.iterrows()):
        geom = row.geometry
        
        # 1. Calculate Snapped Point (Projection)
        proj_dist = geom.project(p_t)
        snap_pt = geom.interpolate(proj_dist)
        dist_error = p_t.distance(snap_pt)

        # 2. Calculate Road Bearing at Snap Point
        # Take a point slightly ahead to get tangent
        pt_ahead = geom.interpolate(min(proj_dist + 1, geom.length))
        road_bearing = fast_bearing((snap_pt.x, snap_pt.y), (pt_ahead.x, pt_ahead.y))

        # --- VISUALIZATION ELEMENTS ---
        
        # Dotted Line (Projection)
        ax.plot([p_t.x, snap_pt.x], [p_t.y, snap_pt.y], 
                color='black', linestyle=':', linewidth=1.5, zorder=2)
        
        # Label Distance ("Emission Dist")
        mid_x = (p_t.x + snap_pt.x) / 2
        mid_y = (p_t.y + snap_pt.y) / 2
        ax.text(mid_x, mid_y, f"{dist_error:.1f}m", fontsize=9, color='black', 
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

        # Draw Snapped Point (S_n)
        ax.scatter([snap_pt.x], [snap_pt.y], color='purple', s=80, zorder=3)
        ax.text(snap_pt.x + 2, snap_pt.y + 2, f"S_{i+1}", fontsize=12, color='purple', fontweight='bold')

        # Road Heading Arrow
        arrow_len = 8
        dx_r = arrow_len * math.sin(math.radians(road_bearing))
        dy_r = arrow_len * math.cos(math.radians(road_bearing))
        ax.arrow(snap_pt.x, snap_pt.y, dx_r, dy_r, head_width=2, color='purple', zorder=4)

        # Checkmark vs X
        marker = "✔" if dist_error < MAX_VALID_DIST else "✘"
        color = "green" if dist_error < MAX_VALID_DIST else "red"
        ax.text(snap_pt.x - 5, snap_pt.y - 5, marker, fontsize=20, color=color, fontweight='bold', zorder=10)

    # B. Draw GPS Point (P_t)
    ax.scatter([p_t.x], [p_t.y], color='blue', s=200, zorder=5, edgecolors='white', linewidth=2, label='GPS Point')
    ax.text(p_t.x + 3, p_t.y + 3, "$P_t$", fontsize=14, color='blue', fontweight='bold')

    # GPS Heading Arrow
    arrow_len = 10
    dx_g = arrow_len * math.sin(math.radians(gps_bearing))
    dy_g = arrow_len * math.cos(math.radians(gps_bearing))
    ax.arrow(p_t.x, p_t.y, dx_g, dy_g, head_width=3, color='blue', zorder=6, label='GPS Course')

    # Heading Difference Arc (Simple visual approx)
    # Just a small text annotation for angle diff is often cleaner than drawing arcs in matplotlib
    ax.text(p_t.x - 15, p_t.y - 15, f"GPS Heading: {gps_bearing:.0f}°", color='blue', fontsize=10)

    # C. Legend & Cleanup
    plt.title(f"Phase 2: Candidate Selection (Point {POINT_INDEX})", fontsize=16)
    plt.axis('off')

    filename = f"Phase2_Candidates_Point_{POINT_INDEX}.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"✅ Saved Phase 2 Diagram: {filename}")
    plt.show()

if __name__ == "__main__":
    main()