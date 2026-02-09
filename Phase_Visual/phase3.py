import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
from shapely.geometry import box

# =========================
# CONFIGURATION
# =========================
ROADS_FILENAME = "../map/IITB_Auto_POV_Shapefile.gpkg"
CSV_FILENAME = "../data/Discrete points.csv"
POINT_INDEX = 4   # <--- Keep this same index
SEARCH_RADIUS_M = 100

def main():
    print(f"--- 🧹 Generating CLEAN Phase 3 Diagram for Point {POINT_INDEX} ---")

    # 1. Load Data
    try:
        gdf_roads = gpd.read_file(ROADS_FILENAME)
        if not gdf_roads.crs.is_projected:
            gdf_roads = gdf_roads.to_crs(epsg=32643)

        df = pd.read_csv(CSV_FILENAME)
        lat_col = next(c for c in df.columns if c.lower() in ['lat', 'latitude'])
        lon_col = next(c for c in df.columns if c.lower() in ['lon', 'longitude', 'lng'])
        
        row = df.iloc[POINT_INDEX]
        p_t = gpd.points_from_xy([row[lon_col]], [row[lat_col]], crs="EPSG:4326")[0]
        p_t = gpd.GeoSeries([p_t], crs="EPSG:4326").to_crs(gdf_roads.crs).iloc[0]
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # 2. Find Candidates & Sort
    search_box = box(p_t.x - SEARCH_RADIUS_M, p_t.y - SEARCH_RADIUS_M,
                     p_t.x + SEARCH_RADIUS_M, p_t.y + SEARCH_RADIUS_M)
    candidates = gdf_roads[gdf_roads.intersects(search_box)].copy()
    candidates['dist'] = candidates.geometry.distance(p_t)
    candidates = candidates.sort_values('dist').head(2)
    
    if len(candidates) < 2:
        print("❌ Need 2 candidates.")
        return

    road_a = candidates.iloc[0] # Green (Main)
    road_b = candidates.iloc[1] # Red (Alley)

    # =========================
    # PLOTTING
    # =========================
    fig, ax = plt.subplots(figsize=(14, 10)) # Wider aspect ratio
    ax.set_aspect('equal')
    ax.set_facecolor('white')

    # Zoom (Increased to 80m for spacing)
    zoom = 80
    ax.set_xlim(p_t.x - zoom, p_t.x + zoom)
    ax.set_ylim(p_t.y - zoom, p_t.y + zoom)

    # 1. Background Roads (Faint)
    gdf_roads.cx[p_t.x-zoom:p_t.x+zoom, p_t.y-zoom:p_t.y+zoom].plot(
        ax=ax, color='#ecf0f1', linewidth=3, zorder=1
    )

    # 2. Draw Main Roads
    gpd.GeoSeries([road_a.geometry]).plot(ax=ax, color='#2ecc71', linewidth=7, alpha=0.9, zorder=2)
    gpd.GeoSeries([road_b.geometry]).plot(ax=ax, color='#e74c3c', linewidth=7, alpha=0.9, zorder=2)

    # 3. Draw GPS Point
    ax.scatter([p_t.x], [p_t.y], color='#2980b9', s=250, zorder=10, edgecolors='white', linewidth=3)
    ax.text(p_t.x + 4, p_t.y, "$P_t$ (GPS)", fontsize=16, color='#2980b9', fontweight='bold', va='center')

    # --- CLEAN ANNOTATIONS ---

    # Helper: Find midpoint on line for labeling
    def get_midpoint(geom, ref_point):
        proj = geom.project(ref_point)
        return geom.interpolate(proj)

    pt_a = get_midpoint(road_a.geometry, p_t)
    pt_b = get_midpoint(road_b.geometry, p_t)

    # --- ROAD A (WINNER) ---
    # Move label to the Right/Top to avoid clutter
    ax.annotate(
        "ROAD A\n(Main Road)", 
        xy=(pt_a.x, pt_a.y), 
        xytext=(pt_a.x + 25, pt_a.y + 15),
        arrowprops=dict(arrowstyle="->", color='#27ae60', lw=2),
        fontsize=14, fontweight='bold', color='#27ae60',
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#2ecc71", lw=2)
    )

    # Stat Bar A (Full)
    bar_x, bar_y = pt_a.x + 25, pt_a.y + 5
    # Green Bar
    ax.add_patch(patches.Rectangle((bar_x, bar_y), 30, 4, color='#2ecc71', zorder=15)) 
    # Label
    ax.text(bar_x, bar_y - 6, "Weight: 13,766 (High)", fontsize=11, fontweight='bold', color='#27ae60')

    # "Bonus" Arrow pointing at the gap between Point and Road A
    mid_gap_x = (p_t.x + pt_a.x)/2
    mid_gap_y = (p_t.y + pt_a.y)/2
    ax.annotate(
        "Popularity Bonus!\n(Cost Reduced)", 
        xy=(mid_gap_x, mid_gap_y), 
        xytext=(mid_gap_x + 30, mid_gap_y - 30),
        arrowprops=dict(facecolor='#f1c40f', shrink=0.05, edgecolor='black'),
        fontsize=12, fontweight='bold', color='#f39c12', ha='center',
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#f1c40f", alpha=1.0)
    )

    # --- ROAD B (LOSER) ---
    # Move label Left/Bottom
    ax.annotate(
        "ROAD B\n(Alley)", 
        xy=(pt_b.x, pt_b.y), 
        xytext=(pt_b.x - 40, pt_b.y - 20),
        arrowprops=dict(arrowstyle="->", color='#c0392b', lw=2),
        fontsize=14, fontweight='bold', color='#c0392b',
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#e74c3c", lw=2)
    )

    # Stat Bar B (Empty)
    bar_b_x, bar_b_y = pt_b.x - 40, pt_b.y - 32
    # Outline
    ax.add_patch(patches.Rectangle((bar_b_x, bar_b_y), 30, 4, fill=False, edgecolor='#c0392b', lw=1, zorder=15))
    # Tiny Red Fill
    ax.add_patch(patches.Rectangle((bar_b_x, bar_b_y), 3, 4, color='#c0392b', zorder=15)) 
    # Label
    ax.text(bar_b_x, bar_b_y - 6, "Weight: 5 (Low)", fontsize=11, fontweight='bold', color='#c0392b')

    # --- FORMULA HEADER ---
    plt.title("Phase 3: The 'Brain' (Weighting Logic)", fontsize=20, pad=20, fontweight='bold')
    
    # Formula Box (Top Left, Transparent)
    formula_text = (
        r"$\bf{Total\ Cost} = Distance\ Cost - \bf{Popularity\ Bonus}$" + "\n"
        r"(Self-Learning: Higher Weight = Lower Cost)"
    )
    ax.text(0.02, 0.98, formula_text, transform=ax.transAxes, fontsize=16,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='#f7f9f9', edgecolor='#bdc3c7', alpha=0.9))

    plt.axis('off')
    filename = f"Phase3_Weights_Clean_Point_{POINT_INDEX}.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"✅ Saved Clean Diagram: {filename}")
    plt.show()

if __name__ == "__main__":
    main()