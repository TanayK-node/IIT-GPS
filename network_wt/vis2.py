import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import pandas as pd
import geopandas as gpd
from shapely.strtree import STRtree
from shapely.geometry import LineString, MultiLineString

# =========================
# CONFIGURATION
# =========================
INPUT_MAP_FILE = "../map/IITB_Auto_POV_Shapefile.gpkg"
TRAJECTORY_FILE = "../data/Discrete points.csv"
TARGET_EDGE_INDEX = 0  # <--- CHANGE THIS to inspect a specific road (e.g., 0, 105, 200)

def main():
    # 1. Load Data (Simplified for Viz)
    print("Loading Data...")
    roads = gpd.read_file(INPUT_MAP_FILE).to_crs(epsg=32643)
    
    # Flatten Geometry to match your main script's indexing
    geoms = []
    for g in roads.geometry:
        if isinstance(g, MultiLineString):
            for sub in g.geoms: geoms.append(sub)
        else:
            geoms.append(g)
            
    # Check Index
    if TARGET_EDGE_INDEX >= len(geoms):
        print(f"Error: Index {TARGET_EDGE_INDEX} is too high. Max is {len(geoms)-1}")
        return

    target_geom = geoms[TARGET_EDGE_INDEX]
    target_len = target_geom.length
    print(f"Analyzing Edge {TARGET_EDGE_INDEX} (Length: {target_len:.1f}m)...")

    # 2. Map Match (Just for this one edge to get data)
    # We cheat and just find points close to this line to simulate matching
    df = pd.read_csv(TRAJECTORY_FILE)
    df["timestamp"] = pd.to_datetime(df["Date"] + " " + df["IST_time"], dayfirst=True)
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs=4326).to_crs(32643)
    
    # Find points within 20m of this specific edge
    # (In your real script, you'd use the Viterbi output, but this is a quick standalone viz)
    candidates = gdf[gdf.geometry.distance(target_geom) < 20]
    
    if len(candidates) < 5:
        print("Not enough points found on this edge to visualize KDE.")
        return

    # 3. Project Points to 1D
    projections = [target_geom.project(pt) for pt in candidates.geometry]
    projections = np.array(sorted(projections))

    # 4. Calculate KDE
    kde = gaussian_kde(projections, bw_method=0.15)
    x_grid = np.linspace(0, target_len, 200)
    density = kde(x_grid)
    
    # Normalize (0 to 1)
    density_norm = density / density.max()

    # 5. Find Cut Points
    threshold = 0.2
    is_active = density_norm > threshold
    transitions = np.where(np.diff(is_active.astype(int)) != 0)[0]
    cut_points = [x_grid[t] for t in transitions]

    # =========================
    # 6. PLOT
    # =========================
    plt.figure(figsize=(12, 6))
    
    # A. The Density Curve
    plt.plot(x_grid, density_norm, color='blue', lw=2, label='Traffic Density (KDE)')
    plt.fill_between(x_grid, density_norm, color='blue', alpha=0.1)
    
    # B. The Raw Points (Rug Plot)
    plt.scatter(projections, np.zeros_like(projections) + 0.03, 
                color='black', marker='|', s=100, label='GPS Points', zorder=5)
    
    # C. The Threshold Line
    plt.axhline(threshold, color='red', linestyle='--', alpha=0.5, label='Cut Threshold (20%)')
    
    # D. The Cut Points
    for cp in cut_points:
        plt.axvline(cp, color='green', lw=3, linestyle='-', label=f'CUT HERE ({cp:.1f}m)')
        
    plt.title(f"KDE Analysis for Edge {TARGET_EDGE_INDEX}", fontsize=14)
    plt.xlabel("Distance Along Road (meters)")
    plt.ylabel("Normalized Usage intensity")
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_file = f"kde_edge_{TARGET_EDGE_INDEX}.png"
    plt.savefig(output_file)
    print(f"✅ Visualization saved to {output_file}")
    plt.show()

if __name__ == "__main__":
    main()