import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import geopandas as gpd
import folium
import branca.colormap as cm
from shapely.geometry import LineString, MultiLineString

# =========================
# CONFIGURATION
# =========================
WEIGHTS_FILE = "road_weights.json"
SHAPEFILE = "map/IITB_Auto_POV_Shapefile.gpkg"
OUTPUT_DIR = "report_assets"

# Create output folder
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    print(f"--- 📊 Generating Report Assets in '{OUTPUT_DIR}/' ---")

    # 1. Load the Brain (Weights)
    if not os.path.exists(WEIGHTS_FILE):
        print(f"Error: {WEIGHTS_FILE} not found.")
        return

    with open(WEIGHTS_FILE, 'r') as f:
        data = json.load(f)

    # Convert to DataFrame
    df = pd.DataFrame(list(data.items()), columns=['Segment_ID', 'Usage_Count'])
    df['Segment_ID'] = df['Segment_ID'].astype(int)
    df['Usage_Count'] = df['Usage_Count'].astype(int)
    df = df.sort_values(by='Usage_Count', ascending=False).reset_index(drop=True)

    # =========================
    # PART 1: TEXT STATS (Copy to Report)
    # =========================
    total_usage = df['Usage_Count'].sum()
    if not df.empty:
        top_road = df.iloc[0]
        print("\n[TEXT SUMMARY - COPY THIS]")
        print(f"Total Data Points Processed: {total_usage}")
        print(f"Total Unique Road Segments Learned: {len(df)}")
        print(f"Most Popular Segment: ID {top_road['Segment_ID']} ({top_road['Usage_Count']} hits)")
        print(f"Top 5 Segments account for {df.head(5)['Usage_Count'].sum() / total_usage:.1%} of all traffic.")
        print("-" * 30)

    # =========================
    # PART 2: DISTRIBUTION CHART (PNG)
    # =========================
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    # Plot top 15 roads
    top_n = df.head(15)
    if not top_n.empty:
        ax = sns.barplot(x="Usage_Count", y="Segment_ID", data=top_n, orient='h', 
                         order=top_n['Segment_ID'], palette="viridis")
        
        plt.title("Most Frequently Traversed Road Segments", fontsize=15)
        plt.xlabel("Usage Count (GPS Points)", fontsize=12)
        plt.ylabel("Road Segment ID", fontsize=12)
        
        # Add value labels to bars
        for i in ax.containers:
            ax.bar_label(i, padding=3)

        plt.tight_layout()
        chart_path = os.path.join(OUTPUT_DIR, "usage_distribution.png")
        plt.savefig(chart_path, dpi=300)
        print(f"✅ Saved Chart: {chart_path}")

    # =========================
    # PART 3: GEOSPATIAL HEATMAP (HTML)
    # =========================
    if os.path.exists(SHAPEFILE):
        print("\n[Generating Geospatial Heatmap...]")
        
        # Load map
        roads = gpd.read_file(SHAPEFILE)
        
        # --- CRITICAL FIX: REPROJECT TO LAT/LON ---
        if roads.crs is None:
            # Assume it's WGS84 if undefined, but this is risky
            roads.set_crs(epsg=4326, inplace=True)
        else:
            # Convert to WGS84 (Lat/Lon) regardless of input projection
            roads = roads.to_crs(epsg=4326)
        # ------------------------------------------
        
        # Create a 'count' column, default 0
        roads['usage_count'] = 0
        
        # Map weights to the shapefile index
        # We try to match index. If your shapefile uses a column like 'fid' or 'id', change 'roads.index' below.
        for idx, count in data.items():
            i = int(idx)
            if i in roads.index:
                roads.at[i, 'usage_count'] = count

        # Filter only used roads for cleaner map
        used_roads = roads[roads['usage_count'] > 0].copy()
        
        if used_roads.empty:
            print("⚠️ No matching road segments found. Check if road_weights.json IDs match the shapefile indices.")
            return

        # Calculate center
        center_lat = used_roads.geometry.centroid.y.mean()
        center_lon = used_roads.geometry.centroid.x.mean()
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles="cartodbpositron")

        # Create Color Scale
        linear = cm.LinearColormap(["#2ecc71", "#f1c40f", "#e74c3c"], 
                                   vmin=used_roads['usage_count'].min(), 
                                   vmax=used_roads['usage_count'].max())
        
        # Add roads to map
        for _, row in used_roads.iterrows():
            geoms = []
            if isinstance(row.geometry, MultiLineString):
                geoms = list(row.geometry.geoms)
            elif isinstance(row.geometry, LineString):
                geoms = [row.geometry]
                
            for geom in geoms:
                # Folium needs list of [Lat, Lon]
                # Shapely coords are (Lon, Lat) -> we must swap them (y, x)
                coords = [(y, x) for x, y in geom.coords]
                
                folium.PolyLine(
                    locations=coords,
                    color=linear(row['usage_count']),
                    weight=5,
                    opacity=0.9,
                    tooltip=f"ID: {row.name} | Count: {row['usage_count']}"
                ).add_to(m)

        # Add legend
        linear.caption = 'Traffic Frequency (GPS Points)'
        m.add_child(linear)

        map_path = os.path.join(OUTPUT_DIR, "learned_heatmap.html")
        m.save(map_path)
        print(f"✅ Saved Heatmap: {map_path}")
        print("(Open this HTML file in your browser and take a screenshot for your report)")

    else:
        print(f"⚠️ Shapefile not found at {SHAPEFILE}. Skipping map generation.")

if __name__ == "__main__":
    main()