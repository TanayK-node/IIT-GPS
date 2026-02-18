import geopandas as gpd
import folium
import os

# 1. Load the GeoJSON file
# Make sure "drawn_routes.geojson" is in the same folder as this script
filename = "drawn_routes.geojson"

if not os.path.exists(filename):
    print(f"Error: {filename} not found.")
    exit()

print(f"Loading {filename}...")
gdf = gpd.read_file(filename)

# 2. Set up the Map Center
# We calculate the average center of all lines to center the map automatically
center_lat = gdf.geometry.centroid.y.mean()
center_lon = gdf.geometry.centroid.x.mean()

m = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles="cartodbpositron")

# 3. Define a list of colors to cycle through
colors = [
    'red', 'blue', 'green', 'purple', 'orange', 'darkred',
    'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'pink', 'black'
]

# 4. Add each route to the map
print(f"Found {len(gdf)} route segments. Adding to map...")

for idx, row in gdf.iterrows():
    # Cycle through colors based on the index
    color = colors[idx % len(colors)]
    
    # Create a generic label
    label = f"Route Segment {idx}"
    
    # Add the geometry
    folium.GeoJson(
        row.geometry,
        name=label,
        style_function=lambda x, color=color: {
            'color': color,
            'weight': 5,
            'opacity': 0.8
        },
        tooltip=label
    ).add_to(m)

# 5. Add Layer Control to toggle lines on/off
folium.LayerControl().add_to(m)

# 6. Save the map
output_file = "view_routes.html"
m.save(output_file)

print(f"Success! Map saved as: {output_file}")
print("Open this HTML file in your browser to view the routes.")