import folium
import pandas as pd

df = pd.read_csv("yellow_reference_route_points.csv")

m = folium.Map(
    location=[df.lat.mean(), df.lon.mean()],
    zoom_start=16,
    tiles="cartodbpositron"
)

# Plot points
for _, r in df.iterrows():
    folium.CircleMarker(
        [r.lat, r.lon],
        radius=3,
        color="red",
        fill=True
    ).add_to(m)

m.save("yellow_points.html")
