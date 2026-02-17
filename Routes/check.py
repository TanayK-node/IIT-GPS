import geopandas as gpd

roads = gpd.read_file("../map/IITB_Auto_POV_Shapefile.gpkg")
print(roads.head())
print(roads.geom_type.unique())
print(len(roads))
