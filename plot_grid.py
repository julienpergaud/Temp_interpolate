import pandas as pd
import geopandas as gpd


df=pd.read_csv('Results/TEMP_MLR_grid_2019-01-13_20_00_00.csv',delimiter=',')
dfpoint=pd.read_csv('descriptors_grid_TheMa.csv',delimiter=';') 
dfFus=pd.concat([df,dfpoint],axis=1)
gdf = gpd.GeoDataFrame(dfFus, geometry=gpd.points_from_xy(dfFus.LON, dfFus.LAT), crs="EPSG:4326") 


