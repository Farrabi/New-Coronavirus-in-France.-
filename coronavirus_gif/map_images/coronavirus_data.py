import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import time 
import warnings
import os
warnings.filterwarnings('ignore')

vmin, vmax=0, 20

shapefile = 'ne_10m_admin_0_countries.shp'
gdf = gpd.read_file(shapefile)[['ADMIN', 'ADM0_A3', 'geometry']]
gdf.columns = ['country', 'country_code', 'geometry']
coronavirus_data=pd.read_csv('data-preview.csv')
coronavirus_data.drop(['Province/Statelub ', 'Long'], axis=1)
print(coronavirus_data.head())









# corona_data= pd.read_excel('COVID-19-geographic-disbtribution-worldwide-2020-03-09.xls')
# corona_data_clean=corona_data.drop(['NewDeaths', 'GeoId', 'EU'], axis=1)
# corona_data_clean.sort_values(by=['DateRep'], inplace=True, ascending=True)
# dates = corona_data_clean['DateRep'].drop_duplicates()
# corona_data_clean= corona_data_clean[corona_data_clean['DateRep']=='2020-02-29 00:00:00']
# merged = gdf.merge(corona_data_clean, how='left', left_on = 'country', right_on = 'CountryExp')
# merged.fillna(0, inplace=True)
# fig=merged.plot(column='NewConfCases', legend=False, cmap='Reds',linewidth=0.2, edgecolor='0.1', norm=plt.Normalize(vmin=vmin, vmax=vmax))
# plt.show()

"""
for i in dates:	
	corona_data_clean_1= corona_data_clean[corona_data_clean['DateRep']==i]
	merged = gdf.merge(corona_data_clean_1, how='left', left_on = 'country', right_on = 'CountryExp')
	merged.fillna(0, inplace=True)
	fig=merged.plot(column='NewConfCases', legend=False, cmap='Reds',linewidth=0.2, edgecolor='0.1', norm=plt.Normalize(vmin=vmin, vmax=vmax))
	fig.axis('off')
	fig.set_title('Coronavirus new cases evolution in {}'.format(str(i)))
	filepath = os.path.join('/home/zeus/Documents/Python/coronavirus/map_images', str(i)+'_violence.png')
	chart = fig.get_figure()
	chart.savefig(filepath, dpi=300)
"""



 

 



