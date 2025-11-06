import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df=pd.read_csv('bmw_sales_clean.csv')
#Salse x regions
region_sales = (df.groupby('region')['sales_volume']
                .sum()
                .sort_values(ascending=False))
model_sales=(df.groupby('model', as_index=False)['sales_volume']
            .sum()
            .sort_values(by='sales_volume', ascending=False))
model_region_sales = (df.groupby(['model', 'region'])['sales_volume']
                    .sum()
                    .unstack()
                    .fillna(0))
fuel_type_sales=(df.groupby('fuel_type', as_index=False)['sales_volume']
                    .sum()
                    .sort_values(by='sales_volume', ascending=False))
transmission_time_sales=(df.groupby(['transmission', 'year'])['sales_volume']
                    .sum()
                    .unstack()
                    .fillna(0)
                    .sort_index(axis=1))
#saving dataframes to csv
region_sales.to_csv('bmw_region_sales.csv')
model_sales.to_csv('bmw_model_sales.csv')
model_region_sales.to_csv('bmw_model_region_sales.csv')
fuel_type_sales.to_csv('bmw_fuel_type_sales.csv')
transmission_time_sales.to_csv('bmw_transmission_time_sales.csv')
#visualizations
plt.figure(figsize=(10,6))
region_sales.plot(kind='bar', color='skyblue')
plt.title('BMW Sales by Region')
plt.xlabel('Region')
plt.ylabel('Total Sales Volume')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()