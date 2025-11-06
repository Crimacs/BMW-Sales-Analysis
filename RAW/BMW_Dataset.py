import numpy as np
import pandas as pd
df = pd.read_csv('BMW_sales_data.csv')
df.columns = (
    df.columns.str.strip()
    .str.lower()
    .str.replace(' ', '_')
)
cols_to_normalize = ['region', 'color', 'fuel_type', 'transmission', 'sales_classification']
for col in cols_to_normalize:
    df[col] = df[col].str.strip().str.lower()

df=df.astype({
    'year' : 'int64',
    'engine_size_l' : 'float64',
    'mileage_km' : 'int64',
    'price_usd' : 'float64',
    'sales_volume' : 'int64',
    'sales_classification' : 'category'
})

df.to_csv("bmw_sales_clean.csv", index=False, encoding="utf-8-sig")


