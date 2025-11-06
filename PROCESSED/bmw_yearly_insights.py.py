import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('bmw_sales_clean.csv')
yearly = (
    df.groupby('year', as_index=False)['sales_volume']
    .sum()
    .rename(columns={'sales_volume': 'total_sales'})
    .sort_values('year')
)

yearly['yoy_abs']= yearly["total_sales"].diff()
yearly['yoy_pct'] = yearly['total_sales'].pct_change()
print(yearly)

start_year = int(yearly['year'].iloc[0])
end_year = int(yearly['year'].iloc[-1])
n_years = end_year - start_year
if n_years > 0 and yearly['total_sales'].iloc[0] > 0:
    cagr = (yearly['total_sales'].iloc[-1] / yearly['total_sales'].iloc[0]) ** (1 / n_years) - 1
else:
    cagr = np.nan

print("Vendite globali per anno:" , yearly.head())
print(f"Periodo: {start_year}–{end_year}  |  Intervalli: {n_years}")
print(f"CAGR globale: {cagr:.4f}  (≈ {cagr*100:.2f}%)\n")

top_year = yearly.loc[yearly['total_sales'].idxmax()]
bot_year = yearly.loc[yearly['total_sales'].idxmin()]
print(f"Anno TOP vendite: {int(top_year['year'])}  |  total_sales = {int(top_year['total_sales'])}")
print(f"Anno BOTTOM vendite: {int(bot_year['year'])}  |  total_sales = {int(bot_year['total_sales'])}\n")

best_yoy = yearly.loc[yearly['yoy_pct'].idxmax()] if yearly['yoy_pct'].notna().any() else None
worst_yoy = yearly.loc[yearly['yoy_pct'].idxmin()] if yearly['yoy_pct'].notna().any() else None
if best_yoy is not None and worst_yoy is not None:
    print(f"Miglior YoY: {int(best_yoy['year'])}  |  yoy_abs = {int(best_yoy['yoy_abs'])}  |  yoy_pct = {best_yoy['yoy_pct']*100:.2f}%")
    print(f"Peggior YoY: {int(worst_yoy['year'])} |  yoy_abs = {int(worst_yoy['yoy_abs'])} |  yoy_pct = {worst_yoy['yoy_pct']*100:.2f}%\n")

plt.figure(figsize=(12, 6))
plt.plot(yearly['year'], yearly['total_sales'], marker='o')
plt.title('BMW Vendite Globali Annuali')
plt.xlabel('Anno')
plt.ylabel('Volume Totale Vendite')
plt.grid(True)
plt.tight_layout()
plt.show()

yearly.to_csv("yearly_global_sales.csv", index=False, encoding="utf-8-sig")
    