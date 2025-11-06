import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('bmw_sales_clean.csv')

# Display basic statistics
print(df.describe().T)

# Descriptive statistics for fuel_type
print(df['fuel_type'].value_counts())
# Descriptive statistics region
print(df['region'].value_counts())

#Correlation matrix
corr = df.corr(numeric_only=True)

#Aggregate
agg= (df.groupby(['year', 'region', 'model'], as_index=False)
       .agg(
        total_sales=('sales_volume', 'sum'),
        avg_price=('price_usd', 'mean'),
        avg_engine=('engine_size_l', 'mean'))
      )
 #Heatmap
sns.heatmap(agg[['total_sales', 'avg_price', 'avg_engine']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

#Analysis of price for fuel types
plt.figure(figsize=(10, 6))
sns.boxplot(data=df,  x='fuel_type', y='price_usd', palette='Set2')
plt.title('Price Distribution by Fuel Type')
plt.xlabel('Fuel Type')
plt.ylabel('Price (USD)')
plt.show()

#Test Anova for price differences among fuel types
from scipy.stats import f_oneway
f, p = f_oneway(
    df[df['fuel_type'] == 'petrol']['price_usd'],
    df[df['fuel_type'] == 'diesel']['price_usd'],
    df[df['fuel_type'] == 'electric']['price_usd'],
    df[df['fuel_type'] == 'hybrid']['price_usd']
)
print(f'F-statistic: {f:.3f}, p-value: {p:.5f}')

#Aggregate for linear regression
import statsmodels.formula.api as smf 
df_model= df_model = (
    df.groupby(['year', 'region', 'fuel_type'], as_index=False)
      .agg(total_sales=('sales_volume', 'sum'),
           avg_price=('price_usd', 'mean'))
)

print(df_model.head(), df_model.shape)

# Linear regression model
model= smf.ols('total_sales ~ avg_price + C(fuel_type) + C(region) + C(year)', data=df_model).fit(cov_type='HC3')
print(model.summary())

#VIF per multicollinearity check
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
X= pd.get_dummies(df_model[['avg_price', 'fuel_type', 'region']], drop_first=True)
X= sm.add_constant(X)
X= X.astype(float)
vif= pd.DataFrame({
    'feature': X.columns,
    'VIF': [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
})
print(vif)

#Residuals vs predicted plot
pred = model.fittedvalues
residuals = model.resid

plt.scatter(pred , residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.title('Residuals vs Predicted Values')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.show()

#Log regression model
df_model2 = df_model[(df_model['total_sales'] > 0) & (df_model['avg_price'] > 0)].copy()
df_model2['ln_sales'] = np.log(df_model2['total_sales'])
df_model2['ln_price'] = np.log(df_model2['avg_price'])

log_model= smf.ols('ln_sales ~ ln_price + C(fuel_type) + C(region) + C(year)', data=df_model2).fit(cov_type='HC3')
print(log_model.summary())

