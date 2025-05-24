  import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load 5Y yield and CPI data
df_yield = pd.read_csv('/Users/sthoman/Documents/DGS5.csv')
df_cpi = pd.read_csv('//Users/sthoman/Documents/CPIAUCSL.csv')

# Prepare 5Y yield data
df_yield['date'] = pd.to_datetime(df_yield['observation_date'])
df_yield['5Y_yield'] = pd.to_numeric(df_yield['DGS5'], errors='coerce')
df_yield = df_yield.set_index('date')[['5Y_yield']].resample('MS').mean().reset_index()

# Prepare CPI data
df_cpi['date'] = pd.to_datetime(df_cpi['observation_date'])
df_cpi['CPI'] = pd.to_numeric(df_cpi['CPIAUCSL'], errors='coerce')
df_cpi = df_cpi[['date', 'CPI']]

# Merge and align data
df = pd.merge(df_yield, df_cpi, on='date', how='inner').dropna()
df = df.set_index('date')
all_months = pd.date_range(df.index.min(), df.index.max(), freq='MS')
df = df.reindex(all_months)
df = df.dropna(subset=['5Y_yield', 'CPI'])

# Calculate rolling 5-year (60 months) realized inflation, annualized
df['rolling_5y_inflation'] = (
    (df['CPI'].shift(-60) / df['CPI']) ** (1/5) - 1
) * 100

# Calculate ex post real yield
df['realized_real_yield'] = df['5Y_yield'] - df['rolling_5y_inflation']

df_aligned = df.dropna(subset=['rolling_5y_inflation', 'realized_real_yield'])

# Plot 5Y yield vs realized 5Y inflation
plt.figure(figsize=(12, 5))
plt.plot(df_aligned.index, df_aligned['5Y_yield'], label='5Y Treasury Yield (%)')
plt.plot(df_aligned.index, df_aligned['rolling_5y_inflation'], label='Realized 5Y Inflation (%)')
plt.ylabel('Percent')
plt.xlabel('Date (Bond Purchase Date)')
plt.title('5Y Treasury Yield vs. Realized 5Y Inflation')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot ex post real yield
plt.figure(figsize=(12, 5))
real_yield = df_aligned['realized_real_yield']
dates = df_aligned.index

plt.plot(dates, real_yield, label='Ex Post Real Yield (%)')
plt.fill_between(dates, real_yield, 0, where=(real_yield < 0), color='red', alpha=0.3, label='Financial Repression')
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')

plt.xlabel('Date (Bond Purchase Date)')
plt.ylabel('Percent')
plt.title('Ex Post Real Yield (Shaded = Financial Repression)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print(df_aligned[['5Y_yield', 'rolling_5y_inflation', 'realized_real_yield']].head(12))
