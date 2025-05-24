import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_yield = pd.read_csv('DGS10.csv')
df_cpi = pd.read_csv('CPIAUCSL.csv')

df_yield['date'] = pd.to_datetime(df_yield['observation_date'])
df_yield['10Y_yield'] = pd.to_numeric(df_yield['DGS10'], errors='coerce')
df_yield = df_yield.set_index('date')[['10Y_yield']].resample('MS').mean().reset_index()

df_cpi['date'] = pd.to_datetime(df_cpi['observation_date'])
df_cpi['CPI'] = pd.to_numeric(df_cpi['CPIAUCSL'], errors='coerce')
df_cpi = df_cpi[['date', 'CPI']]

df = pd.merge(df_yield, df_cpi, on='date', how='inner').dropna()
df = df.set_index('date')

all_months = pd.date_range(df.index.min(), df.index.max(), freq='MS')
df = df.reindex(all_months)
df = df.dropna(subset=['10Y_yield', 'CPI'])

df['rolling_10y_inflation'] = (
    (df['CPI'].shift(-120) / df['CPI']) ** (1/10) - 1
) * 100
df['realized_real_yield'] = df['10Y_yield'] - df['rolling_10y_inflation']

df_aligned = df.dropna(subset=['rolling_10y_inflation', 'realized_real_yield'])

plt.figure(figsize=(12, 5))
plt.plot(df_aligned.index, df_aligned['10Y_yield'], label='10Y Treasury Yield (%)')
plt.plot(df_aligned.index, df_aligned['rolling_10y_inflation'], label='Realized 10Y Inflation (%)')
plt.ylabel('Percent')
plt.xlabel('Date (Bond Purchase Date)')
plt.title('10Y Treasury Yield vs. Realized 10Y Inflation')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 5))
real_yield = df_aligned['realized_real_yield']
dates = df_aligned.index

plt.plot(dates, real_yield, label='Ex Post Real Yield (%)', color='purple')
plt.fill_between(dates, real_yield, 0, where=(real_yield < 0), color='red', alpha=0.3, label='Financial Repression')
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')

plt.xlabel('Date (Bond Purchase Date)')
plt.ylabel('Percent')
plt.title('Ex Post Real Yield (Shaded = Financial Repression)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print(df_aligned[['10Y_yield', 'rolling_10y_inflation', 'realized_real_yield']].head(12))
