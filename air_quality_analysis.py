import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

file_path = "city_day.csv"

print("Loading dataset...")
df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("Initial Data Shape (rows, columns):", df.shape)

print("\nStarting Data Cleaning...")

df = df.drop_duplicates()

numeric_columns = ['AQI', 'PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

df = df[df['AQI'] >= 0]
df = df[df['PM2.5'] >= 0]

df = df.dropna(subset=['AQI', 'PM2.5', 'City', 'Date'])

df['Date'] = pd.to_datetime(df['Date'])

print("Data Cleaning completed!")
print("Cleaned Data Shape:", df.shape)

print("\nStarting Data Preprocessing...")

df['is_false_safe'] = (df['AQI'] < 100) & (df['PM2.5'] > 15)

def get_aqi_category(aqi):
    if aqi <= 50:
        return 'Good'
    elif aqi <= 100:
        return 'Satisfactory'
    elif aqi <= 200:
        return 'Moderate'
    elif aqi <= 300:
        return 'Poor'
    elif aqi <= 400:
        return 'Very Poor'
    else:
        return 'Severe'

df['AQI_Category'] = df['AQI'].apply(get_aqi_category)

def get_pm25_risk(pm25):
    if pm25 <= 15:
        return 'Safe (<=15)'
    elif pm25 <= 30:
        return 'Moderate (16-30)'
    elif pm25 <= 60:
        return 'Poor (31-60)'
    elif pm25 <= 90:
        return 'Unhealthy (61-90)'
    else:
        return 'Hazardous (>90)'

df['PM2.5_Risk_Range'] = df['PM2.5'].apply(get_pm25_risk)

print("Data Preprocessing completed!")

print("\nGenerating Visualizations...")

sns.set_theme(style="whitegrid")

print("Generating 1. Scatter Plot...")
plt.figure(figsize=(10, 6))

normal_days = df[~df['is_false_safe']]
false_safe_days = df[df['is_false_safe']]

plt.scatter(normal_days['AQI'], normal_days['PM2.5'], 
            color='#3498db', alpha=0.5, label='Normal Days', s=20)

plt.scatter(false_safe_days['AQI'], false_safe_days['PM2.5'], 
            color='#e74c3c', alpha=0.7, label='Hidden Risk (AQI < 100 & PM2.5 > 15)', s=25)

correlation = df['AQI'].corr(df['PM2.5'])

coefficients = np.polyfit(df['AQI'], df['PM2.5'], 1)
trendline_func = np.poly1d(coefficients)

x_trend = np.linspace(df['AQI'].min(), df['AQI'].max(), 100)
plt.plot(x_trend, trendline_func(x_trend), color='#2c3e50', linestyle='--', linewidth=2, 
         label=f'Trendline (Pearson Corr: {correlation:.2f})')

plt.title('AQI vs PM2.5 Concentration: Detecting Hidden Pollution Risks', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Air Quality Index (AQI)', fontsize=12)
plt.ylabel('PM2.5 Concentration (µg/m³)', fontsize=12)
plt.legend(loc='upper left', frameon=True)
plt.tight_layout()
plt.savefig('vis1_scatter_plot.png', dpi=300)
plt.close()

print("Generating 2. Heatmap...")
plt.figure(figsize=(10, 8))

aqi_order = ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe']
pm25_order = ['Safe (<=15)', 'Moderate (16-30)', 'Poor (31-60)', 'Unhealthy (61-90)', 'Hazardous (>90)']

heatmap_data = pd.crosstab(df['AQI_Category'], df['PM2.5_Risk_Range'])

heatmap_data = heatmap_data.reindex(index=aqi_order, columns=pm25_order, fill_value=0)

sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlOrRd', linewidths=0.5, 
            cbar_kws={'label': 'Number of Recorded Days'})

plt.title('Distribution of PM2.5 Risk Ranges Across AQI Categories', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('PM2.5 Concentration Risk Tier', fontsize=12)
plt.ylabel('Official Reported AQI Tier', fontsize=12)
plt.tight_layout()
plt.savefig('vis2_heatmap.png', dpi=300)
plt.close()

print("Generating 3. Time Series Plot...")
plt.figure(figsize=(12, 6))

top_city = df['City'].value_counts().idxmax()
city_ts = df[df['City'] == top_city].copy()

city_ts['YearMonth'] = city_ts['Date'].dt.to_period('M')
ts_monthly = city_ts.groupby('YearMonth')[['AQI', 'PM2.5']].mean().reset_index()

ts_monthly['Date'] = ts_monthly['YearMonth'].dt.to_timestamp()

plt.plot(ts_monthly['Date'], ts_monthly['AQI'], color='#2980b9', linewidth=2.5, marker='o', label='Monthly Average AQI')

plt.plot(ts_monthly['Date'], ts_monthly['PM2.5'], color='#c0392b', linewidth=2.5, marker='s', label='Monthly Average PM2.5 (µg/m³)')

plt.axhline(15, color='#27ae60', linestyle='--', linewidth=1.5, label='WHO 24h PM2.5 Safety Threshold (15 µg/m³)')

plt.title(f'Time Series Comparison: Monthly Average AQI vs PM2.5 Levels\n(Representative City: {top_city})', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Timeline (Year)', fontsize=12)
plt.ylabel('Concentration / Index Value', fontsize=12)
plt.legend(loc='upper right', frameon=True)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig('vis3_time_series.png', dpi=300)
plt.close()

print("Generating 4. City Comparison Bar Chart...")
plt.figure(figsize=(12, 7))

city_stats = []
for city, group in df.groupby('City'):
    safe_aqi_days = group[group['AQI'] < 100]
    total_safe = len(safe_aqi_days)
    
    if total_safe >= 50:
        false_safe_count = len(safe_aqi_days[safe_aqi_days['PM2.5'] > 15])
        percentage = (false_safe_count / total_safe) * 100
        city_stats.append({'City': city, 'False_Safe_Percentage': percentage})

df_city_stats = pd.DataFrame(city_stats)

df_city_stats = df_city_stats.sort_values(by='False_Safe_Percentage', ascending=False)

df_top_cities = df_city_stats.head(15)

sns.barplot(x='False_Safe_Percentage', y='City', data=df_top_cities, 
            hue='City', palette='viridis', legend=False)

for index, value in enumerate(df_top_cities['False_Safe_Percentage']):
    plt.text(value + 0.5, index, f'{value:.1f}%', va='center', fontsize=10)

plt.title('Percentage of False Safe AQI Days Across Major Cities\n(Proportion of Days with AQI < 100 but PM2.5 Exceeding WHO Limits)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('False Safe Days Percentage (%)', fontsize=12)
plt.ylabel('City Name', fontsize=12)

plt.xlim(0, max(df_top_cities['False_Safe_Percentage']) + 10)
plt.tight_layout()
plt.savefig('vis4_city_comparison.png', dpi=300)
plt.close()

print("Generating 5. Violin Plot...")
plt.figure(figsize=(12, 6))

aqi_order = ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe']

sns.violinplot(x='AQI_Category', y='PM2.5', data=df, order=aqi_order, 
               hue='AQI_Category', palette='muted', legend=False, cut=0)

plt.axhline(15, color='#27ae60', linestyle='--', linewidth=2, label='WHO 24h Guideline Limit (15 µg/m³)')
plt.axhline(60, color='#e67e22', linestyle=':', linewidth=2, label='Indian NAAQS 24h Limit (60 µg/m³)')

plt.title('Distribution Density of PM2.5 Concentrations Within Official AQI Categories', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Official Reported AQI Tier', fontsize=12)
plt.ylabel('Observed PM2.5 Concentration (µg/m³)', fontsize=12)

max_y_threshold = df['PM2.5'].quantile(0.99)
plt.ylim(0, max(max_y_threshold, 200))

plt.legend(loc='upper left', frameon=True)
plt.tight_layout()
plt.savefig('vis5_violin_plot.png', dpi=300)
plt.close()

print("Generating 6. Correlation Matrix...")
plt.figure(figsize=(10, 8))

pollutants = ['AQI', 'PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
available_pollutants = [col for col in pollutants if col in df.columns]

corr_matrix = df[available_pollutants].corr()

sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, 
            linewidths=0.5, square=True, cbar_kws={'shrink': 0.8})

plt.title('Pearson Correlation Matrix: Pollutant Interactions with Aggregate AQI', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('vis6_correlation_matrix.png', dpi=300)
plt.close()

print("\nSuccess! All complete executable code flows have concluded.")
print("Research output figures saved locally as standard high-resolution PNG assets.")
