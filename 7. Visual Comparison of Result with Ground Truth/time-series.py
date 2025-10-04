import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv('ground_data.csv')


df['date'] = pd.to_datetime(df['date'])


plt.figure(figsize=(14, 7))


plt.plot(df['date'], df['Predicted_PM2.5'], label='Predicted PM2.5', color='blue', linewidth=2)


plt.plot(df['date'], df['pm25_mean'], label='PM25 Mean', color='orange', linewidth=2)


plt.xlabel('Date')
plt.ylabel('PM2.5 Concentration (µg/m³)')
plt.title('Time-Series Comparison: Predicted PM2.5 vs PM25 Mean')


plt.legend()


plt.xticks(rotation=45)


plt.tight_layout()
plt.show()
