import pandas as pd
import matplotlib.pyplot as plt

# Load predicted data
df_pred = pd.read_csv("reconstructed_pm25.csv")
df_pred['date'] = pd.to_datetime(df_pred['date'])

# Load true data
df_true = pd.read_csv("cleaned_pm25.csv")
df_true['date'] = pd.to_datetime(df_true['date'])

# Merge on 'date'
df_merged = pd.merge(df_true[['date', 'pm25_mean']], 
                     df_pred[['date', 'Predicted_PM2.5']], 
                     on='date', how='inner')

# Drop NA if any
df_merged = df_merged.dropna()

# Scatterplot
plt.figure(figsize=(6, 6))
plt.scatter(df_merged['pm25_mean'], df_merged['Predicted_PM2.5'],
            alpha=0.6, edgecolor='k', color='royalblue')

max_val = max(df_merged['pm25_mean'].max(), df_merged['Predicted_PM2.5'].max()) + 5
plt.plot([0, max_val], [0, max_val], 'r--', lw=2, label='1:1 Reference')

plt.xlabel('True PM$_{2.5}$ (µg/m³)', fontsize=12)
plt.ylabel('Predicted PM$_{2.5}$ (µg/m³)', fontsize=12)
plt.title('True vs Predicted PM$_{2.5}$ - Daily Reconstruction', fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig("figure6_true_vs_predicted_pm25_scatter.png", dpi=400)
plt.show()
