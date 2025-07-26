import pandas as pd
import matplotlib.pyplot as plt

# Load datasets
df_pred = pd.read_csv("pm25_reconstruction_output.csv")
df_true = pd.read_csv("final_cleaned_dataset.csv")

# Convert datetime columns to datetime type if not already
df_pred['datetime'] = pd.to_datetime(df_pred['datetime'])
df_true['datetime'] = pd.to_datetime(df_true['datetime'])

# Merge on datetime to align predictions with true values
df_merged = pd.merge(df_true[['datetime', 'pm25']], df_pred[['datetime', 'pm25_predicted']],
                     on='datetime', how='inner')

# Scatter plot
plt.figure(figsize=(8, 8))
plt.scatter(df_merged['pm25'], df_merged['pm25_predicted'], alpha=0.5, edgecolor='k')

# Plot 45-degree line for perfect prediction
min_val = min(df_merged['pm25'].min(), df_merged['pm25_predicted'].min())
max_val = max(df_merged['pm25'].max(), df_merged['pm25_predicted'].max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)

plt.xlabel('True PM2.5 (µg/m³)')
plt.ylabel('Predicted PM2.5 (µg/m³)')
plt.title('Scatterplot: Hourly True vs Predicted PM2.5')
plt.grid(True)
plt.tight_layout()

plt.savefig('scatterplot_pm25_true_vs_predicted.png', dpi=300)
plt.show()
