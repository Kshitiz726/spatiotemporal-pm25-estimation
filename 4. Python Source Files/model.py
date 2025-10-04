import matplotlib.pyplot as plt

# Create figure and axis
fig, ax = plt.subplots(figsize=(12, 4))

# Box definitions: label and (x, y) position
boxes = {
    "Raw Data\n(PM2.5 + Weather)": (0.1, 0.5),
    "Lag Feature Generation\n(t-1, t-2, t-3)": (0.3, 0.5),
    "Cyclical Encoding\n(hour, month)": (0.5, 0.5),
    "Z-score Scaling\n(StandardScaler)": (0.7, 0.5),
    "NumPy Arrays\nReady for Model": (0.9, 0.5)
}

# Draw boxes
for label, (x, y) in boxes.items():
    ax.text(
        x, y, label,
        fontsize=12,
        ha='center', va='center',
        bbox=dict(boxstyle="round,pad=0.5", fc="skyblue", ec="black", lw=1.5)
    )

# Draw connected arrows between boxes
positions = list(boxes.values())
for i in range(len(positions) - 1):
    start = positions[i]
    end = positions[i + 1]
    ax.annotate(
        '',
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="->", lw=2, color='black'),
        xycoords='axes fraction', textcoords='axes fraction'
    )

# Remove axis
ax.axis('off')

# Save high-resolution image for your paper
plt.tight_layout()
plt.savefig("feature_engineering_pipeline_kathmandu.png", dpi=400)
plt.show()
