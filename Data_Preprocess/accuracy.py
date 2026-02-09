import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Data
df = pd.read_csv('../data/Discrete points.csv')

# Ensure accuracy is numeric
df['accuracy'] = pd.to_numeric(df['accuracy'], errors='coerce')
df = df.dropna(subset=['accuracy'])

# 2. Calculate Statistics
stats = df['accuracy'].describe()
p95 = df['accuracy'].quantile(0.95)
p99 = df['accuracy'].quantile(0.99)

# IQR Method (Interquartile Range)
Q1 = df['accuracy'].quantile(0.25)
Q3 = df['accuracy'].quantile(0.75)
IQR = Q3 - Q1
iqr_threshold = Q3 + 1.5 * IQR

print(f"--- Statistics ---")
print(f"Mean Accuracy: {stats['mean']:.2f} m")
print(f"Median Accuracy: {stats['50%']:.2f} m")
print(f"75th Percentile: {stats['75%']:.2f} m")  # 75% of points are better than this

print(f"\n--- Suggested Cutoffs ---")
print(f"Strict (IQR Rule): Keep data <= {iqr_threshold:.2f} m")
print(f"Balanced (95th %): Keep data <= {p95:.2f} m")
print(f"Loose (99th %):    Keep data <= {p99:.2f} m")

# 3. Visualize
plt.figure(figsize=(12, 5))

# Boxplot - Great for seeing outliers
plt.subplot(1, 2, 1)
sns.boxplot(x=df['accuracy'], color='skyblue')
plt.axvline(iqr_threshold, color='red', linestyle='--', label=f'IQR Threshold ({iqr_threshold:.1f}m)')
plt.title('Boxplot of GPS Accuracy')
plt.legend()

# Histogram - See the spread
plt.subplot(1, 2, 2)
sns.histplot(df['accuracy'], bins=50, kde=False, color='gray')
plt.axvline(p95, color='orange', linestyle='--', label=f'95th % ({p95:.1f}m)')
plt.title('Distribution of Accuracy Values')
plt.yscale('log') # Log scale helps see small bars at the end
plt.legend()

plt.tight_layout()
plt.show()