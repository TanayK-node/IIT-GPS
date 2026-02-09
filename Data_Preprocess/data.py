import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Define Haversine Function for Distance Calculation
def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees) using numpy.
    Returns distance in meters.
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371000  # Radius of earth in meters
    return c * r

# 2. Load Data
file_path = 'data/Discrete points.csv'
df = pd.read_csv(file_path)

# 3. Preprocessing
# Combine Date and Time into a single datetime object
df['timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
# Sort by Driver and Time to ensure correct order
df = df.sort_values(by=['driverId', 'timestamp']).reset_index(drop=True)

# 4. Calculate Speed
# We shift the dataframe to get the "Previous" point for every row
df['prev_lat'] = df.groupby('driverId')['latitude'].shift(1)
df['prev_lon'] = df.groupby('driverId')['longitude'].shift(1)
df['prev_time'] = df.groupby('driverId')['timestamp'].shift(1)

# Calculate distance (meters) and time difference (seconds)
df['dist_m'] = haversine_np(df['prev_lon'], df['prev_lat'], df['longitude'], df['latitude'])
df['time_diff_s'] = (df['timestamp'] - df['prev_time']).dt.total_seconds()

# Calculate speed in km/h
df['speed_kmph'] = (df['dist_m'] / df['time_diff_s']) * 3.6

# Clean up infinite/NaN speeds (first points or 0 time difference)
df.loc[df['time_diff_s'] <= 0, 'speed_kmph'] = 0
df['speed_kmph'] = df['speed_kmph'].fillna(0)

# 5. Identify Outliers
# Define a threshold (e.g., 40 km/h)
MAX_SPEED_THRESHOLD = 40.0
df['is_outlier'] = df['speed_kmph'] > MAX_SPEED_THRESHOLD

# Print Statistics
total_points = len(df)
total_outliers = df['is_outlier'].sum()

print(f"Total points processed: {total_points}")
print(f"Total outliers detected (> {MAX_SPEED_THRESHOLD} km/h): {total_outliers}")

# 6. Visualization
# We pick the driver with the most outliers to show a meaningful example
outlier_counts = df[df['is_outlier']].groupby('driverId').size()
target_driver = outlier_counts.idxmax() if not outlier_counts.empty else df['driverId'].iloc[0]

print(f"Visualizing data for driver: {target_driver}")
driver_df = df[df['driverId'] == target_driver]

plt.figure(figsize=(14, 6))

# Plot Valid Speeds
valid_points = driver_df[~driver_df['is_outlier']]
plt.plot(valid_points['timestamp'], valid_points['speed_kmph'], 
         label='Valid Speed', color='green', marker='o', markersize=3, linestyle='-', alpha=0.6)

# Plot Outliers (Spikes)
outlier_points = driver_df[driver_df['is_outlier']]
plt.scatter(outlier_points['timestamp'], outlier_points['speed_kmph'], 
            color='red', label='Outlier (Spike)', s=50, zorder=5)

# Add threshold line
plt.axhline(y=MAX_SPEED_THRESHOLD, color='orange', linestyle='--', label=f'Threshold ({MAX_SPEED_THRESHOLD} km/h)')

plt.title(f"Speed Profile for Driver {target_driver}")
plt.xlabel("Time")
plt.ylabel("Speed (km/h)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.gcf().autofmt_xdate() # Rotate dates slightly for readability

plt.show()