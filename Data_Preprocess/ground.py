import pandas as pd
import numpy as np

def haversine_np(lon1, lat1, lon2, lat2):
    """Distance calculation for speed"""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a)) * 6371000 
    return c

# 1. Load Data
df = pd.read_csv('../data/Discrete points.csv')
df['timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
df = df.sort_values(by=['driverId', 'timestamp']).reset_index(drop=True)

# 2. Calculate Kinematics (Speed & Acceleration)
# Shift columns to compare Current Point vs Previous Point
df['prev_lat'] = df.groupby('driverId')['latitude'].shift(1)
df['prev_lon'] = df.groupby('driverId')['longitude'].shift(1)
df['prev_time'] = df.groupby('driverId')['timestamp'].shift(1)

# Calculate Deltas
df['dist_m'] = haversine_np(df['prev_lon'], df['prev_lat'], df['longitude'], df['latitude'])
df['time_diff_s'] = (df['timestamp'] - df['prev_time']).dt.total_seconds()

# A. SPEED Calculation
df['speed_mps'] = df['dist_m'] / df['time_diff_s'] # meters per second
df['speed_kmph'] = df['speed_mps'] * 3.6           # km per hour

# B. ACCELERATION Calculation
# Acceleration = (Current Speed - Previous Speed) / Time Difference
df['prev_speed_mps'] = df.groupby('driverId')['speed_mps'].shift(1)
df['accel_mps2'] = (df['speed_mps'] - df['prev_speed_mps']) / df['time_diff_s']

# 3. Apply Ground Knowledge Thresholds
# These thresholds are based on passenger vehicle physics
MAX_SPEED = 140.0       # km/h (Conservative highway max)
MAX_ACCEL = 6.0         # m/s^2 (0-100 km/h in ~4.6s is a sports car limit)
MAX_DECEL = -10.0       # m/s^2 (Hard emergency braking is ~9.8 m/s^2)

# Create boolean masks for impossible physics
impossible_speed = df['speed_kmph'] > MAX_SPEED
impossible_accel = df['accel_mps2'] > MAX_ACCEL 
impossible_decel = df['accel_mps2'] < MAX_DECEL

# Combine filters
df['is_physics_error'] = impossible_speed | impossible_accel | impossible_decel

# 4. Filter the Data
df_clean = df[~df['is_physics_error']].copy()

print(f"--- Ground Knowledge Filtering Results ---")
print(f"Original Points: {len(df)}")
print(f"Impossible Speed (> {MAX_SPEED} km/h): {impossible_speed.sum()}")
print(f"Impossible Accel (> {MAX_ACCEL} m/s²): {impossible_accel.sum()}")
print(f"Impossible Decel (< {MAX_DECEL} m/s²): {impossible_decel.sum()}")
print(f"Cleaned Points:  {len(df_clean)}")

# Save
df_clean.to_csv('Cleaned_Physics_Method.csv', index=False)