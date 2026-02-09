import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. CONFIGURATION
# ==========================================
FILE_PATH = '../data/Discrete points.csv'
OUTPUT_CSV = 'Cleaned_Segmented_Trajectory.csv'
REPORT_IMG = 'preprocessing_report.png'

# Constraints
MAX_SPEED_KMPH = 140.0
MAX_ACCEL_MPS2 = 6.0
MAX_DECEL_MPS2 = -10.0
SEGMENT_GAP_THRESHOLD = 180.0  # Seconds (3 mins gap = new segment)

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def haversine_np(lon1, lat1, lon2, lat2):
    """Vectorized Haversine Distance (Meters)"""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371000
    return c * r

def create_segments(df, time_threshold=180.0):
    """
    Assigns a unique 'segment_id' to continuous driving periods.
    New Segment if:
    1. Driver changes
    2. Time gap > threshold (e.g., signal loss or parking)
    """
    df = df.copy()
    
    # Calculate time difference between consecutive points (for the same driver)
    # We must group by driver first to avoid diffing across different drivers
    df['temp_delta_t'] = df.groupby('driverId')['timestamp'].diff().dt.total_seconds().fillna(0)
    
    # Logic: New segment if driver changes OR time gap is large
    # (The first row of the DF is always the start of a segment)
    cond_driver_change = df['driverId'] != df['driverId'].shift(1)
    cond_time_gap = df['temp_delta_t'] > time_threshold
    
    is_new_segment = cond_driver_change | cond_time_gap
    
    # Cumulative sum creates a unique ID for each group
    df['segment_id'] = is_new_segment.cumsum()
    
    # Cleanup temp column
    df.drop(columns=['temp_delta_t'], inplace=True)
    
    print(f"Segmentation Complete: Created {df['segment_id'].max()} segments from {df['driverId'].nunique()} drivers.")
    return df

def calculate_kinematics(df):
    """
    Calculates Speed/Accel. 
    CRITICAL: Groups by 'segment_id' so we don't calculate speed across gaps.
    """
    df = df.copy()
    
    # 1. Shift within the SEGMENT (not just driver)
    # If it's the start of a segment, prev_lat will be NaN -> Speed will be NaN (0)
    g = df.groupby(['driverId', 'segment_id'])
    df['prev_lat'] = g['latitude'].shift(1)
    df['prev_lon'] = g['longitude'].shift(1)
    df['prev_time'] = g['timestamp'].shift(1)
    
    # 2. Deltas
    df['dist_m'] = haversine_np(df['prev_lon'], df['prev_lat'], df['longitude'], df['latitude'])
    df['delta_t'] = (df['timestamp'] - df['prev_time']).dt.total_seconds()
    
    # 3. Speed & Accel
    # Handle division by zero or very small deltas
    df['speed_mps'] = np.where(df['delta_t'] > 0, df['dist_m'] / df['delta_t'], 0)
    df['speed_kmph'] = df['speed_mps'] * 3.6
    
    # Accel requires shifting speed
    df['prev_speed'] = g['speed_mps'].shift(1)
    df['accel_mps2'] = np.where(df['delta_t'] > 0, 
                                (df['speed_mps'] - df['prev_speed']) / df['delta_t'], 
                                0)
    
    # Fill NaNs (Start of segments) with 0
    cols_to_fix = ['dist_m', 'speed_mps', 'speed_kmph', 'accel_mps2']
    df[cols_to_fix] = df[cols_to_fix].fillna(0)
    
    return df

def get_performance_metrics(df, label="Data"):
    """Generates standard trajectory quality metrics."""
    return {
        "Metric_Set": label,
        "Total_Points": len(df),
        "Num_Segments": df['segment_id'].nunique(),
        "Avg_Speed_kmph": round(df['speed_kmph'].mean(), 2),
        "Speed_Std_Dev": round(df['speed_kmph'].std(), 2), # Smoothness
        "Physics_Violations": (
            (df['speed_kmph'] > MAX_SPEED_KMPH) | 
            (df['accel_mps2'] > MAX_ACCEL_MPS2)
        ).sum()
    }

# ==========================================
# 3. MAIN PIPELINE
# ==========================================

# A. LOAD & SORT
df_raw = pd.read_csv(FILE_PATH)
df_raw['timestamp'] = pd.to_datetime(df_raw['Date'] + ' ' + df_raw['Time'])
df_raw = df_raw.sort_values(by=['driverId', 'timestamp']).reset_index(drop=True)
df_raw['accuracy'] = pd.to_numeric(df_raw['accuracy'], errors='coerce')

# B. SEGMENTATION (Step 1)
df_segmented = create_segments(df_raw, time_threshold=SEGMENT_GAP_THRESHOLD)

# C. INITIAL KINEMATICS
df_segmented = calculate_kinematics(df_segmented)
metrics_raw = get_performance_metrics(df_segmented, "Raw (Segmented)")

# D. ACCURACY FILTER (Step 2 - IQR Method)
Q1 = df_segmented['accuracy'].quantile(0.25)
Q3 = df_segmented['accuracy'].quantile(0.75)
IQR = Q3 - Q1
acc_limit = Q3 + 1.5 * IQR
df_acc_clean = df_segmented[df_segmented['accuracy'] <= acc_limit].copy()

# E. PHYSICS FILTER (Step 3)
# Re-calculate kinematics because dropping points changes neighbors!
df_acc_clean = calculate_kinematics(df_acc_clean)

bad_physics = (df_acc_clean['speed_kmph'] > MAX_SPEED_KMPH) | \
              (df_acc_clean['accel_mps2'] > MAX_ACCEL_MPS2) | \
              (df_acc_clean['accel_mps2'] < MAX_DECEL_MPS2)

df_final = df_acc_clean[~bad_physics].copy()

# Final Kinematics Recalculation (for final metrics)
df_final = calculate_kinematics(df_final)
metrics_final = get_performance_metrics(df_final, "Cleaned & Segmented")

# ==========================================
# 4. REPORTING
# ==========================================

# Print Metrics
results = pd.DataFrame([metrics_raw, metrics_final])
print("\n--- TRAJECTORY PREPROCESSING REPORT ---")
print(results.T)

# Save
df_final.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved to {OUTPUT_CSV}")

# Visualization
plt.figure(figsize=(12, 6))

# Plot 1: Outlier Reduction
plt.subplot(1, 2, 1)
plt.boxplot([df_segmented['speed_kmph'], df_final['speed_kmph']], labels=['Raw', 'Clean'])
plt.yscale('log')
plt.title('Effect of Filters on Speed Outliers')
plt.ylabel('Speed (Log Scale)')

# Plot 2: Segment Length Distribution
plt.subplot(1, 2, 2)
seg_lengths = df_final.groupby('segment_id').size()
plt.hist(seg_lengths, bins=30, color='purple', alpha=0.7)
plt.title('Distribution of Points per Segment')
plt.xlabel('Number of Points in Segment')
plt.ylabel('Count of Segments')

plt.tight_layout()
plt.savefig(REPORT_IMG)
print(f"Plot saved to {REPORT_IMG}")