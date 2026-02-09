import pandas as pd
import numpy as np

def generate_segmented_trajectory(file_path, gap_threshold=80.0):
    """
    Reads a GPS CSV file and assigns segment IDs based on driver changes
    and time gaps.
    
    Args:
        file_path (str): Path to the input CSV file.
        gap_threshold (float): Time in seconds to define a new segment (e.g., signal loss).
        
    Returns:
        pd.DataFrame: DataFrame with a new 'segment_id' column.
    """
    # 1. Load Data
    df = pd.read_csv(file_path)
    
    # 2. Preprocessing: Create Timestamp and Sort
    df['timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df = df.sort_values(by=['driverId', 'timestamp'])
    
    # 3. Compute Time Difference (Delta t)
    # Group by driver to ensure diff is calculated within the same driver's track
    df['delta_t'] = df.groupby('driverId')['timestamp'].diff().dt.total_seconds()
    
    # 4. Segment Detection Logic
    # Condition A: Driver Change (driverId != previous driverId)
    driver_change = df['driverId'] != df['driverId'].shift(1)
    
    # Condition B: Time Gap (delta_t > threshold)
    # Note: 'delta_t' is NaN for the first point of a driver, which we treat as a start.
    time_gap = df['delta_t'] > gap_threshold
    
    # A new segment starts if either condition is met
    # (The first row is always a start)
    start_of_segment = driver_change | time_gap | (df.index == 0)
    
    # 5. Assign Segment ID
    # Cumulative sum increments the ID every time a True value is encountered
    df['segment_id'] = start_of_segment.cumsum()
    
    return df

# Usage
file_path = 'data/Discrete points.csv'
segmented_df = generate_segmented_trajectory(file_path, gap_threshold=80.0)

# Display result
print(segmented_df[['driverId', 'timestamp', 'segment_id', 'delta_t']].head())

# Save to CSV
segmented_df.to_csv('Segmented_Trajectory.csv', index=False)
print("Saved segmented dataset to 'Segmented_Trajectory.csv'")