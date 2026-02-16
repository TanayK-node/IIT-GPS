import pandas as pd
import numpy as np
import os
import glob

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    R = 6371000  # Radius of Earth in meters
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def process_file(file_path, output_folder):
    try:
        # 1. Load the data
        df = pd.read_csv(file_path)
        
        # Check if required columns exist
        required_cols = ['driverId', 'timestamp', 'latitude', 'longitude']
        if not all(col in df.columns for col in required_cols):
            print(f"Skipping {file_path}: Missing required columns.")
            return

        # 2. Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # 3. Remove Duplicates
        initial_count = len(df)
        df = df.drop_duplicates(subset=['driverId', 'timestamp'], keep='first')
        removed_count = initial_count - len(df)

        # 4. Sort by driver and timestamp
        df = df.sort_values(by=['driverId', 'timestamp'])

        # 5. Calculate Distance
        df['prev_lat'] = df.groupby('driverId')['latitude'].shift(1)
        df['prev_lon'] = df.groupby('driverId')['longitude'].shift(1)
        df['prev_timestamp'] = df.groupby('driverId')['timestamp'].shift(1)

        df['dist_m'] = haversine(
            df['prev_lat'], df['prev_lon'], 
            df['latitude'], df['longitude']
        )

        # 6. Calculate Time Difference (dt)
        df['time_diff_s'] = (df['timestamp'] - df['prev_timestamp']).dt.total_seconds()

        # Handle NaNs
        df['dist_m'] = df['dist_m'].fillna(0)
        df['time_diff_s'] = df['time_diff_s'].fillna(0)

        # 7. Calculate Speed (m/s)
        df['calculated_speed_mps'] = np.where(
            df['time_diff_s'] > 0, 
            df['dist_m'] / df['time_diff_s'], 
            0
        )

        # 8. Apply Status Logic (Threshold = 5 m/s)
        threshold_speed = 5
        df['status'] = np.where(df['calculated_speed_mps'] < threshold_speed, 'stationary', 'moving')

        # 9. Clean up helper columns (optional)
        # df = df.drop(columns=['prev_lat', 'prev_lon', 'prev_timestamp'])

        # 10. Save to output folder
        base_name = os.path.basename(file_path)
        output_path = os.path.join(output_folder, f"fixed_{base_name}")
        df.to_csv(output_path, index=False)
        
        print(f"Processed {base_name}: Removed {removed_count} duplicates. Saved to {output_path}")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# --- Main Execution ---
input_folder = 'data'  # REPLACE with your input folder path
output_folder = 'cleaned_data' # REPLACE with your desired output folder path

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Get list of all CSV files
csv_files = glob.glob(os.path.join(input_folder, "*.csv"))

print(f"Found {len(csv_files)} CSV files in '{input_folder}'. Processing...")

for file in csv_files:
    process_file(file, output_folder)

print("Batch processing complete.")