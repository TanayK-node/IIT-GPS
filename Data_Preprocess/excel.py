import json
import pandas as pd
import os

# Configuration
WEIGHTS_FILE = "road_weights.json"
OUTPUT_EXCEL = "Road_Segment_Weights.xlsx"

def main():
    # 1. Check if the weights file exists
    if not os.path.exists(WEIGHTS_FILE):
        print(f"❌ Error: Could not find '{WEIGHTS_FILE}'")
        return

    print(f"Reading {WEIGHTS_FILE}...")

    # 2. Load the JSON data
    with open(WEIGHTS_FILE, "r") as f:
        data = json.load(f)

    # 3. Convert to a Pandas DataFrame
    # The JSON keys are Segment IDs, values are the Weightage (Usage Count)
    df = pd.DataFrame(list(data.items()), columns=["Segment_ID", "Weightage"])

    # 4. Clean and Sort data
    # Ensure ID is an integer for proper sorting
    df["Segment_ID"] = df["Segment_ID"].astype(int)
    df["Weightage"] = df["Weightage"].astype(int)
    
    # Sort by Weightage (Highest first) so you see important roads at the top
    df = df.sort_values(by="Weightage", ascending=False).reset_index(drop=True)

    # 5. Export to Excel
    try:
        df.to_excel(OUTPUT_EXCEL, index=False)
        print(f"✅ Success! Generated Excel file: {OUTPUT_EXCEL}")
        print(f"   - Total Segments: {len(df)}")
        print(f"   - Top Segment ID: {df.iloc[0]['Segment_ID']} (Weight: {df.iloc[0]['Weightage']})")
    except ImportError:
        print("❌ Error: You need 'openpyxl' installed to save Excel files.")
        print("   Run this command: pip install openpyxl")
    except PermissionError:
        print(f"❌ Error: Please close '{OUTPUT_EXCEL}' if it is currently open.")

if __name__ == "__main__":
    main()