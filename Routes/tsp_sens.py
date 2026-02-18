import subprocess
import re
import os
import shutil
import pandas as pd
import time
import sys  # <--- Added to fix environment issue

# ============================================================
# CONFIGURATION
# ============================================================
TSP_SCRIPT = "tsp.py"
BACKUP_SCRIPT = "tsp_backup.py"

SCENARIOS = [
    {
        "name": "Scenario A (Conservative)",
        "desc": "Top 20% Roads | 15m Accuracy",
        "EDGE_PERCENTILE": 80,  # Higher percentile = fewer roads
        "BUFFER_DISTANCE": 15
    },
    {
        "name": "Scenario B (Balanced)",
        "desc": "Top 35% Roads | 20m Accuracy",
        "EDGE_PERCENTILE": 65,
        "BUFFER_DISTANCE": 20
    },
    {
        "name": "Scenario C (Aggressive)",
        "desc": "Top 50% Roads | 30m Accuracy",
        "EDGE_PERCENTILE": 50,
        "BUFFER_DISTANCE": 30
    }
]

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def backup_script():
    if os.path.exists(TSP_SCRIPT):
        shutil.copy(TSP_SCRIPT, BACKUP_SCRIPT)
        print(f"✅ Backed up {TSP_SCRIPT} to {BACKUP_SCRIPT}")
    else:
        print(f"❌ Error: {TSP_SCRIPT} not found!")
        sys.exit(1)

def restore_script():
    if os.path.exists(BACKUP_SCRIPT):
        shutil.copy(BACKUP_SCRIPT, TSP_SCRIPT)
        os.remove(BACKUP_SCRIPT)
        print(f"✅ Restored original {TSP_SCRIPT}")

def modify_settings(percentile, buffer):
    try:
        with open(TSP_SCRIPT, 'r', encoding='utf-8') as f:
            content = f.read()

        # Regex replacement for settings
        # We look for variable = number pattern
        content = re.sub(r"EDGE_PERCENTILE\s*=\s*\d+", f"EDGE_PERCENTILE = {percentile}", content)
        content = re.sub(r"BUFFER_DISTANCE\s*=\s*\d+", f"BUFFER_DISTANCE = {buffer}", content)

        with open(TSP_SCRIPT, 'w', encoding='utf-8') as f:
            f.write(content)
            
    except Exception as e:
        print(f"❌ Error modifying settings: {e}")

def parse_output(output_str):
    """Extracts coverage stats from tsp.py stdout"""
    total_covered = 0
    total_points = 0
    
    lines = output_str.split('\n')
    for line in lines:
        # Look for "Total GPS points: 69654"
        if "Total GPS points:" in line:
            match = re.search(r"Total GPS points:\s*(\d+)", line)
            if match:
                total_points = int(match.group(1))
        
        # Look for "Route 1: 42092 (60.43%)"
        if "Route" in line and "%" in line and "(" in line:
            # Extract count: "Route 1: 42092" -> 42092
            match = re.search(r":\s*(\d+)", line)
            if match:
                total_covered += int(match.group(1))

    if total_points > 0:
        coverage_pct = (total_covered / total_points) * 100
    else:
        coverage_pct = 0
        
    return total_covered, coverage_pct

# ============================================================
# MAIN EXECUTION
# ============================================================

print("🚀 STARTING SENSITIVITY ANALYSIS...\n")
backup_script()
results = []

try:
    for scenario in SCENARIOS:
        print(f"\n------------------------------------------------")
        print(f"Running: {scenario['name']}")
        print(f"Settings: {scenario['desc']}")
        print(f"------------------------------------------------")
        
        # 1. Modify Script
        modify_settings(scenario['EDGE_PERCENTILE'], scenario['BUFFER_DISTANCE'])
        
        # 2. Run Script using sys.executable (Fixes ModuleNotFoundError)
        start_time = time.time()
        
        process = subprocess.run(
            [sys.executable, TSP_SCRIPT],  # <--- CRITICAL FIX HERE
            capture_output=True, 
            text=True
        )
        duration = time.time() - start_time
        
        # Check for failure
        if process.returncode != 0:
            print(f"❌ Error running scenario (Code {process.returncode})")
            # Print the last few lines of the error for debugging
            err_lines = process.stderr.strip().split('\n')
            print("   Last error lines:")
            for line in err_lines[-3:]:
                print(f"   | {line}")
            continue

        # 3. Parse Results
        covered_count, covered_pct = parse_output(process.stdout)
        
        print(f"   -> Finished in {duration:.1f}s")
        print(f"   -> Total Coverage: {covered_pct:.2f}% ({covered_count} points)")
        
        results.append({
            "Scenario": scenario['name'],
            "Description": scenario['desc'],
            "Percentile": scenario['EDGE_PERCENTILE'],
            "Buffer (m)": scenario['BUFFER_DISTANCE'],
            "Covered Points": covered_count,
            "Coverage %": covered_pct
        })

except Exception as e:
    print(f"\n❌ Critical Error during execution: {e}")

finally:
    restore_script()

# ============================================================
# RESULTS SUMMARY
# ============================================================

print("\n\n========================================================")
print("             SENSITIVITY ANALYSIS REPORT                ")
print("========================================================")

if not results:
    print("❌ No results were generated. Check the errors above.")
else:
    df = pd.DataFrame(results)
    
    # Display table
    print(df[['Scenario', 'Description', 'Coverage %', 'Covered Points']].to_string(index=False))
    print("\n========================================================")

    # Analysis Logic
    try:
        baseline = df.loc[df['Scenario'] == "Scenario B (Balanced)", "Coverage %"].values[0]
        conservative = df.loc[df['Scenario'] == "Scenario A (Conservative)", "Coverage %"].values[0]

        diff = baseline - conservative

        print("\n>>> CONCLUSION:")
        if diff < 5:
            print(f"✅ ROBUST SOLUTION: Dropping to conservative settings only lost {diff:.1f}% coverage.")
            print("   Recommendation: You can safely use stricter routes (Scenario A) to save fuel/time.")
        else:
            print(f"⚠️ SENSITIVE SOLUTION: Dropping to conservative settings lost {diff:.1f}% coverage.")
            print("   Recommendation: Stick to Scenario B (Balanced) to maintain revenue.")
            
    except IndexError:
        print("⚠️ Could not perform comparison (missing scenario data).")

print("========================================================")