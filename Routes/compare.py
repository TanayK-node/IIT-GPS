import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# SETTINGS
# ============================================================
TSP_FILE = "iitb_driver_allocation_spatial.xlsx"
MANUAL_FILE = "final_manual_allocation.xlsx"
TOTAL_GPS_POINTS = 69654  # From your previous logs

# ============================================================
# 1. LOAD DATA
# ============================================================
def load_data(file, name):
    try:
        df = pd.read_excel(file)
        df['Strategy'] = name
        return df
    except FileNotFoundError:
        print(f"Error: Could not find {file}. Make sure you ran both scripts first.")
        return None

tsp_df = load_data(TSP_FILE, "TSP (Algorithmic)")
manual_df = load_data(MANUAL_FILE, "Manual (Intuitive)")

if tsp_df is None or manual_df is None:
    exit()

# Combine for analysis
combined = pd.concat([tsp_df, manual_df])

# ============================================================
# 2. KEY METRICS CALCULATION
# ============================================================

def calculate_metrics(df, strategy_name):
    total_covered = df['demand'].sum()
    coverage_pct = (total_covered / TOTAL_GPS_POINTS) * 100
    
    # Gini Coefficient for Balance (0 = Perfect Equality, 1 = High Inequality)
    # We measure inequality in "Passengers per Driver"
    # Ideally, every driver should serve roughly the same number of passengers.
    
    # Avoid div by zero
    df = df[df['allocated_drivers'] > 0]
    passengers_per_driver = df['demand'] / df['allocated_drivers']
    
    # Simple Variance as a proxy for balance
    balance_std = passengers_per_driver.std()
    avg_load = passengers_per_driver.mean()
    
    return {
        "Strategy": strategy_name,
        "Total Covered Demand": total_covered,
        "Coverage %": f"{coverage_pct:.2f}%",
        "Avg Passengers/Driver": f"{avg_load:.2f}",
        "Load Imbalance (StdDev)": f"{balance_std:.2f}" 
        # Lower StdDev is better (means drivers work equally hard)
    }

metrics = []
metrics.append(calculate_metrics(tsp_df, "TSP (Algorithmic)"))
metrics.append(calculate_metrics(manual_df, "Manual (Intuitive)"))

results_df = pd.DataFrame(metrics)

print("\n========================================================")
print("             STRATEGY COMPARISON RESULTS                ")
print("========================================================")
print(results_df.to_string(index=False))
print("========================================================\n")

# ============================================================
# 3. VISUALIZATION
# ============================================================

sns.set_style("whitegrid")
plt.figure(figsize=(14, 6))

# Plot 1: Hourly Demand Comparison
plt.subplot(1, 2, 1)
hourly_total = combined.groupby(['hour', 'Strategy'])['demand'].sum().reset_index()
sns.lineplot(data=hourly_total, x='hour', y='demand', hue='Strategy', marker="o", linewidth=2.5)
plt.title("Hourly Demand Coverage: TSP vs Manual")
plt.ylabel("Passengers Covered")
plt.xlabel("Hour of Day")
plt.xticks(range(0, 24, 2))

# Plot 2: Route Load Distribution (Boxplot)
# Shows if some routes are overloaded while others are empty
plt.subplot(1, 2, 2)
combined['load'] = combined['demand'] / combined['allocated_drivers']
sns.boxplot(data=combined, x='Strategy', y='load', palette="Set2")
plt.title("Driver Workload Distribution (Passengers per Driver)")
plt.ylabel("Passengers / Driver (Lower spread is better)")

plt.tight_layout()
plt.savefig("comparison_analysis.png")
print("Saved comparison chart as: comparison_analysis.png")

# ============================================================
# 4. AUTOMATED RECOMMENDATION
# ============================================================

tsp_cov = float(metrics[0]["Coverage %"].replace("%",""))
man_cov = float(metrics[1]["Coverage %"].replace("%",""))
tsp_bal = float(metrics[0]["Load Imbalance (StdDev)"])
man_bal = float(metrics[1]["Load Imbalance (StdDev)"])

print("\n>>> FINAL RECOMMENDATION:")

if tsp_cov > man_cov + 5:
    print(f"✅ RECOMMENDED: TSP STRATEGY")
    print(f"   Reason: TSP covers significantly more demand (+{tsp_cov - man_cov:.1f}%).")
    print("   Even if routes are complex, the efficiency gain is too high to ignore.")

elif man_cov > tsp_cov:
    print(f"✅ RECOMMENDED: MANUAL STRATEGY")
    print(f"   Reason: Manual routes actually cover more demand ({man_cov:.1f}% vs {tsp_cov:.1f}%).")
    print("   Since they are also simpler to drive, this is a clear win.")

else:
    # If coverage is similar (within 5%), look at Balance
    if tsp_bal < man_bal:
        print(f"✅ RECOMMENDED: TSP STRATEGY")
        print("   Reason: Coverage is similar, but TSP balances driver workload better.")
        print("   (Drivers won't complain about unfair routes).")
    else:
        print(f"✅ RECOMMENDED: MANUAL STRATEGY")
        print("   Reason: Coverage and Balance are similar.")
        print("   Manual routes are preferred because they are easier for humans to learn.")