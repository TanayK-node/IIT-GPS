import pandas as pd
import matplotlib.pyplot as plt

# Read Excel file
# Replace 'your_file.xlsx' with your actual file name
df = pd.read_excel("test.xlsx")

# Assuming column names are exactly: x and dt
x = df["x"]
dt = df["dt"]

# Plot line graph
plt.figure()
plt.plot(x, dt, marker='o')

# Labels and title
plt.xlabel("X")
plt.ylabel("dt")
plt.title("Line Graph of dt vs X")

plt.grid(True)
plt.show()
