import matplotlib.pyplot as plt
import numpy as np
import csv

x = []
y = []
time = []

# Read data from CSV
with open('data.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        time.append(float(row['time']))  # Convert to float for plotting
        x.append(float(row['x']))        # Convert to float for plotting
        y.append(float(row['y']))        # Convert to float for plotting

# Convert to numpy arrays
xArr = np.array(x)
yArr = np.array(y)
timeArr = np.array(time)

# Create subplots - 2 rows, 2 columns
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: X vs Y trajectory
ax1.plot(xArr, yArr, 'b-', alpha=0.7, linewidth=1)
ax1.scatter(xArr[0], yArr[0], color='green', s=50, label='Start', zorder=5)
ax1.scatter(xArr[-1], yArr[-1], color='red', s=50, label='End', zorder=5)
ax1.set_xlabel('X Position')
ax1.set_ylabel('Y Position')
ax1.set_title('X vs Y Trajectory')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.axis('equal')  # Equal aspect ratio for better trajectory visualization

# Plot 2: X vs Time
ax2.plot(timeArr, xArr, 'r-', linewidth=1.5)
ax2.set_xlabel('Time')
ax2.set_ylabel('X Position')
ax2.set_title('X Position vs Time')
ax2.grid(True, alpha=0.3)

# Plot 3: Y vs Time
ax3.plot(timeArr, yArr, 'g-', linewidth=1.5)
ax3.set_xlabel('Time')
ax3.set_ylabel('Y Position')
ax3.set_title('Y Position vs Time')
ax3.grid(True, alpha=0.3)

# Plot 4: Both X and Y vs Time on same plot
ax4.plot(timeArr, xArr, 'r-', label='X Position', linewidth=1.5)
ax4.plot(timeArr, yArr, 'g-', label='Y Position', linewidth=1.5)
ax4.set_xlabel('Time')
ax4.set_ylabel('Position')
ax4.set_title('X and Y Position vs Time')
ax4.grid(True, alpha=0.3)
ax4.legend()

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plots
plt.show()

# Print some basic statistics
print(f"\nData Statistics:")
print(f"Total data points: {len(timeArr)}")
print(f"Time range: {timeArr.min():.2f} to {timeArr.max():.2f}")
print(f"X range: {xArr.min():.2f} to {xArr.max():.2f}")
print(f"Y range: {yArr.min():.2f} to {yArr.max():.2f}")
print(f"Duration: {timeArr.max() - timeArr.min():.2f} time units")