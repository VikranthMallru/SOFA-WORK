import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load CSV
# df = pd.read_csv('tdcr_output.csv')
df = pd.read_csv('tdcr_spine.csv')
# Find all x, y, z columns
x_cols = [col for col in df.columns if col.startswith('x')]
y_cols = [col for col in df.columns if col.startswith('y')]
z_cols = [col for col in df.columns if col.startswith('z')]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# For each row (keypress), plot all ROI points
for idx, row in df.iterrows():
    xs = [row[col] for col in x_cols]
    ys = [row[col] for col in y_cols]
    zs = [row[col] for col in z_cols]
    ax.scatter(xs, ys, zs, color='blue')

limit = 100
offset = 50
ax.set_xlim([-limit+offset, limit+offset])
ax.set_ylim([-limit+offset, limit+offset])
ax.set_zlim([-limit+offset, limit+offset])

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
# ax.set_title('3D ROI Points for Each Keypress')

# Add scaling to make axes equal
ax.set_box_aspect([1, 1, 1])  # <-- This line ensures equal scaling

plt.show()
