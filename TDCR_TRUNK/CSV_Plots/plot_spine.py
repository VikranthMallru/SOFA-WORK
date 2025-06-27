import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

# Load your CSV file
# df = pd.read_csv('tdcr_output.csv')
df = pd.read_csv('tdcr_spine.csv')
# Identify all x, y, z columns
x_cols = [col for col in df.columns if col.startswith('x')]
y_cols = [col for col in df.columns if col.startswith('y')]
z_cols = [col for col in df.columns if col.startswith('z')]

# Compute global min/max for all points and all rows
all_x = df[x_cols].values.flatten()
all_y = df[y_cols].values.flatten()
all_z = df[z_cols].values.flatten()

x_min, x_max = np.nanmin(all_x), np.nanmax(all_x)
y_min, y_max = np.nanmin(all_y), np.nanmax(all_y)
z_min, z_max = np.nanmin(all_z), np.nanmax(all_z)

# To keep aspect ratio equal
max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
x_center = (x_max + x_min) / 2
y_center = (y_max + y_min) / 2
z_center = (z_max + z_min) / 2

x_lim = [x_center - max_range/2, x_center + max_range/2]
y_lim = [y_center - max_range/2, y_center + max_range/2]
z_lim = [z_center - max_range/2, z_center + max_range/2]

# Function to plot a row's points
def plot_row(row_idx):
    xs = df.loc[row_idx, x_cols].values
    ys = df.loc[row_idx, y_cols].values
    zs = df.loc[row_idx, z_cols].values
    ax.clear()
    ax.scatter(xs, ys, zs, c='b', marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Frame: {row_idx + 1}')
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_zlim(z_lim)
    # For matplotlib >=3.3, keep aspect ratio equal:
    try:
        ax.set_box_aspect([1,1,1])
    except Exception:
        pass
    plt.draw()

# Set up the figure and slider
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(bottom=0.2)

slider_ax = fig.add_axes([0.2, 0.05, 0.6, 0.03])
slider = Slider(slider_ax, 'Frame Slider', 1, len(df), valinit=1, valstep=1)

plot_row(0)

def update(val):
    plot_row(int(slider.val) - 1)

slider.on_changed(update)
plt.show()
