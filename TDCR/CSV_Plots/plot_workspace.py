import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['cmr10']  # Computer Modern Roman 10pt
plt.rcParams['axes.formatter.use_mathtext'] = True
plt.rcParams['font.size'] = 14  # Set global font size very large for all elements
# --- Load Data ---
df = pd.read_csv('tdcr_spine.csv')
x_cols = [col for col in df.columns if col.startswith('x')]
y_cols = [col for col in df.columns if col.startswith('y')]
z_cols = [col for col in df.columns if col.startswith('z')]

# Gather all points into a single array
all_points = []
for idx, row in df.iterrows():
    xs = [row[col] for col in x_cols]
    ys = [row[col] for col in y_cols]
    zs = [row[col] for col in z_cols]
    for x, y, z in zip(xs, ys, zs):
        all_points.append([x, y, z])
all_points = np.array(all_points)

# Use first point as axis point
axis_point = all_points[0]

# Compute axis limits for equal aspect ratio
def get_limits(points):
    x_min, y_min, z_min = np.min(points, axis=0)
    x_max, y_max, z_max = np.max(points, axis=0)
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
    x_center = (x_max + x_min) / 2
    y_center = (y_max + y_min) / 2
    z_center = (z_max + z_min) / 2
    return [x_center - max_range/2, x_center + max_range/2], \
           [y_center - max_range/2, y_center + max_range/2], \
           [z_center - max_range/2, z_center + max_range/2]

# Rotation function
def rotate_points(points, angle_deg, axis_point, axis='y'):
    theta = np.deg2rad(angle_deg)
    translated = points - axis_point
    c, s = np.cos(theta), np.sin(theta)
    if axis == 'y':
        R = np.array([[c, 0, s],
                      [0, 1, 0],
                      [-s, 0, c]])
    else:
        raise ValueError("Only y-axis rotation implemented")
    rotated = translated @ R.T
    return rotated + axis_point

# --- Precompute workspace points for all rotations ---
angles = np.arange(0, 360, 1)
cos_vals = np.cos(np.deg2rad(angles))
sin_vals = np.sin(np.deg2rad(angles))
translated = all_points - axis_point
workspace_points = []
for c, s in zip(cos_vals, sin_vals):
    R = np.array([[c, 0, s],
                  [0, 1, 0],
                  [-s, 0, c]])
    rotated = translated @ R.T
    workspace_points.append(rotated + axis_point)
workspace_points = np.vstack(workspace_points)

# --- Set up plot ---
fig = plt.figure(figsize=(9, 8))
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(bottom=0.20)

slider_ax = fig.add_axes([0.2, 0.08, 0.6, 0.03])
slider = Slider(slider_ax, 'Rotation (deg)', 0, 359, valinit=0, valstep=1)

button_ax = fig.add_axes([0.81, 0.015, 0.14, 0.05])
toggle_button = Button(button_ax, 'Toggle Mode')

# --- State ---
mode = {'show_workspace': False}
sc = None

# --- Plotting functions ---
def plot_slider_mode(angle):
    ax.clear()
    rotated = rotate_points(all_points, angle, axis_point, axis='y')
    xlim, ylim, zlim = get_limits(workspace_points)
    ax.scatter(rotated[:,0], rotated[:,1], rotated[:,2], color='blue', s=2)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    # Hide x-axis and its values
    ax.set_xlabel('')
    ax.set_xticks([])
    try:
        ax.xaxis.set_visible(False)
    except Exception:
        pass
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Workspace')
    try:
        ax.set_box_aspect([1, 1, 1])
    except Exception:
        pass
    fig.canvas.draw_idle()

def plot_workspace_mode():
    ax.clear()
    xlim, ylim, zlim = get_limits(workspace_points)
    ax.scatter(workspace_points[:,0], workspace_points[:,1], workspace_points[:,2],
               color='blue', s=1, alpha=0.4, rasterized=True)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Full Robot Workspace')
    try:
        ax.set_box_aspect([1, 1, 1])
    except Exception:
        pass
    fig.canvas.draw_idle()

# --- Event handlers ---
def on_slider(val):
    if not mode['show_workspace']:
        plot_slider_mode(slider.val)

def on_toggle(event):
    mode['show_workspace'] = not mode['show_workspace']
    if mode['show_workspace']:
        slider_ax.set_visible(False)
        plot_workspace_mode()
        toggle_button.label.set_text('Slider Mode')
    else:
        slider_ax.set_visible(True)
        plot_slider_mode(slider.val)
        toggle_button.label.set_text('Workspace Mode')
    fig.canvas.draw_idle()

slider.on_changed(on_slider)
toggle_button.on_clicked(on_toggle)

# --- Initial plot ---
plot_slider_mode(0)

plt.show()
