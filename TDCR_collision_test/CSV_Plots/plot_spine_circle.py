import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# --- Set matplotlib to use Computer Modern (default LaTeX) font ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['cmr10']  # Computer Modern Roman 10pt
plt.rcParams["axes.formatter.use_mathtext"] = True

# Load both CSV files
df_spine = pd.read_csv('tdcr_spine.csv')
df_tdcr = pd.read_csv('tdcr_output.csv')

# Identify all x, y, z columns for both files
x_cols_spine = [col for col in df_spine.columns if col.startswith('x')]
y_cols_spine = [col for col in df_spine.columns if col.startswith('y')]
z_cols_spine = [col for col in df_spine.columns if col.startswith('z')]

x_cols_tdcr = [col for col in df_tdcr.columns if col.startswith('x')]
y_cols_tdcr = [col for col in df_tdcr.columns if col.startswith('y')]
z_cols_tdcr = [col for col in df_tdcr.columns if col.startswith('z')]

# Compute global min/max for all points and all rows for consistent axes
all_x = np.concatenate([df_spine[x_cols_spine].values.flatten(), df_tdcr[x_cols_tdcr].values.flatten()])
all_y = np.concatenate([df_spine[y_cols_spine].values.flatten(), df_tdcr[y_cols_tdcr].values.flatten()])
all_z = np.concatenate([df_spine[z_cols_spine].values.flatten(), df_tdcr[z_cols_tdcr].values.flatten()])

x_min, x_max = np.nanmin(all_x), np.nanmax(all_x)
y_min, y_max = np.nanmin(all_y), np.nanmax(all_y)
z_min, z_max = np.nanmin(all_z), np.nanmax(all_z)

max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
x_center = (x_max + x_min) / 2
y_center = (y_max + y_min) / 2
z_center = (z_max + z_min) / 2

x_lim = [x_center - max_range/2, x_center + max_range/2]
y_lim = [y_center - max_range/2, y_center + max_range/2]
z_lim = [z_center - max_range/2, z_center + max_range/2]

def fit_circle_3d(p1, p2, p3):
    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)
    norm_normal = np.linalg.norm(normal)
    if norm_normal == 0:
        return None, np.inf, None, None, None
    normal = normal / norm_normal
    e1 = v1 / np.linalg.norm(v1)
    e2 = np.cross(normal, e1)
    def to_2d(p):
        return np.array([np.dot(p - p1, e1), np.dot(p - p1, e2)])
    p1_2d = to_2d(p1)
    p2_2d = to_2d(p2)
    p3_2d = to_2d(p3)
    A = np.array([
        [2*(p2_2d[0]-p1_2d[0]), 2*(p2_2d[1]-p1_2d[1])],
        [2*(p3_2d[0]-p1_2d[0]), 2*(p3_2d[1]-p1_2d[1])]
    ])
    b = np.array([
        p2_2d[0]**2 + p2_2d[1]**2 - p1_2d[0]**2 - p1_2d[1]**2,
        p3_2d[0]**2 + p3_2d[1]**2 - p1_2d[0]**2 - p1_2d[1]**2
    ])
    try:
        center_2d = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None, np.inf, None, None, None
    radius = np.linalg.norm(center_2d - p1_2d)
    center_3d = p1 + center_2d[0]*e1 + center_2d[1]*e2
    return center_3d, radius, normal, e1, e2

def generate_circle_points(center, radius, e1, e2, num_points=100):
    angles = np.linspace(0, 2*np.pi, num_points)
    circle_points = np.array([center + radius*np.cos(a)*e1 + radius*np.sin(a)*e2 for a in angles])
    return circle_points

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(left=0.05, right=0.75, bottom=0.2, top=0.85)

# Slider with larger font size for label
slider_ax = fig.add_axes([0.2, 0.05, 0.5, 0.03])
slider = Slider(slider_ax, 'Frame Slider', 1, min(len(df_spine), len(df_tdcr)), valinit=1, valstep=1)
slider.label.set_fontsize(16)  # Increase slider label font size

# Create axes for checkboxes and legend
# [0.8, 0.45, 0.15, 0.18]
check_ax = plt.axes([0.8, 0.45, 0.15, 0.18])
# check_ax = plt.axes([100,100,100,100])
check = CheckButtons(check_ax, ['Spine', 'ROI_Points', 'Fitting Circle'], [True, True, True])
check_ax.set_visible(True)


for text in check.labels:
    text.set_fontsize(16)  # Increase checkbox label font size

legend_ax = plt.axes([0.8, 0.65, 0.15, 0.12])
legend_ax.axis('off')

spine_scatter = None
tdcr_scatter = None
circle_line = None
line_fit = None
legend_handles = []

def plot_row(row_idx):
    global spine_scatter, tdcr_scatter, circle_line, line_fit, legend_handles

    ax.clear()
    legend_ax.clear()
    legend_ax.axis('off')

    # Points from spine file
    xs_spine = df_spine.loc[row_idx, x_cols_spine].values
    ys_spine = df_spine.loc[row_idx, y_cols_spine].values
    zs_spine = df_spine.loc[row_idx, z_cols_spine].values
    points_spine = np.vstack([xs_spine, ys_spine, zs_spine]).T

    # Points from TDCR file
    xs_tdcr = df_tdcr.loc[row_idx, x_cols_tdcr].values
    ys_tdcr = df_tdcr.loc[row_idx, y_cols_tdcr].values
    zs_tdcr = df_tdcr.loc[row_idx, z_cols_tdcr].values

    # Plot ROI points in green
    tdcr_scatter = ax.scatter(xs_tdcr, ys_tdcr, zs_tdcr, c='g', marker='o', label='ROI Points', visible=check.get_status()[1])

    # Plot spine points in blue
    spine_scatter = ax.scatter(xs_spine, ys_spine, zs_spine, c='b', marker='o', label='Spine Points', visible=check.get_status()[0])

    # Fit circle or line for spine points
    first_point = points_spine[0]
    middle_point = points_spine[len(points_spine)//2]
    last_point = points_spine[-1]
    center, radius, normal, e1, e2 = fit_circle_3d(first_point, middle_point, last_point)
    large_radius_threshold = 1e6

    circle_line = None
    line_fit = None
    if radius == np.inf or radius > large_radius_threshold or center is None:
        line_fit, = ax.plot([first_point[0], last_point[0]],
                            [first_point[1], last_point[1]],
                            [first_point[2], last_point[2]],
                            'r-', linewidth=2.5, label='Fitted Line', visible=check.get_status()[2] and check.get_status()[0])
        line_vec = last_point - first_point
        line_len = np.linalg.norm(line_vec)
        if line_len == 0:
            total_error = 0
            percent_error = 0
        else:
            line_unit = line_vec / line_len
            diffs = points_spine - first_point
            proj_lengths = np.dot(diffs, line_unit)
            proj_points = first_point + np.outer(proj_lengths, line_unit)
            dists = np.linalg.norm(points_spine - proj_points, axis=1)
            total_error = np.sum(np.abs(dists))
            percent_error = (total_error / (len(points_spine) * line_len)) * 100
        radius_or_length = line_len
        radius_label = "Line Length"
    else:
        circle_points = generate_circle_points(center, radius, e1, e2)
        segments = [[circle_points[i], circle_points[i+1]] for i in range(len(circle_points) - 1)]
        circle_line = Line3DCollection(segments, colors='r', linewidths=1.5, alpha=1.0)
        if check.get_status()[2] and check.get_status()[0]:
            ax.add_collection3d(circle_line)
        distances = np.linalg.norm(points_spine - center, axis=1)
        total_error = np.sum(np.abs(distances - radius))
        percent_error = (total_error / (len(points_spine) * radius)) * 100 if radius != 0 else 0
        radius_or_length = radius
        radius_label = "Radius"

    # Build external legend
    legend_handles = []
    legend_labels = []

    if tdcr_scatter.get_visible():
        legend_handles.append(tdcr_scatter)
        legend_labels.append('ROI Points')

    if spine_scatter.get_visible():
        legend_handles.append(spine_scatter)
        legend_labels.append('Spine Points')

    if check.get_status()[2] and check.get_status()[0]:
        if circle_line is not None:
            legend_handles.append(plt.Line2D([0], [0], color='r', lw=5))
            legend_labels.append('Fitted Circle')
        elif line_fit is not None:
            legend_handles.append(line_fit)
            legend_labels.append('Fitted Line')

    legend_ax.legend(legend_handles, legend_labels, loc='upper left', fontsize=13, frameon=True)  # Increased font size

    ax.set_xlabel('X')  # Axes labels remain default size
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(
        f'{radius_label}: {radius_or_length:.4f} ; '
        f'Total Error: {total_error:.4f} ; Percent Error: {percent_error:.2f}%',
        fontsize=18  # Increased title font size
    )
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_zlim(z_lim)
    try:
        ax.set_box_aspect([1, 1, 1])
    except Exception:
        pass
    plt.draw()

def update(val):
    plot_row(int(slider.val) - 1)

def toggle_visibility(label):
    status = check.get_status()
    if label == 'Spine':
        if spine_scatter is not None:
            spine_scatter.set_visible(status[0])
        if circle_line is not None:
            circle_line.set_visible(status[2] and status[0])
        if line_fit is not None:
            line_fit.set_visible(status[2] and status[0])
    elif label == 'ROI_Points':
        if tdcr_scatter is not None:
            tdcr_scatter.set_visible(status[1])
    elif label == 'Fitting Circle':
        if circle_line is not None:
            circle_line.set_visible(status[2] and status[0])
        if line_fit is not None:
            line_fit.set_visible(status[2] and status[0])
    plt.draw()
    plot_row(int(slider.val) - 1)

slider.on_changed(update)
check.on_clicked(toggle_visibility)

plot_row(0)
plt.show()
