import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons
from scipy.optimize import minimize

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['cmr10']
plt.rcParams["axes.formatter.use_mathtext"] = True

# --- Load data ---
df_spine = pd.read_csv('tdcr_trunk_spine.csv')
df_tdcr = pd.read_csv('tdcr_trunk_output.csv')

x_cols_spine = [col for col in df_spine.columns if col.startswith('x')]
y_cols_spine = [col for col in df_spine.columns if col.startswith('y')]
z_cols_spine = [col for col in df_spine.columns if col.startswith('z')]
x_cols_tdcr = [col for col in df_tdcr.columns if col.startswith('x')]
y_cols_tdcr = [col for col in df_tdcr.columns if col.startswith('y')]
z_cols_tdcr = [col for col in df_tdcr.columns if col.startswith('z')]

def get_axis_limits():
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
    margin = 0.2
    x_lim = [x_center - max_range/2 - margin*max_range, x_center + max_range/2 + margin*max_range]
    y_lim = [y_center - max_range/2 - margin*max_range, y_center + max_range/2 + margin*max_range]
    z_lim = [z_center - max_range/2 - margin*max_range, z_center + max_range/2 + margin*max_range]
    return x_lim, y_lim, z_lim

x_lim, y_lim, z_lim = get_axis_limits()

def fit_plane(points):
    centroid = np.mean(points, axis=0)
    _, _, vh = np.linalg.svd(points - centroid)
    normal = vh[2, :]
    e1 = vh[0, :]
    e2 = vh[1, :]
    return centroid, normal, e1, e2

def project_to_plane(points, origin, e1, e2):
    rel = points - origin
    x = np.dot(rel, e1)
    y = np.dot(rel, e2)
    return np.stack([x, y], axis=1)

def map_to_3d(xy_points, origin, e1, e2):
    return origin + xy_points[:,0][:,None]*e1 + xy_points[:,1][:,None]*e2

def spiral_cost(params, xy_points):
    a, b, x0, y0 = params
    x = xy_points[:, 0] - x0
    y = xy_points[:, 1] - y0
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    theta = np.unwrap(theta)
    idx = np.argsort(theta)
    r = r[idx]
    theta = theta[idx]
    r_pred = a * np.exp(b * theta)
    return np.sum((r - r_pred) ** 2)

def fit_log_spiral(xy_points, initial=None):
    if initial is None:
        x0, y0 = np.median(xy_points, axis=0)
        r0 = np.median(np.sqrt((xy_points[:,0]-x0)**2 + (xy_points[:,1]-y0)**2))
        initial = [r0, 0.0, x0, y0]
    res = minimize(
        spiral_cost, initial, args=(xy_points,), method='Nelder-Mead',
        options={'maxiter': 3000, 'xatol': 1e-6, 'fatol': 1e-6}
    )
    return res.x

def fit_circle_2d(xy_points):
    x = xy_points[:,0]
    y = xy_points[:,1]
    def cost(c):
        xc, yc, r = c
        return np.sum((np.sqrt((x-xc)**2 + (y-yc)**2) - r)**2)
    x0, y0 = np.mean(x), np.mean(y)
    r0 = np.mean(np.sqrt((x-x0)**2 + (y-y0)**2))
    res = minimize(cost, [x0, y0, r0])
    return res.x

def circle_error(xy_points, xc, yc, r):
    dists = np.sqrt((xy_points[:,0]-xc)**2 + (xy_points[:,1]-yc)**2)
    return np.mean(np.abs(dists - r))

def is_straight_line(points, tol=0.995):
    points = points - np.mean(points, axis=0)
    u, s, vh = np.linalg.svd(points)
    var_ratio = s[0] / np.sum(s)
    return var_ratio > tol

prev_fit_params = None
fit_params_per_frame = []
model_type_per_frame = []
line_endpoints_per_frame = []
circle_params_per_frame = []

for row_idx in range(len(df_spine)):
    xs_spine = df_spine.loc[row_idx, x_cols_spine].values
    ys_spine = df_spine.loc[row_idx, y_cols_spine].values
    zs_spine = df_spine.loc[row_idx, z_cols_spine].values
    points_spine = np.vstack([xs_spine, ys_spine, zs_spine]).T
    centroid, normal, e1, e2 = fit_plane(points_spine)
    points_2d = project_to_plane(points_spine, centroid, e1, e2)
    straight = is_straight_line(points_spine)
    if straight:
        # Fit line: use first and last points, and stop at last point
        p1 = points_spine[0]
        p2 = points_spine[-1]
        line_endpoints_per_frame.append((p1, p2))
        fit_params_per_frame.append(None)
        model_type_per_frame.append("line")
        circle_params_per_frame.append(None)
        prev_fit_params = None
    else:
        # Fit circle
        xc, yc, rc = fit_circle_2d(points_2d)
        circle_err = circle_error(points_2d, xc, yc, rc)
        # Fit spiral
        params_spiral = fit_log_spiral(points_2d, initial=prev_fit_params)
        a, b, xs, ys = params_spiral
        x_shift = points_2d[:,0] - xs
        y_shift = points_2d[:,1] - ys
        r = np.sqrt(x_shift**2 + y_shift**2)
        theta = np.arctan2(y_shift, x_shift)
        theta = np.unwrap(theta)
        idx = np.argsort(theta)
        r = r[idx]
        theta = theta[idx]
        r_pred = a * np.exp(b * theta)
        spiral_err = np.mean(np.abs(r - r_pred))
        # Choose best model
        if circle_err < spiral_err * 0.9:
            # Circle is significantly better
            circle_params_per_frame.append((xc, yc, rc, centroid, e1, e2, points_2d))
            fit_params_per_frame.append(None)
            model_type_per_frame.append("circle")
            line_endpoints_per_frame.append(None)
            prev_fit_params = None
        else:
            fit_params_per_frame.append((params_spiral, centroid, e1, e2, points_2d))
            model_type_per_frame.append("spiral")
            line_endpoints_per_frame.append(None)
            circle_params_per_frame.append(None)
            prev_fit_params = params_spiral

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(left=0.05, right=0.75, bottom=0.2, top=0.85)
slider_ax = fig.add_axes([0.2, 0.05, 0.5, 0.03])
slider = Slider(slider_ax, 'Frame Slider', 1, len(df_spine), valinit=1, valstep=1)
slider.label.set_fontsize(16)
check_ax = plt.axes([0.8, 0.45, 0.15, 0.18])
check = CheckButtons(check_ax, ['Spine', 'ROI_Points', 'Fitted Model'], [True, True, True])
for text in check.labels:
    text.set_fontsize(16)
legend_ax = plt.axes([0.8, 0.65, 0.15, 0.12])
legend_ax.axis('off')

spine_scatter = None
tdcr_scatter = None
model_line = None
legend_handles = []

def plot_row(row_idx):
    global spine_scatter, tdcr_scatter, model_line, legend_handles
    ax.clear()
    legend_ax.clear()
    legend_ax.axis('off')

    xs_spine = df_spine.loc[row_idx, x_cols_spine].values
    ys_spine = df_spine.loc[row_idx, y_cols_spine].values
    zs_spine = df_spine.loc[row_idx, z_cols_spine].values
    points_spine = np.vstack([xs_spine, ys_spine, zs_spine]).T

    xs_tdcr = df_tdcr.loc[row_idx, x_cols_tdcr].values
    ys_tdcr = df_tdcr.loc[row_idx, y_cols_tdcr].values
    zs_tdcr = df_tdcr.loc[row_idx, z_cols_tdcr].values

    tdcr_scatter = ax.scatter(xs_tdcr, ys_tdcr, zs_tdcr, c='g', marker='o', s=40, label='ROI Points', visible=check.get_status()[1])
    spine_scatter = ax.scatter(xs_spine, ys_spine, zs_spine, c='b', marker='o', s=40, label='Spine Points', visible=check.get_status()[0])

    model_type = model_type_per_frame[row_idx]
    percent_error = 0.0
    fit_label = ""
    fit_desc = ""

    if model_type == "line":
        p1, p2 = line_endpoints_per_frame[row_idx]
        model_line = ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                              'r-', linewidth=3, label='Fitted Line',
                              visible=check.get_status()[2] and check.get_status()[0])[0]
        fit_label = "Line"
    elif model_type == "circle":
        xc, yc, rc, centroid, e1, e2, points_2d = circle_params_per_frame[row_idx]
        theta_data = np.arctan2(points_2d[:,1]-yc, points_2d[:,0]-xc)
        theta_data = np.unwrap(theta_data)
        idx = np.argsort(theta_data)
        theta_sorted = theta_data[idx]
        # Only plot arc between first and last sorted points
        theta_start = theta_sorted[0]
        theta_end = theta_sorted[-1]
        theta_fit = np.linspace(theta_start, theta_end, 600)
        x_fit = xc + rc * np.cos(theta_fit)
        y_fit = yc + rc * np.sin(theta_fit)
        circle_2d = np.stack([x_fit, y_fit], axis=1)
        circle_3d = map_to_3d(circle_2d, centroid, e1, e2)
        model_line = ax.plot(circle_3d[:,0], circle_3d[:,1], circle_3d[:,2], 'r-', linewidth=3,
                              label='Fitted Circle',
                              visible=check.get_status()[2] and check.get_status()[0])[0]
        dists = np.sqrt((points_2d[:,0]-xc)**2 + (points_2d[:,1]-yc)**2)
        percent_error = (np.sum(np.abs(dists - rc)) / np.sum(dists)) * 100 if np.sum(dists) != 0 else 0
        fit_label = "Circle"
        fit_desc = f"r={rc:.2f}, Center=({xc:.2f},{yc:.2f})"
    elif model_type == "spiral":
        params, centroid, e1, e2, points_2d = fit_params_per_frame[row_idx]
        a, b, xc, yc = params
        x_shift = points_2d[:,0] - xc
        y_shift = points_2d[:,1] - yc
        r = np.sqrt(x_shift**2 + y_shift**2)
        theta = np.arctan2(y_shift, x_shift)
        theta = np.unwrap(theta)
        idx = np.argsort(theta)
        r = r[idx]
        theta = theta[idx]
        r_pred = a * np.exp(b * theta)
        abs_error = np.abs(r - r_pred)
        percent_error = (np.sum(abs_error) / np.sum(r)) * 100 if np.sum(r) != 0 else 0

        # Plot spiral only over the theta range of the data, and ensure the start/end match the first/last projected points
        theta_start = theta[0]
        theta_end = theta[-1]
        theta_fit = np.linspace(theta_start, theta_end, 600)
        r_fit = a * np.exp(b * theta_fit)
        x_fit = xc + r_fit * np.cos(theta_fit)
        y_fit = yc + r_fit * np.sin(theta_fit)
        spiral_2d = np.stack([x_fit, y_fit], axis=1)
        spiral_3d = map_to_3d(spiral_2d, centroid, e1, e2)
        model_line = ax.plot(spiral_3d[:,0], spiral_3d[:,1], spiral_3d[:,2], 'r-', linewidth=3,
                              label='Fitted Spiral',
                              visible=check.get_status()[2] and check.get_status()[0])[0]
        fit_label = "Log Spiral"
        fit_desc = f"a={a:.2f}, b={b:.2f}, Center=({xc:.2f},{yc:.2f})"

    legend_handles = []
    legend_labels = []
    if tdcr_scatter.get_visible():
        legend_handles.append(tdcr_scatter)
        legend_labels.append('ROI Points')
    if spine_scatter.get_visible():
        legend_handles.append(spine_scatter)
        legend_labels.append('Spine Points')
    if model_line is not None and model_line.get_visible():
        legend_handles.append(plt.Line2D([0], [0], color='r', lw=5))
        legend_labels.append('Fitted Model')
    legend_ax.legend(legend_handles, legend_labels, loc='upper left', fontsize=13, frameon=True)

    f = 14
    ax.set_xlabel('X', fontsize=f)
    ax.set_ylabel('Y', fontsize=f)
    ax.set_zlabel('Z', fontsize=f)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=14)
    ax.set_title(
        f'{fit_label}: {fit_desc} ; % Error: {percent_error:.2f}',
        fontsize=17
    )
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_zlim(z_lim)
    ax.grid(True)
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
        if model_line is not None:
            model_line.set_visible(status[2] and status[0])
    elif label == 'ROI_Points':
        if tdcr_scatter is not None:
            tdcr_scatter.set_visible(status[1])
    elif label == 'Fitted Model':
        if model_line is not None:
            model_line.set_visible(status[2] and status[0])
    plt.draw()

plot_row(int(slider.val) - 1)
slider.on_changed(update)
check.on_clicked(toggle_visibility)
plot_row(0)
plt.show()
