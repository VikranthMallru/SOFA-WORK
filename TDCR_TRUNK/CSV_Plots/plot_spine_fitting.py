import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons
from scipy.optimize import minimize, Bounds

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

# --- Reference plane for 2D projection (from first frame) ---
xs0 = df_spine.loc[0, x_cols_spine].values
ys0 = df_spine.loc[0, y_cols_spine].values
zs0 = df_spine.loc[0, z_cols_spine].values
points0 = np.vstack([xs0, ys0, zs0]).T
ref_centroid, ref_normal, ref_e1, ref_e2 = fit_plane(points0)

x2d_lim = (-150, 150)
y2d_lim = (-150, 150)

def safe_exp(x):
    x = np.clip(x, -20, 20)
    return np.exp(x)

def fit_log_spiral_explicit(x, y, maxiter=5000):
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) < 5:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    x0_init = np.mean(x)
    y0_init = np.mean(y)
    theta = np.unwrap(np.arctan2(y - y0_init, x - x0_init))
    r = np.sqrt((x - x0_init)**2 + (y - y0_init)**2)
    r[r <= 0] = 1e-6
    p = np.polyfit(theta, np.log(r + 1e-12), 1)
    a0 = np.exp(p[1])
    b0 = p[0]
    theta0 = 0.0
    params0 = [x0_init, y0_init, np.log(a0), b0, theta0]
    min_x, max_x = np.min(x), np.max(x)
    min_y, max_y = np.min(y), np.max(y)
    bounds = Bounds(
        [min_x, min_y, -10, -2, -2*np.pi],
        [max_x, max_y, 10, 2, 2*np.pi]
    )
    def spiral_cost(params):
        x0, y0, loga, b, theta_off = params
        theta_data = np.unwrap(np.arctan2(y - y0, x - x0))
        r_data = np.sqrt((x - x0)**2 + (y - y0)**2)
        model_log_r = loga + b * (theta_data + theta_off)
        return np.sum((np.log(r_data + 1e-12) - model_log_r)**2)
    res = minimize(
        spiral_cost, params0, method='L-BFGS-B', bounds=bounds,
        options={'maxiter': maxiter}
    )
    x0_fit, y0_fit, loga_fit, b_fit, theta_off_fit = res.x
    a_fit = np.exp(np.clip(loga_fit, -20, 20))
    return x0_fit, y0_fit, a_fit, b_fit, theta_off_fit

def calculate_rms_percent_error(x, y, x0, y0, a, b, theta_off):
    theta_data = np.unwrap(np.arctan2(y - y0, x - x0))
    r_data = np.sqrt((x - x0)**2 + (y - y0)**2)
    r_model = a * safe_exp(b * (theta_data + theta_off))
    rms = np.sqrt(np.mean((r_data - r_model)**2))
    mean_radius = np.mean(r_data)
    if mean_radius < 1e-8:
        return np.nan
    return 100 * rms / mean_radius

def is_straight(points_2d, angle_thresh=0.5, svd_ratio_thresh=20):
    points_centered = points_2d - np.mean(points_2d, axis=0)
    u, s, vh = np.linalg.svd(points_centered)
    svd_ratio = s[0] / (s[1] + 1e-8)
    centroid = np.mean(points_2d, axis=0)
    thetas = np.arctan2(points_2d[:,1] - centroid[1], points_2d[:,0] - centroid[0])
    angle_span = np.max(thetas) - np.min(thetas)
    return svd_ratio > svd_ratio_thresh or angle_span < angle_thresh

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(left=0.05, right=0.75, bottom=0.2, top=0.85)
slider_ax = fig.add_axes([0.2, 0.05, 0.5, 0.03])
slider = Slider(slider_ax, 'Frame Slider', 1, len(df_spine), valinit=1, valstep=1)
slider.label.set_fontsize(16)
check_ax = plt.axes([0.8, 0.45, 0.15, 0.18])
check = CheckButtons(check_ax, ['Spine', 'ROI_Points', '2D Mode', 'Show Spiral'], [True, True, False, True])
for text in check.labels:
    text.set_fontsize(16)
legend_ax = plt.axes([0.8, 0.65, 0.15, 0.12])
legend_ax.axis('off')
spine_scatter = None
tdcr_scatter = None
spiral_line = None
legend_handles = []

def plot_row(row_idx, mode_2d, show_spiral):
    global spine_scatter, tdcr_scatter, spiral_line, legend_handles, ax
    fig.delaxes(ax)
    if not mode_2d:
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(111)
    legend_ax.clear()
    legend_ax.axis('off')
    xs_spine = df_spine.loc[row_idx, x_cols_spine].values
    ys_spine = df_spine.loc[row_idx, y_cols_spine].values
    zs_spine = df_spine.loc[row_idx, z_cols_spine].values
    points_spine = np.vstack([xs_spine, ys_spine, zs_spine]).T
    xs_tdcr = df_tdcr.loc[row_idx, x_cols_tdcr].values
    ys_tdcr = df_tdcr.loc[row_idx, y_cols_tdcr].values
    zs_tdcr = df_tdcr.loc[row_idx, z_cols_tdcr].values
    points_tdcr = np.vstack([xs_tdcr, ys_tdcr, zs_tdcr]).T
    status = check.get_status()
    rms_percent = None
    spiral_line = None
    if not mode_2d:
        spine_scatter = ax.scatter(xs_spine, ys_spine, zs_spine, c='b', marker='o', s=40, label='Spine Points', visible=status[0])
        tdcr_scatter = ax.scatter(xs_tdcr, ys_tdcr, zs_tdcr, c='g', marker='o', s=40, label='ROI Points', visible=status[1])
        ax.set_xlabel('X', fontsize=14)
        ax.set_ylabel('Y', fontsize=14)
        ax.set_zlabel('Z', fontsize=14)
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_zlim(z_lim)
        try:
            ax.set_box_aspect([1, 1, 1])
        except Exception:
            pass
        ax.set_title('3D Spine and ROI Visualization', fontsize=16)
    else:
        spine_2d = project_to_plane(points_spine, ref_centroid, ref_e1, ref_e2)
        tdcr_2d = project_to_plane(points_tdcr, ref_centroid, ref_e1, ref_e2)
        spine_scatter = ax.scatter(spine_2d[:, 0], spine_2d[:, 1], c='b', marker='o', s=40, label='Spine Points', visible=status[0])
        tdcr_scatter = ax.scatter(tdcr_2d[:, 0], tdcr_2d[:, 1], c='g', marker='o', s=40, label='ROI Points', visible=status[1])
        # --- Straightness check and spiral fitting ---
        if show_spiral and status[0] and len(spine_2d) >= 5:
            if is_straight(spine_2d):
                rms_percent = None
            else:
                x = spine_2d[:, 0]
                y = spine_2d[:, 1]
                x0, y0, a, b, theta_off = fit_log_spiral_explicit(x, y)
                if not (np.isnan(a) or np.isinf(a) or np.isnan(b) or np.isinf(b)):
                    theta_data = np.unwrap(np.arctan2(y - y0, x - x0))
                    theta_min, theta_max = np.min(theta_data), np.max(theta_data)
                    theta_range = np.linspace(theta_min, theta_max, 500)
                    r_fit = a * safe_exp(b * (theta_range + theta_off))
                    valid = np.isfinite(r_fit) & (r_fit > 0)
                    x_fit = x0 + r_fit[valid] * np.cos(theta_range[valid])
                    y_fit = y0 + r_fit[valid] * np.sin(theta_range[valid])
                    spiral_line, = ax.plot(x_fit, y_fit, 'r-', lw=2, label='Log Spiral Fit')
                    rms_percent = calculate_rms_percent_error(x, y, x0, y0, a, b, theta_off)
        ax.set_xlabel('Plane X', fontsize=14)
        ax.set_ylabel('Plane Y', fontsize=14)
        title = '2D Projection onto Reference Plane'
        if rms_percent is not None:
            title += f'   |   RMS Error: {rms_percent:.1f}%'
        ax.set_title(title, fontsize=15)
        ax.set_xlim(x2d_lim)
        ax.set_ylim(y2d_lim)
        ax.axis('equal')
        ax.grid(True)
    legend_handles = []
    legend_labels = []
    if status[1]:
        legend_handles.append(tdcr_scatter)
        legend_labels.append('ROI Points')
    if status[0]:
        legend_handles.append(spine_scatter)
        legend_labels.append('Spine Points')
    if mode_2d and show_spiral and status[0] and spiral_line is not None:
        legend_handles.append(spiral_line)
        legend_labels.append('Log Spiral Fit')
    legend_ax.legend(legend_handles, legend_labels, loc='upper left', fontsize=13, frameon=True)
    plt.draw()

def update(val):
    mode_2d = check.get_status()[2]
    show_spiral = check.get_status()[3] if len(check.labels) > 3 else False
    plot_row(int(slider.val) - 1, mode_2d, show_spiral)

def toggle_visibility(label):
    update(None)

plot_row(0, False, True)
slider.on_changed(update)
check.on_clicked(toggle_visibility)
plt.show()
