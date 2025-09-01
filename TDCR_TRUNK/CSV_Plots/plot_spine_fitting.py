import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons
from scipy.optimize import minimize, Bounds, differential_evolution

# ------- CONFIG: Edit these to control frame skipping -------
IGNORE_FIRST_N_FRAMES = 500          # Number of initial frames to ignore (edit this)
IGNORE_FIRST_X_DISPLACEMENT = 6.0   # Minimum displacement in mm to consider starting point in error plot mode (edit this; shifts the error plot to ignore below this)

# ------- VISUAL: Use Computer Modern (LaTeX) fonts -------
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

# ------- VISUAL: Consistent global limits with small margin (considering only non-ignored frames) -------
all_x = np.concatenate([df_spine[x_cols_spine].values.flatten()[IGNORE_FIRST_N_FRAMES:], df_tdcr[x_cols_tdcr].values.flatten()[IGNORE_FIRST_N_FRAMES:]])
all_y = np.concatenate([df_spine[y_cols_spine].values.flatten()[IGNORE_FIRST_N_FRAMES:], df_tdcr[y_cols_tdcr].values.flatten()[IGNORE_FIRST_N_FRAMES:]])
all_z = np.concatenate([df_spine[z_cols_spine].values.flatten()[IGNORE_FIRST_N_FRAMES:], df_tdcr[z_cols_tdcr].values.flatten()[IGNORE_FIRST_N_FRAMES:]])
x_min, x_max = np.nanmin(all_x), np.nanmax(all_x)
y_min, y_max = np.nanmin(all_y), np.nanmax(all_y)
z_min, z_max = np.nanmin(all_z), np.nanmax(all_z)
max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
x_center = (x_max + x_min) / 2
y_center = (y_max + y_min) / 2
z_center = (z_max + z_min) / 2
margin = 0
x_lim = [x_center - max_range/2 - margin*max_range, x_center + max_range/2 + margin*max_range]
y_lim = [y_center - max_range/2 - margin*max_range, y_center + max_range/2 + margin*max_range]
z_lim = [z_center - max_range/2 - margin*max_range, z_center + max_range/2 + margin*max_range]

def get_axis_limits():
    return x_lim, y_lim, z_lim

def fit_plane(points):
    points = points[~np.isnan(points).any(axis=1)]
    if len(points) < 3:
        return np.mean(points, axis=0), np.array([0, 0, 1]), np.array([1, 0, 0]), np.array([0, 1, 0])
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    if np.allclose(centered.std(axis=0), 0):
        return centroid, np.array([0, 0, 1]), np.array([1, 0, 0]), np.array([0, 1, 0])
    centered += np.random.normal(0, 1e-10, centered.shape)
    _, _, vh = np.linalg.svd(centered)
    normal = vh[2, :]
    e1 = vh[0, :]
    e2 = vh[1, :]
    return centroid, normal, e1, e2

def project_to_plane(points, origin, e1, e2):
    rel = points - origin
    x = np.dot(rel, e1)
    y = np.dot(rel, e2)
    return np.stack([x, y], axis=1)

xs0 = df_spine.loc[IGNORE_FIRST_N_FRAMES, x_cols_spine].values
ys0 = df_spine.loc[IGNORE_FIRST_N_FRAMES, y_cols_spine].values
zs0 = df_spine.loc[IGNORE_FIRST_N_FRAMES, z_cols_spine].values
points0 = np.vstack([xs0, ys0, zs0]).T
ref_centroid, ref_normal, ref_e1, ref_e2 = fit_plane(points0)

x2d_lim = (-150, 150)
y2d_lim = (-150, 150)

def safe_exp(x):
    x = np.clip(x, -20, 20)
    return np.exp(x)

spiral_fit_cache = {}

def fit_log_spiral_explicit(x, y, maxiter=20000, use_global_opt=False, iter_refine=2):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 5:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    centroid = np.array([np.median(x), np.median(y)])
    dists = np.sqrt((x - centroid[0])**2 + (y - centroid[1])**2)
    keep_mask = dists < np.percentile(dists, 90)
    x = x[keep_mask]
    y = y[keep_mask]
    if len(x) < 5:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    x0_init = np.median(x)
    y0_init = np.median(y)
    theta = np.unwrap(np.arctan2(y - y0_init, x - x0_init))
    r = np.sqrt((x - x0_init)**2 + (y - y0_init)**2)
    weights = r / np.max(r)
    p = np.polyfit(theta, np.log(np.maximum(r,1e-8)), 1, w=weights)
    a0 = np.exp(p[1])
    b0 = p[0]
    theta0 = 0
    params0 = np.array([x0_init, y0_init, np.log(a0), b0, theta0], dtype=float)
    min_x, max_x = np.min(x), np.max(x)
    min_y, max_y = np.min(y), np.max(y)
    bounds = Bounds([min_x, min_y, -10, -2, -2*np.pi], [max_x, max_y, 10, 2, 2*np.pi])
    def spiral_cost(params):
        x0, y0, loga, b, theta_off = params
        theta_data = np.unwrap(np.arctan2(y - y0, x - x0))
        r_data = np.sqrt((x - x0)**2 + (y - y0)**2)
        if np.all(r_data == 0):
            return 1e10
        model_log_r = loga + b*(theta_data + theta_off)
        log_r_data = np.log(np.maximum(r_data,1e-8))
        reg = 0.01*(b**2 + theta_off**2)
        return np.sum((log_r_data - model_log_r)**2) + reg
    if use_global_opt:
        bounds_list = [(min_x,max_x),(min_y,max_y),(-10,10),(-2,2),(-2*np.pi,2*np.pi)]
        res = differential_evolution(spiral_cost, bounds_list, maxiter=maxiter, tol=1e-10)
    else:
        res = minimize(spiral_cost, params0, method='L-BFGS-B', bounds=bounds, options={'maxiter':maxiter,'ftol':1e-10})
    x0_fit, y0_fit, loga_fit, b_fit, theta_off_fit = res.x
    a_fit = np.exp(np.clip(loga_fit,-20,20))
    for _ in range(iter_refine):
        theta_data = np.unwrap(np.arctan2(y - y0_fit, x - x0_fit))
        r_data = np.sqrt((x - x0_fit)**2 + (y - y0_fit)**2)
        r_model = a_fit * safe_exp(b_fit * (theta_data + theta_off_fit))
        residuals = np.abs(r_data - r_model)
        keep_mask = residuals < np.percentile(residuals, 80)
        if np.sum(keep_mask) < 5:
            break
        x_refine = x[keep_mask]
        y_refine = y[keep_mask]
        x0_init = np.median(x_refine)
        y0_init = np.median(y_refine)
        theta = np.unwrap(np.arctan2(y_refine - y0_init, x_refine - x0_init))
        r = np.sqrt((x_refine - x0_init)**2 + (y_refine - y0_init)**2)
        p = np.polyfit(theta, np.log(np.maximum(r,1e-8)), 1, w=(r/np.max(r)))
        a0 = np.exp(p[1])
        b0 = p[0]
        theta0 = 0
        params0 = np.array([x0_init, y0_init, np.log(a0), b0, theta0], dtype=float)
        if use_global_opt:
            res = differential_evolution(spiral_cost, bounds_list, maxiter=maxiter, tol=1e-10)
        else:
            res = minimize(spiral_cost, params0, method='L-BFGS-B', bounds=bounds, options={'maxiter':maxiter,'ftol':1e-10})
        x0_fit, y0_fit, loga_fit, b_fit, theta_off_fit = res.x
        a_fit = np.exp(np.clip(loga_fit,-20,20))
    return x0_fit, y0_fit, a_fit, b_fit, theta_off_fit

def fit_log_spiral_explicit_cached(frame_idx):
    actual_idx = frame_idx + IGNORE_FIRST_N_FRAMES
    if actual_idx in spiral_fit_cache:
        return spiral_fit_cache[actual_idx]
    xs_spine = df_spine.loc[actual_idx, x_cols_spine].values
    ys_spine = df_spine.loc[actual_idx, y_cols_spine].values
    zs_spine = df_spine.loc[actual_idx, z_cols_spine].values
    points_spine = np.vstack([xs_spine, ys_spine, zs_spine]).T
    spine_2d = project_to_plane(points_spine, ref_centroid, ref_e1, ref_e2)
    x = spine_2d[:, 0]
    y = spine_2d[:, 1]
    fit = fit_log_spiral_explicit(x, y)
    spiral_fit_cache[actual_idx] = fit
    return fit

def calculate_displacement(points):
    ref_points = np.vstack([
        df_spine.loc[IGNORE_FIRST_N_FRAMES, x_cols_spine].values,
        df_spine.loc[IGNORE_FIRST_N_FRAMES, y_cols_spine].values,
        df_spine.loc[IGNORE_FIRST_N_FRAMES, z_cols_spine].values]).T
    curr_points = points
    if len(ref_points) != len(curr_points):
        return 0
    diffs = curr_points - ref_points
    distances = np.linalg.norm(diffs, axis=1)
    return np.max(distances)

fig = plt.figure(figsize=(10, 8))
main_ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(left=0.05, right=0.85, bottom=0.2, top=0.85)
error_ax = fig.add_axes([0.1, 0.1, 0.75, 0.75])
error_ax.set_visible(False)

slider_ax = fig.add_axes([0.2, 0.05, 0.5, 0.03])
slider = Slider(slider_ax, 'Frame Slider', 1, len(df_spine) - IGNORE_FIRST_N_FRAMES, valinit=1, valstep=1)
slider.label.set_fontsize(16)

check_ax = plt.axes([0.85, 0.3, 0.13, 0.4])
check_labels = ['Spine', 'ROI_Points', '2D Mode', 'Show Spiral', 'Error Plot Mode', 'Show All Frames']
check_vals = [True, True, False, True, False, False]
check = CheckButtons(check_ax, check_labels, check_vals)
for text in check.labels:
    text.set_fontsize(12)

legend_ax = plt.axes([0.85, 0.75, 0.13, 0.15])
legend_ax.axis('off')

spine_scatter = None
tdcr_scatter = None
spiral_line = None

def plot_row(row_idx, mode_2d, show_spiral, error_plot_mode, show_all_errors):
    global spine_scatter, tdcr_scatter, spiral_line, main_ax, error_ax, legend_ax
    legend_ax.clear()
    legend_ax.axis('off')
    main_ax.clear()
    error_ax.clear()
    error_ax.set_visible(error_plot_mode)
    main_ax.set_visible(not error_plot_mode)

    actual_idx = row_idx + IGNORE_FIRST_N_FRAMES

    xs_spine = df_spine.loc[actual_idx, x_cols_spine].values
    ys_spine = df_spine.loc[actual_idx, y_cols_spine].values
    zs_spine = df_spine.loc[actual_idx, z_cols_spine].values
    points_spine = np.vstack([xs_spine, ys_spine, zs_spine]).T

    xs_tdcr = df_tdcr.loc[actual_idx, x_cols_tdcr].values
    ys_tdcr = df_tdcr.loc[actual_idx, y_cols_tdcr].values
    zs_tdcr = df_tdcr.loc[actual_idx, z_cols_tdcr].values
    points_tdcr = np.vstack([xs_tdcr, ys_tdcr, zs_tdcr]).T
    status = check.get_status()

    if error_plot_mode:
        if show_all_errors:
            fixed_color = 'tab:blue'
            n_frames = len(df_spine) - IGNORE_FIRST_N_FRAMES
            for i in range(n_frames):
                actual_i = i + IGNORE_FIRST_N_FRAMES
                x0_i, y0_i, a_i, b_i, theta_off_i = fit_log_spiral_explicit_cached(i)
                xs_spine_i = df_spine.loc[actual_i, x_cols_spine].values
                ys_spine_i = df_spine.loc[actual_i, y_cols_spine].values
                zs_spine_i = df_spine.loc[actual_i, z_cols_spine].values
                points_spine_i = np.vstack([xs_spine_i, ys_spine_i, zs_spine_i]).T
                spine_2d_i = project_to_plane(points_spine_i, ref_centroid, ref_e1, ref_e2)
                x_i = spine_2d_i[:, 0]
                y_i = spine_2d_i[:, 1]
                theta_data_i = np.unwrap(np.arctan2(y_i - y0_i, x_i - x0_i))
                r_data_i = np.sqrt((x_i - x0_i)**2 + (y_i - y0_i)**2)
                r_model_i = a_i * safe_exp(b_i * (theta_data_i + theta_off_i))
                distances_i = np.abs(r_data_i - r_model_i)
                point_indices_i = np.arange(len(distances_i))
                # Shift for all frames plot
                valid_indices_i = np.where(r_data_i >= IGNORE_FIRST_X_DISPLACEMENT)[0]
                offset_idx_i = valid_indices_i[0] if len(valid_indices_i) > 0 else 0
                shifted_indices_i = point_indices_i[offset_idx_i:] - offset_idx_i
                shifted_distances_i = distances_i[offset_idx_i:]
                error_ax.plot(shifted_indices_i, shifted_distances_i, color=fixed_color)
            error_ax.set_xlabel('Point Index (shifted)', fontsize=14)
            error_ax.set_ylabel('Distance', fontsize=14)
            error_ax.set_title('Error Plot - All Frames (shifted)', fontsize=15)
            error_ax.grid(True)
        else:
            x0, y0, a, b, theta_off = fit_log_spiral_explicit_cached(row_idx)
            spine_2d = project_to_plane(points_spine, ref_centroid, ref_e1, ref_e2)
            x = spine_2d[:, 0]
            y = spine_2d[:, 1]
            theta_data = np.unwrap(np.arctan2(y - y0, x - x0))
            r_data = np.sqrt((x - x0)**2 + (y - y0)**2)
            r_model = a * safe_exp(b * (theta_data + theta_off))
            distances = np.abs(r_data - r_model)
            point_indices = np.arange(len(distances))
            # Shift for single frame error plot
            valid_indices = np.where(r_data >= IGNORE_FIRST_X_DISPLACEMENT)[0]
            offset_idx = valid_indices[0] if len(valid_indices) > 0 else 0
            shifted_indices = point_indices[offset_idx:] - offset_idx
            shifted_distances = distances[offset_idx:]
            error_ax.plot(shifted_indices, shifted_distances, 'b-o', label='Distance from Fitted Curve')
            error_ax.set_xlabel('Point Index (shifted)', fontsize=14)
            error_ax.set_ylabel('Distance', fontsize=14)
            error_ax.set_title(f'Error Plot - Frame {actual_idx + 1} (shifted)', fontsize=15)
            error_ax.grid(True)
            error_ax.legend(fontsize=12)
    else:
        if mode_2d:
            if main_ax.name == '3d':
                fig.delaxes(main_ax)
                main_ax = fig.add_subplot(111)
            spine_2d = project_to_plane(points_spine, ref_centroid, ref_e1, ref_e2)
            tdcr_2d = project_to_plane(points_tdcr, ref_centroid, ref_e1, ref_e2)
            if status[0]:
                spine_scatter = main_ax.scatter(spine_2d[:, 0], spine_2d[:, 1], c='b', marker='o', s=40, label='Spine Points')
            if status[1]:
                tdcr_scatter = main_ax.scatter(tdcr_2d[:, 0], tdcr_2d[:, 1], c='g', marker='o', s=40, label='ROI Points')
            if show_spiral:
                x = spine_2d[:, 0]
                y = spine_2d[:, 1]
                x0, y0, a, b, theta_off = fit_log_spiral_explicit_cached(row_idx)
                theta_data = np.unwrap(np.arctan2(y - y0, x - x0))
                theta_min, theta_max = np.min(theta_data), np.max(theta_data)
                theta_fit = np.linspace(theta_min, theta_max, 200)
                r_fit = a * safe_exp(b * (theta_fit + theta_off))
                x_fit = x0 + r_fit * np.cos(theta_fit)
                y_fit = y0 + r_fit * np.sin(theta_fit)
                spiral_line = main_ax.plot(x_fit, y_fit, 'r-', lw=2, label='Fitted Spiral')
            main_ax.set_xlabel('Plane X', fontsize=14)
            main_ax.set_ylabel('Plane Y', fontsize=14)
            title = '2D Projection onto Reference Plane'
            main_ax.set_title(title, fontsize=15)
            main_ax.set_xlim(x2d_lim)
            main_ax.set_ylim(y2d_lim)
            main_ax.axis('equal')
            main_ax.grid(True)
        else:
            if main_ax.name != '3d':
                fig.delaxes(main_ax)
                main_ax = fig.add_subplot(111, projection='3d')
            if status[0]:
                spine_scatter = main_ax.scatter(xs_spine, ys_spine, zs_spine, c='b', marker='o', s=40, label='Spine Points')
            if status[1]:
                tdcr_scatter = main_ax.scatter(xs_tdcr, ys_tdcr, zs_tdcr, c='g', marker='o', s=40, label='ROI Points')
            f = 12
            lp = 20
            main_ax.set_xlabel('X', fontsize=f, labelpad=lp)
            main_ax.set_ylabel('Y', fontsize=f, labelpad=lp)
            main_ax.set_zlabel('Z', fontsize=f, labelpad=lp)
            main_ax.tick_params(axis='both', which='major', labelsize=20)
            main_ax.tick_params(axis='both', which='minor', labelsize=14)
            main_ax.tick_params(axis='x', pad=10)
            main_ax.tick_params(axis='y', pad=10)
            main_ax.tick_params(axis='z', pad=10)
            main_ax.set_xlim(x_lim)
            main_ax.set_ylim(y_lim)
            main_ax.set_zlim(z_lim)
            try:
                main_ax.set_box_aspect([1, 1, 1])
            except Exception:
                pass

        legend_handles = []
        legend_labels = []
        if status[1] and tdcr_scatter is not None:
            legend_handles.append(tdcr_scatter)
            legend_labels.append('ROI Points')
        if status[0] and spine_scatter is not None:
            legend_handles.append(spine_scatter)
            legend_labels.append('Spine Points')
        if legend_handles:
            legend_ax.legend(legend_handles, legend_labels, loc='upper left', fontsize=13, frameon=True)

    plt.draw()

def update(val):
    status = check.get_status()
    mode_2d = status[2]
    show_spiral = status[3]
    error_plot_mode = status[4]
    show_all_errors = status[5]
    slider_ax.set_visible(not (error_plot_mode and show_all_errors))
    if error_plot_mode and show_all_errors:
        plot_row(0, mode_2d, show_spiral, error_plot_mode, show_all_errors)
    else:
        plot_row(int(slider.val) - 1, mode_2d, show_spiral, error_plot_mode, show_all_errors)

def toggle_visibility(label):
    update(None)

plot_row(0, False, True, False, False)
slider.on_changed(update)
check.on_clicked(toggle_visibility)
plt.show()
