import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons
from scipy.optimize import minimize, Bounds, differential_evolution
import matplotlib.lines as mlines  # Added for custom legend handle
from matplotlib import cm  # Added for colormaps







# ------- CONFIG: Edit these to control frame skipping -------
IGNORE_FIRST_N_FRAMES = 0        
IGNORE_FIRST_X_DISPLACEMENT = 4.0   
EVERY_N_FRAMES = 1 #min 1
REFERENCE_FRAME = 0  # New: Fixed frame for plane fitting (e.g., 0 for initial state)
IGNORE_FRAMES = [0,1,2,3,7]  # Add specific frame numbers (actual indices) to ignore, e.g., [10, 25, 50]
# ----------------------------------------------------------







# ------- VISUAL: Use Computer Modern (LaTeX) fonts -------
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['cmr10']
plt.rcParams["axes.formatter.use_mathtext"] = True
# Global tick label size adjustments
plt.rcParams['xtick.labelsize'] = 28
plt.rcParams['ytick.labelsize'] = 28
# --- Load data ---
df_spine = pd.read_csv('tdcr_trunk_spine.csv')
df_tdcr = pd.read_csv('tdcr_trunk_output.csv')
x_cols_spine = [col for col in df_spine.columns if col.startswith('x')]
y_cols_spine = [col for col in df_spine.columns if col.startswith('y')]
z_cols_spine = [col for col in df_spine.columns if col.startswith('z')]
x_cols_tdcr = [col for col in df_tdcr.columns if col.startswith('x')]
y_cols_tdcr = [col for col in df_tdcr.columns if col.startswith('y')]
z_cols_tdcr = [col for col in df_tdcr.columns if col.startswith('z')]
# ------- VISUAL: Consistent global limits with small margin (using ALL frames) -------
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
margin = 0.1  # Increased margin for better visibility
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
    
    print(f"Reference plane: centroid={centroid}, normal={normal}")  # Debugging print
    return centroid, normal, e1, e2
def project_to_plane(points, origin, e1, e2):
    rel = points - origin
    x = np.dot(rel, e1)
    y = np.dot(rel, e2)
    return np.stack([x, y], axis=1)
# Compute averaged reference over first few frames (e.g., 0-2)
avg_points = []
for idx in range(3):  # Average first 3 frames; adjust as needed
    if idx < len(df_spine):
        xs = df_spine.loc[idx, x_cols_spine].values
        ys = df_spine.loc[idx, y_cols_spine].values
        zs = df_spine.loc[idx, z_cols_spine].values
        avg_points.extend(np.vstack([xs, ys, zs]).T)
avg_points = np.array(avg_points)
ref_centroid, ref_normal, ref_e1, ref_e2 = fit_plane(avg_points)
x2d_lim = (-150, 150)
y2d_lim = (-150, 150)
transform_x_lim_single = (-5, 105)
transform_x_lim_all = (-10, 110)
def safe_exp(x):
    x = np.clip(x, -20, 20)
    return np.exp(x)
def calculate_point_displacements(points_current):
    """Calculate displacement of each point from reference frame"""
    ref_points = np.vstack([
        df_spine.loc[REFERENCE_FRAME, x_cols_spine].values,
        df_spine.loc[REFERENCE_FRAME, y_cols_spine].values,
        df_spine.loc[REFERENCE_FRAME, z_cols_spine].values]).T
    
    if len(ref_points) != len(points_current):
        return np.zeros(len(points_current))
    
    diffs = points_current - ref_points
    distances = np.linalg.norm(diffs, axis=1)
    return distances
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
        df_spine.loc[REFERENCE_FRAME, x_cols_spine].values,
        df_spine.loc[REFERENCE_FRAME, y_cols_spine].values,
        df_spine.loc[REFERENCE_FRAME, z_cols_spine].values]).T
    
    curr_points = points
    
    if len(ref_points) != len(curr_points):
        return 0
    
    diffs = points - ref_points
    distances = np.linalg.norm(diffs, axis=1)
    return np.max(distances)
# NEW: Function to determine if a frame should be considered
def should_consider_frame(frame_idx):
    actual_frame = frame_idx + IGNORE_FIRST_N_FRAMES
    if actual_frame in IGNORE_FRAMES:
        return False
    return (frame_idx % EVERY_N_FRAMES) == 0
# NEW: Precompute all spiral fits for every Nth frame to make GUI faster (lookup only)
print("Precomputing spiral fits for every Nth frame... This may take a moment.")
n_frames = len(df_spine) - IGNORE_FIRST_N_FRAMES
considered_frames = [i for i in range(n_frames) if should_consider_frame(i)]
total_considered = len(considered_frames)
for count, i in enumerate(considered_frames, 1):
    _ = fit_log_spiral_explicit_cached(i)  # Populates the cache
    if count % 100 == 0:  # Progress indicator
        print(f"Precomputed {count}/{total_considered} frames.")
print("Precomputation complete. Starting GUI...")
# Create figure with adjusted positioning
fig = plt.figure(figsize=(10, 8))
main_ax = fig.add_subplot(111, projection='3d')
# Fixed UI positioning to prevent overlapping, increased distance
plt.subplots_adjust(left=0.15, right=0.80, bottom=0.25, top=0.85)  # Increased left margin to shift graph right
deviation_ax = fig.add_axes([0.15, 0.15, 0.65, 0.65])  # Adjusted to shift right
deviation_ax.set_visible(False)
slider_ax = fig.add_axes([0.2, 0.08, 0.5, 0.03])  # Moved slider up
slider = Slider(slider_ax, 'Frame Slider', 1, len(considered_frames), valinit=1, valstep=1)
slider.label.set_fontsize(16)




# Group 1: Display Elements
display_ax = plt.axes([0.85, 0.65, 0.13, 0.15])  # Shifted right
display_labels = ['Spine', 'ROI_Points', 'Show Spiral']
display_vals = [True, True, True]
display_check = CheckButtons(display_ax, display_labels, display_vals)
for text in display_check.labels:
    text.set_fontsize(12)
fig.text(0.85, 0.80, 'Display', fontsize=14, fontweight='bold')




# Group 2: View Modes
modes_ax = plt.axes([0.85, 0.50, 0.13, 0.10])  # Shifted right
modes_labels = ['2D Mode', 'Deviation Plot Mode']
modes_vals = [False, False]
modes_check = CheckButtons(modes_ax, modes_labels, modes_vals)
for text in modes_check.labels:
    text.set_fontsize(12)
fig.text(0.85, 0.60, 'Modes', fontsize=14, fontweight='bold')




# Group 3: Multi-Frame Options
multi_ax = plt.axes([0.85, 0.35, 0.13, 0.15])  # Shifted right
multi_labels = ['Show All Frames', 'Show All in 2D', 'Show Average']
multi_vals = [False, False, False]
multi_check = CheckButtons(multi_ax, multi_labels, multi_vals)
for text in multi_check.labels:
    text.set_fontsize(12)
fig.text(0.85, 0.50, 'Multi-Frame', fontsize=14, fontweight='bold')




# Group 4: 2D Options
options_ax = plt.axes([0.85, 0.25, 0.13, 0.05])  # Shifted right
options_labels = ['Transform 2D']
options_vals = [False]
options_check = CheckButtons(options_ax, options_labels, options_vals)
for text in options_check.labels:
    text.set_fontsize(12)
fig.text(0.85, 0.30, '2D Options', fontsize=14, fontweight='bold')




legend_ax = plt.axes([0.85, 0.85, 0.13, 0.15])  # Shifted right
legend_ax.axis('off')
spine_scatter = None
tdcr_scatter = None
spiral_line = None
def plot_row(considered_idx, mode_2d, show_spiral, deviation_plot_mode, show_all_deviations, show_average, show_all_2d, show_transform):
    global spine_scatter, tdcr_scatter, spiral_line, main_ax, deviation_ax, legend_ax
    
    legend_ax.clear()
    legend_ax.axis('off')
    main_ax.clear()
    deviation_ax.clear()
    deviation_ax.set_visible(deviation_plot_mode)
    main_ax.set_visible(not deviation_plot_mode)
    
    # Map considered_idx to actual frame_idx
    if considered_idx < len(considered_frames):
        row_idx = considered_frames[considered_idx]
    else:
        row_idx = 0  # Fallback
    
    actual_idx = row_idx + IGNORE_FIRST_N_FRAMES
    xs_spine = df_spine.loc[actual_idx, x_cols_spine].values
    ys_spine = df_spine.loc[actual_idx, y_cols_spine].values
    zs_spine = df_spine.loc[actual_idx, z_cols_spine].values
    points_spine = np.vstack([xs_spine, ys_spine, zs_spine]).T
    # Apply displacement filtering - set coordinates to reference if displacement < threshold
    spine_displacements = calculate_point_displacements(points_spine)
    if np.all(spine_displacements == 0):  # Skip filtering if no changes
        low_displacement_mask = np.zeros_like(spine_displacements, dtype=bool)
    else:
        low_displacement_mask = spine_displacements < IGNORE_FIRST_X_DISPLACEMENT
    xs_spine[low_displacement_mask] = df_spine.loc[REFERENCE_FRAME, x_cols_spine].values[low_displacement_mask]
    ys_spine[low_displacement_mask] = df_spine.loc[REFERENCE_FRAME, y_cols_spine].values[low_displacement_mask]
    zs_spine[low_displacement_mask] = df_spine.loc[REFERENCE_FRAME, z_cols_spine].values[low_displacement_mask]
    points_spine = np.vstack([xs_spine, ys_spine, zs_spine]).T
    xs_tdcr = df_tdcr.loc[actual_idx, x_cols_tdcr].values
    ys_tdcr = df_tdcr.loc[actual_idx, y_cols_tdcr].values
    zs_tdcr = df_tdcr.loc[actual_idx, z_cols_tdcr].values
    points_tdcr = np.vstack([xs_tdcr, ys_tdcr, zs_tdcr]).T
    
    if deviation_plot_mode:
        legend_handles = []
        
        # Calculate average data if needed (but plot later)
        avg_deviations = None
        point_indices_avg = None
        if show_average:
            all_deviations = []
            max_points = 0
            
            for i in considered_frames:
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
                
                filter_idx = int(IGNORE_FIRST_X_DISPLACEMENT)
                filtered_distances_i = distances_i[filter_idx:]
                all_deviations.append(filtered_distances_i)
                max_points = max(max_points, len(filtered_distances_i))
            
            padded_deviations = []
            for dev in all_deviations:
                padded = np.full(max_points, np.nan)
                padded[:len(dev)] = dev
                padded_deviations.append(padded)
            
            avg_deviations = np.nanmean(padded_deviations, axis=0)
            point_indices_avg = np.arange(int(IGNORE_FIRST_X_DISPLACEMENT), int(IGNORE_FIRST_X_DISPLACEMENT) + len(avg_deviations))
        
        # Plot all individual frames with unique colors
        if show_all_deviations:
            num_frames = len(considered_frames)
            colors = cm.viridis(np.linspace(0, 1, num_frames))
            alpha_val = 0.3 if show_average else 1.0  # Translucent when average is shown
            
            # Get L1 values for min, mid, max disp
            l1_values = np.array([df_spine.loc[i + IGNORE_FIRST_N_FRAMES, 'L1'] for i in considered_frames])
            min_disp_idx = np.argmin(l1_values)
            mid_disp_idx = len(l1_values) // 2
            max_disp_idx = np.argmax(l1_values)
            min_disp = l1_values[min_disp_idx]
            mid_disp = l1_values[mid_disp_idx]
            max_disp = l1_values[max_disp_idx]
            
            # Create legend handles for min, mid, max
            min_handle = mlines.Line2D([], [], color=colors[min_disp_idx], linewidth=2, label=f'Min Disp: {min_disp:.2f} mm')
            mid_handle = mlines.Line2D([], [], color=colors[mid_disp_idx], linewidth=2, label=f'Mid Disp: {mid_disp:.2f} mm')
            max_handle = mlines.Line2D([], [], color=colors[max_disp_idx], linewidth=2, label=f'Max Disp: {max_disp:.2f} mm')
            legend_handles.extend([min_handle, mid_handle, max_handle])
            
            for j, i in enumerate(considered_frames):
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
                filter_idx = int(IGNORE_FIRST_X_DISPLACEMENT)
                filtered_indices_i = point_indices_i[filter_idx:]
                filtered_distances_i = distances_i[filter_idx:]
                deviation_ax.plot(filtered_indices_i, filtered_distances_i, color=colors[j], alpha=alpha_val, linewidth=2)
        
        # Plot single frame if neither is selected
        if not show_all_deviations and not show_average:
            x0, y0, a, b, theta_off = fit_log_spiral_explicit_cached(row_idx)
            spine_2d = project_to_plane(points_spine, ref_centroid, ref_e1, ref_e2)
            x = spine_2d[:, 0]
            y = spine_2d[:, 1]
            theta_data = np.unwrap(np.arctan2(y - y0, x - x0))
            r_data = np.sqrt((x - x0)**2 + (y - y0)**2)
            r_model = a * safe_exp(b * (theta_data + theta_off))
            distances = np.abs(r_data - r_model)
            point_indices = np.arange(len(distances))
            filter_idx = int(IGNORE_FIRST_X_DISPLACEMENT)
            filtered_indices = point_indices[filter_idx:]
            filtered_distances = distances[filter_idx:]
            # Get L1 for single frame
            l1_value = df_spine.loc[actual_idx, 'L1']
            single_line, = deviation_ax.plot(filtered_indices, filtered_distances, 'b-o', linewidth=2, label=f"Tendon Displacement: {l1_value:.2f} mm")
            legend_handles.append(single_line)
        
        # Now plot the average line on top if enabled
        if show_average:
            avg_color = 'r' if show_all_deviations else 'tab:blue'
            avg_line, = deviation_ax.plot(point_indices_avg, avg_deviations, avg_color + '-', linewidth=5, label='Average Deviation')
            legend_handles.append(avg_line)
        
        deviation_ax.set_xlabel('Length (cm)', fontsize=24)
        deviation_ax.set_ylabel('Deviation (mm)', fontsize=24)
        deviation_ax.tick_params(axis='both', which='major', labelsize=24)
        deviation_ax.tick_params(axis='both', which='minor', labelsize=18)
        deviation_ax.grid(True)
        deviation_ax.set_ylim(0, 30)  # Updated y-axis limit
        
        if show_average and show_all_deviations:
            deviation_ax.set_title('Deviation Plot', fontsize=20)
        elif show_average:
            deviation_ax.set_title('Average Deviation Plot - Considered Frames', fontsize=20)
        elif show_all_deviations:
            deviation_ax.set_title('Deviation Plot - All Considered Frames', fontsize=20)
        else:
            deviation_ax.set_title(f'Deviation Plot - Frame {actual_idx + 1} ', fontsize=20)
        
        if legend_handles:
            deviation_ax.legend(handles=legend_handles, fontsize=12, loc='upper left')  # Increased font size for bigger legend
    else:
        if mode_2d:
            if main_ax.name == '3d':
                fig.delaxes(main_ax)
                main_ax = fig.add_subplot(111)
            
            legend_handles = []
            legend_labels = []
            
            if show_all_2d:
                num_frames = len(considered_frames)
                colors = cm.viridis(np.linspace(0, 1, num_frames))
                alpha_val = 0.3  # Slightly translucent for overlay
                
                # Collect L1 values for legend
                l1_values = np.array([df_spine.loc[i + IGNORE_FIRST_N_FRAMES, 'L1'] for i in considered_frames])
                min_l1_idx = np.argmin(l1_values)
                mid_l1_idx = len(l1_values) // 2
                max_l1_idx = np.argmax(l1_values)
                min_l1 = l1_values[min_l1_idx]
                mid_l1 = l1_values[mid_l1_idx]
                max_l1 = l1_values[max_l1_idx]
                
                for j, i in enumerate(considered_frames):
                    actual_i = i + IGNORE_FIRST_N_FRAMES
                    xs_spine_i = df_spine.loc[actual_i, x_cols_spine].values
                    ys_spine_i = df_spine.loc[actual_i, y_cols_spine].values
                    zs_spine_i = df_spine.loc[actual_i, z_cols_spine].values
                    points_spine_i = np.vstack([xs_spine_i, ys_spine_i, zs_spine_i]).T
                    
                    # Apply displacement filtering for each frame
                    spine_displacements_i = calculate_point_displacements(points_spine_i)
                    if not np.all(spine_displacements_i == 0):
                        low_displacement_mask_i = spine_displacements_i < IGNORE_FIRST_X_DISPLACEMENT
                        xs_spine_i[low_displacement_mask_i] = df_spine.loc[REFERENCE_FRAME, x_cols_spine].values[low_displacement_mask_i]
                        ys_spine_i[low_displacement_mask_i] = df_spine.loc[REFERENCE_FRAME, y_cols_spine].values[low_displacement_mask_i]
                        zs_spine_i[low_displacement_mask_i] = df_spine.loc[REFERENCE_FRAME, z_cols_spine].values[low_displacement_mask_i]
                        points_spine_i = np.vstack([xs_spine_i, ys_spine_i, zs_spine_i]).T
                    
                    spine_2d_i = project_to_plane(points_spine_i, ref_centroid, ref_e1, ref_e2)
                    
                    xs_tdcr_i = df_tdcr.loc[actual_i, x_cols_tdcr].values
                    ys_tdcr_i = df_tdcr.loc[actual_i, y_cols_tdcr].values
                    zs_tdcr_i = df_tdcr.loc[actual_i, z_cols_tdcr].values
                    points_tdcr_i = np.vstack([xs_tdcr_i, ys_tdcr_i, zs_tdcr_i]).T
                    tdcr_2d_i = project_to_plane(points_tdcr_i, ref_centroid, ref_e1, ref_e2)
                    
                    color = colors[j]
                    
                    # Apply transformation if enabled
                    if show_transform:
                        spine_x = spine_2d_i[:, 1]
                        spine_y = spine_2d_i[:, 0]
                        tdcr_x = tdcr_2d_i[:, 1]
                        tdcr_y = tdcr_2d_i[:, 0]
                    else:
                        spine_x = spine_2d_i[:, 0]
                        spine_y = spine_2d_i[:, 1]
                        tdcr_x = tdcr_2d_i[:, 0]
                        tdcr_y = tdcr_2d_i[:, 1]
                    
                    if display_check.get_status()[0]:  # Spine
                        main_ax.scatter(spine_x, spine_y, c=[color], marker='o', s=40, alpha=alpha_val)
                    
                    if display_check.get_status()[1]:  # ROI_Points
                        main_ax.scatter(tdcr_x, tdcr_y, c=[color], marker='o', s=40, alpha=alpha_val)
                    
                    if show_spiral:
                        x_orig = spine_2d_i[:, 0]
                        y_orig = spine_2d_i[:, 1]
                        x0, y0, a, b, theta_off = fit_log_spiral_explicit_cached(i)
                        theta_data = np.unwrap(np.arctan2(y_orig - y0, x_orig - x0))
                        theta_min, theta_max = np.min(theta_data), np.max(theta_data)
                        theta_fit = np.linspace(theta_min, theta_max, 200)
                        r_fit = a * safe_exp(b * (theta_fit + theta_off))
                        x_fit = x0 + r_fit * np.cos(theta_fit)
                        y_fit = y0 + r_fit * np.sin(theta_fit)
                        if show_transform:
                            x_fit_trans = y_fit
                            y_fit_trans = x_fit
                            main_ax.plot(x_fit_trans, y_fit_trans, color=color, lw=2, alpha=alpha_val)
                        else:
                            main_ax.plot(x_fit, y_fit, color=color, lw=2, alpha=alpha_val)
                
                # Improved legend with representative L1 values
                low = mlines.Line2D([], [], color=colors[min_l1_idx], linewidth=2, label=f'Min L1: {min_l1:.2f} mm')
                mid = mlines.Line2D([], [], color=colors[mid_l1_idx], linewidth=2, label=f'Mid L1: {mid_l1:.2f} mm')
                high = mlines.Line2D([], [], color=colors[max_l1_idx], linewidth=2, label=f'Max L1: {max_l1:.2f} mm')
                main_ax.legend(handles=[low, mid, high], loc='upper left', fontsize=12, title='Tendon Displacement')  # Smaller legend
                
                title = '2D Projection and Curve Fitting'  # Updated title
            else:
                # Single frame plotting
                spine_2d = project_to_plane(points_spine, ref_centroid, ref_e1, ref_e2)
                tdcr_2d = project_to_plane(points_tdcr, ref_centroid, ref_e1, ref_e2)
                
                # Apply transformation if enabled
                if show_transform:
                    spine_x = spine_2d[:, 1]
                    spine_y = spine_2d[:, 0]
                    tdcr_x = tdcr_2d[:, 1]
                    tdcr_y = tdcr_2d[:, 0]
                else:
                    spine_x = spine_2d[:, 0]
                    spine_y = spine_2d[:, 1]
                    tdcr_x = tdcr_2d[:, 0]
                    tdcr_y = tdcr_2d[:, 1]
                
                if display_check.get_status()[0]:
                    spine_scatter = main_ax.scatter(spine_x, spine_y, c='b', marker='o', s=40, label='Spine Points')
                    legend_handles.append(spine_scatter)
                    legend_labels.append('Spine Points')
                if display_check.get_status()[1]:
                    tdcr_scatter = main_ax.scatter(tdcr_x, tdcr_y, c='g', marker='o', s=40, label='ROI Points')
                    legend_handles.append(tdcr_scatter)
                    legend_labels.append('ROI Points')
                
                if show_spiral:
                    x_orig = spine_2d[:, 0]
                    y_orig = spine_2d[:, 1]
                    x0, y0, a, b, theta_off = fit_log_spiral_explicit_cached(row_idx)
                    theta_data = np.unwrap(np.arctan2(y_orig - y0, x_orig - x0))
                    theta_min, theta_max = np.min(theta_data), np.max(theta_data)
                    theta_fit = np.linspace(theta_min, theta_max, 200)
                    r_fit = a * safe_exp(b * (theta_fit + theta_off))
                    x_fit = x0 + r_fit * np.cos(theta_fit)
                    y_fit = y0 + r_fit * np.sin(theta_fit)
                    if show_transform:
                        x_fit_trans = y_fit
                        y_fit_trans = x_fit
                        spiral_line = main_ax.plot(x_fit_trans, y_fit_trans, 'r-', lw=2, label='Fitted Spiral')
                    else:
                        spiral_line = main_ax.plot(x_fit, y_fit, 'r-', lw=2, label='Fitted Spiral')
                    legend_handles.append(spiral_line[0])
                    legend_labels.append('Fitted Spiral')
                
                title = '2D Projection onto Reference Plane'
            
            main_ax.set_xlabel('X', fontsize=20)  # Increased font size for axis labels
            main_ax.set_ylabel('Y', fontsize=20)  # Increased font size for axis labels
            main_ax.tick_params(axis='both', which='major', labelsize=24)
            main_ax.tick_params(axis='both', which='minor', labelsize=18)
            main_ax.set_title(title, fontsize=15)
            if show_transform:
                if show_all_2d:
                    main_ax.set_xlim(transform_x_lim_all)
                else:
                    main_ax.set_xlim(transform_x_lim_single)
            else:
                main_ax.set_xlim(x2d_lim)
            main_ax.set_ylim(y2d_lim)
            main_ax.axis('equal')
            main_ax.grid(True)
        else:
            if main_ax.name != '3d':
                fig.delaxes(main_ax)
                main_ax = fig.add_subplot(111, projection='3d')
            
            # 3D mode does not support "Show All in 2D", so single frame only
            legend_handles = []
            legend_labels = []
            
            if display_check.get_status()[0]:
                spine_scatter = main_ax.scatter(xs_spine, ys_spine, zs_spine, c='b', marker='o', s=40, label='Spine Points')
                legend_handles.append(spine_scatter)
                legend_labels.append('Spine Points')
            if display_check.get_status()[1]:
                tdcr_scatter = main_ax.scatter(xs_tdcr, ys_tdcr, zs_tdcr, c='g', marker='o', s=40, label='ROI Points')
                legend_handles.append(tdcr_scatter)
                legend_labels.append('ROI Points')
            
            f = 12
            lp = 20
            main_ax.set_xlabel('X', fontsize=f, labelpad=lp)
            main_ax.set_ylabel('Y', fontsize=f, labelpad=lp)
            main_ax.set_zlabel('Z', fontsize=f, labelpad=lp)
            main_ax.tick_params(axis='both', which='major', labelsize=24)
            main_ax.tick_params(axis='both', which='minor', labelsize=18)
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
        
        if not show_all_2d and legend_handles:  # Use standard legend if not in all-2D mode
            legend_ax.legend(legend_handles, legend_labels, loc='upper left', fontsize=13, frameon=True)
    
    plt.draw()
def update(val):
    display_status = display_check.get_status()
    modes_status = modes_check.get_status()
    multi_status = multi_check.get_status()
    options_status = options_check.get_status()
    
    mode_2d = modes_status[0]
    deviation_plot_mode = modes_status[1]
    show_spiral = display_status[2]
    show_all_deviations = multi_status[0]
    show_all_2d = multi_status[1]
    show_average = multi_status[2]
    show_transform = options_status[0]
    
    # Hide slider when showing all/average in deviation mode or all in 2D mode
    hide_slider = deviation_plot_mode and (show_all_deviations or show_average) or (mode_2d and show_all_2d and not deviation_plot_mode)
    slider_ax.set_visible(not hide_slider)
    
    if hide_slider:
        plot_row(0, mode_2d, show_spiral, deviation_plot_mode, show_all_deviations, show_average, show_all_2d, show_transform)
    else:
        plot_row(int(slider.val) - 1, mode_2d, show_spiral, deviation_plot_mode, show_all_deviations, show_average, show_all_2d, show_transform)
def toggle_visibility(label):
    update(None)




display_check.on_clicked(toggle_visibility)
modes_check.on_clicked(toggle_visibility)
multi_check.on_clicked(toggle_visibility)
options_check.on_clicked(toggle_visibility)
slider.on_changed(update)




# Initial plot with new parameter
plot_row(0, False, True, True, True, True, False, False)
plt.show()
