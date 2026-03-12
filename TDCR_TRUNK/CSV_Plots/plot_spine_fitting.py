import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons, RadioButtons
from scipy.optimize import minimize, Bounds, differential_evolution
import matplotlib.lines as mlines  # Added for custom legend handle
from matplotlib import cm  # Added for colormaps



# ------- CONFIG: Edit these to control frame skipping -------
# IGNORE_FIRST_N_FRAMES = 11
IGNORE_FIRST_N_FRAMES = 47      
IGNORE_FIRST_X_DISPLACEMENT = 4.0   
EVERY_N_FRAMES = 1 #min 1
REFERENCE_FRAME = 0  # New: Fixed frame for plane fitting (e.g., 0 for initial state)
# IGNORE_FRAMES = [0,1,2,3,7] 
# Add frames 121 to 144 to the ignore list

IGNORE_FRAMES = list(range(76, 107))
# IGNORE_FRAMES = []  # Add specific frame numbers (actual indices) to ignore, e.g., [10, 25, 50]
# Font size configuration variables
SLIDER_LABEL_FONT_SIZE = 16
CHECKBUTTON_LABEL_FONT_SIZE = 12
GROUP_TITLE_FONT_SIZE = 14
AXIS_LABEL_FONT_SIZE = 20
TICK_MAJOR_FONT_SIZE = 12
TICK_MINOR_FONT_SIZE = 18
PLOT_TITLE_FONT_SIZE = 20
LEGEND_ITEM_FONT_SIZE = 13
LEGEND_TITLE_FONT_SIZE = 13
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

def fit_log_spiral_explicit(x, y, maxiter=500):
    """ Fast Nelder-Mead Spiral Fitting """
    mask = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[mask], y[mask]
    if len(x) < 5: return (np.nan,) * 5

    def linear_log_cost(center):
        x0, y0 = center
        dx, dy = x - x0, y - y0
        r = np.sqrt(dx**2 + dy**2)
        if np.any(r < 1e-5): return 1e10
        theta = np.unwrap(np.arctan2(dy, dx))
        # Linear regression: ln(r) = b*theta + intercept
        A = np.vstack([theta, np.ones(len(theta))]).T
        _, residuals, _, _ = np.linalg.lstsq(A, np.log(r), rcond=None)
        return residuals[0] if len(residuals) > 0 else 1e10

    # Start guess at the last point of the spine
    res = minimize(linear_log_cost, [x[-1], y[-1]], method='Nelder-Mead', options={'maxiter': maxiter})
    if not res.success: return (np.nan,) * 5

    x0_fit, y0_fit = res.x
    dx, dy = x - x0_fit, y - y0_fit
    theta_final = np.unwrap(np.arctan2(dy, dx))
    r_final = np.sqrt(dx**2 + dy**2)
    b_fit, intercept = np.linalg.lstsq(np.vstack([theta_final, np.ones(len(theta_final))]).T, np.log(r_final), rcond=None)[0]
    
    return x0_fit, y0_fit, np.exp(intercept), b_fit, 0


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
    # fit = fit_log_spiral_explicit(x, y, use_global_opt=True)
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
plt.subplots_adjust(left=0.15, right=0.80, bottom=0.15, top=0.85)  # Decreased bottom to 0.15 for more space
deviation_ax = fig.add_axes([0.15, 0.15, 0.65, 0.65])  # Adjusted to shift right
deviation_ax.set_visible(False)
slider_ax = fig.add_axes([0.2, 0.05, 0.5, 0.03])  # Moved slider down slightly to 0.05
slider = Slider(slider_ax, 'Frame Slider', 1, len(considered_frames), valinit=1, valstep=1)
slider.label.set_fontsize(SLIDER_LABEL_FONT_SIZE)



# Group 1: Display Elements
display_ax = plt.axes([0.85, 0.75, 0.13, 0.15])  # Shifted up
display_labels = ['Spine', 'ROI_Points', 'Show Spiral']
display_vals = [True, True, True]
display_check = CheckButtons(display_ax, display_labels, display_vals)
for text in display_check.labels:
    text.set_fontsize(CHECKBUTTON_LABEL_FONT_SIZE)
fig.text(0.85, 0.90, 'Display', fontsize=GROUP_TITLE_FONT_SIZE, fontweight='bold')



# Group 2: View Modes
modes_ax = plt.axes([0.85, 0.60, 0.13, 0.10])  # Shifted up
modes_labels = ['2D Mode', 'Deviation Plot Mode']
modes_vals = [False, False]
modes_check = CheckButtons(modes_ax, modes_labels, modes_vals)
for text in modes_check.labels:
    text.set_fontsize(CHECKBUTTON_LABEL_FONT_SIZE)
fig.text(0.85, 0.70, 'Modes', fontsize=GROUP_TITLE_FONT_SIZE, fontweight='bold')



# Group 3: Multi-Frame Options
multi_ax = plt.axes([0.85, 0.45, 0.13, 0.15])  # Shifted up
multi_labels = ['Show All Frames', 'Show All in 2D', 'Show Average']
multi_vals = [False, False, False]
multi_check = CheckButtons(multi_ax, multi_labels, multi_vals)
for text in multi_check.labels:
    text.set_fontsize(CHECKBUTTON_LABEL_FONT_SIZE)
fig.text(0.85, 0.60, 'Multi-Frame', fontsize=GROUP_TITLE_FONT_SIZE, fontweight='bold')



# Group 4: 2D Transform Options (now with checkboxes for multiple selections)
transform_ax = plt.axes([0.85, 0.25, 0.13, 0.18])  # Shifted up, reduced height
transform_labels = ['90° CCW', '90° CW', '180°', 'Flip X', 'Flip Y', 'Flip XY']
transform_vals = [False] * len(transform_labels)
transform_check = CheckButtons(transform_ax, transform_labels, transform_vals)
for text in transform_check.labels:
    text.set_fontsize(CHECKBUTTON_LABEL_FONT_SIZE)
fig.text(0.85, 0.43, '2D Transforms', fontsize=GROUP_TITLE_FONT_SIZE, fontweight='bold')



# New Group: Legend Toggle
legend_ax = plt.axes([0.85, 0.20, 0.13, 0.05])  # Shifted up
legend_labels = ['Show Legend']
legend_vals = [True]
legend_check = CheckButtons(legend_ax, legend_labels, legend_vals)
for text in legend_check.labels:
    text.set_fontsize(CHECKBUTTON_LABEL_FONT_SIZE)
fig.text(0.85, 0.25, 'Legend', fontsize=GROUP_TITLE_FONT_SIZE, fontweight='bold')



# New Group: 3D Options (expanded for two toggles)
trace_ax = plt.axes([0.85, 0.12, 0.13, 0.10])  # Shifted up
trace_labels = ['Show All Points', 'Show Workspace']
trace_vals = [False, False]
trace_check = CheckButtons(trace_ax, trace_labels, trace_vals)
for text in trace_check.labels:
    text.set_fontsize(CHECKBUTTON_LABEL_FONT_SIZE)
fig.text(0.85, 0.23, '3D Options', fontsize=GROUP_TITLE_FONT_SIZE, fontweight='bold')



# New: Axis selection for workspace (radio buttons)
axis_ax = plt.axes([0.85, 0.02, 0.13, 0.10])  # Positioned safely above bottom
axis_radio = RadioButtons(axis_ax, ('X', 'Y', 'Z'), active=1)  # Default to Y
for text in axis_radio.labels:
    text.set_fontsize(CHECKBUTTON_LABEL_FONT_SIZE)
fig.text(0.85, 0.13, 'Workspace Axis', fontsize=GROUP_TITLE_FONT_SIZE, fontweight='bold')



spine_scatter = None
tdcr_scatter = None
spiral_line = None


def apply_transform(points, transform_option):
    """
    Apply a single 2D transformation to points (Nx2 array).
    """
    x = points[:, 0]
    y = points[:, 1]
    if transform_option == '90° CCW':
        return np.column_stack([-y, x])
    elif transform_option == '90° CW':
        return np.column_stack([y, -x])
    elif transform_option == '180°':
        return np.column_stack([-x, -y])
    elif transform_option == 'Flip X':
        return np.column_stack([-x, y])
    elif transform_option == 'Flip Y':
        return np.column_stack([x, -y])
    elif transform_option == 'Flip XY':
        return np.column_stack([-x, -y])
    else:
        return points  # No transform


def transform_2d_points(points, selected_transforms):
    """
    Apply multiple transformations in sequence.
    Order: rotations first (90 CCW, 90 CW, 180), then flips (X, Y, XY).
    """
    # Define order of application
    order = ['90° CCW', '90° CW', '180°', 'Flip X', 'Flip Y', 'Flip XY']
    for opt in order:
        if opt in selected_transforms:
            points = apply_transform(points, opt)
    return points


def plot_row(considered_idx, mode_2d, show_spiral, deviation_plot_mode, show_all_deviations, show_average, show_all_2d, selected_transforms, show_legend=True, show_all_3d_points=False, show_workspace=False, rotation_axis='Y'):
    global spine_scatter, tdcr_scatter, spiral_line, main_ax, deviation_ax
    
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
    # xs_spine[low_displacement_mask] = df_spine.loc[REFERENCE_FRAME, x_cols_spine].values[low_displacement_mask]
    # ys_spine[low_displacement_mask] = df_spine.loc[REFERENCE_FRAME, y_cols_spine].values[low_displacement_mask]
    # zs_spine[low_displacement_mask] = df_spine.loc[REFERENCE_FRAME, z_cols_spine].values[low_displacement_mask]
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
        
        deviation_ax.set_xlabel('Point Index', fontsize=AXIS_LABEL_FONT_SIZE)
        deviation_ax.set_ylabel('Deviation (mm)', fontsize=AXIS_LABEL_FONT_SIZE)
        deviation_ax.tick_params(axis='both', which='major', labelsize=TICK_MAJOR_FONT_SIZE)
        deviation_ax.tick_params(axis='both', which='minor', labelsize=TICK_MINOR_FONT_SIZE)
        deviation_ax.grid(True)
        deviation_ax.set_ylim(0, 30)  # Updated y-axis limit

        # --- NEW CODE: Force exact point indices on X-axis ---
        num_points = len(points_spine)
        filter_idx = int(IGNORE_FIRST_X_DISPLACEMENT)
        visible_indices = np.arange(filter_idx, num_points)
        
        deviation_ax.set_xticks(visible_indices) # Forces exact ticks (e.g., 4, 5, 6, 7...)
        deviation_ax.set_xlim(visible_indices[0], visible_indices[-1]) # Keeps the graph tightly fit
        # -----------------------------------------------------
        
        if show_average and show_all_deviations:
            deviation_ax.set_title('Deviation Plot', fontsize=PLOT_TITLE_FONT_SIZE)
        elif show_average:
            deviation_ax.set_title('Average Deviation Plot - Considered Frames', fontsize=PLOT_TITLE_FONT_SIZE)
        elif show_all_deviations:
            deviation_ax.set_title('Deviation Plot - All Considered Frames', fontsize=PLOT_TITLE_FONT_SIZE)
        else:
            deviation_ax.set_title(f'Deviation Plot - Frame {actual_idx + 1} ', fontsize=PLOT_TITLE_FONT_SIZE)
        
        if show_legend and legend_handles:
            deviation_ax.legend(handles=legend_handles, fontsize=LEGEND_ITEM_FONT_SIZE, loc='upper left', bbox_to_anchor=(-0.3, 1.2))  # Top left outside, more up
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
                    
                    # Apply selected transformations
                    spine_2d_i = transform_2d_points(spine_2d_i, selected_transforms)
                    tdcr_2d_i = transform_2d_points(tdcr_2d_i, selected_transforms)
                    
                    color = colors[j]
                    
                    if display_check.get_status()[0]:  # Spine
                        main_ax.scatter(spine_2d_i[:, 0], spine_2d_i[:, 1], c=[color], marker='o', s=40, alpha=alpha_val)
                    
                    if display_check.get_status()[1]:  # ROI_Points
                        main_ax.scatter(tdcr_2d_i[:, 0], tdcr_2d_i[:, 1], c=[color], marker='o', s=40, alpha=alpha_val)
                    
                    if show_spiral:
                        x = spine_2d_i[:, 0]
                        y = spine_2d_i[:, 1]
                        x0, y0, a, b, theta_off = fit_log_spiral_explicit(x, y)
                        theta_data = np.unwrap(np.arctan2(y - y0, x - x0))
                        theta_min, theta_max = np.min(theta_data), np.max(theta_data)
                        theta_fit = np.linspace(theta_min, theta_max, 200)
                        r_fit = a * safe_exp(b * (theta_fit + theta_off))
                        x_fit = x0 + r_fit * np.cos(theta_fit)
                        y_fit = y0 + r_fit * np.sin(theta_fit)
                        main_ax.plot(x_fit, y_fit, color=color, lw=2, alpha=alpha_val)
                
                # Improved legend with representative L1 values
                low = mlines.Line2D([], [], color=colors[min_l1_idx], linewidth=2, label=f'Min Disp: {min_l1:.2f} mm')
                mid = mlines.Line2D([], [], color=colors[mid_l1_idx], linewidth=2, label=f'Mid Disp: {mid_l1:.2f} mm')
                high = mlines.Line2D([], [], color=colors[max_l1_idx], linewidth=2, label=f'Max Disp: {max_l1:.2f} mm')
                legend_handles = [low, mid, high]
                if show_legend:
                    main_ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(-0.1, 1.2), fontsize=LEGEND_ITEM_FONT_SIZE, title='Tendon Displacement', title_fontsize=LEGEND_TITLE_FONT_SIZE)  # Adjusted less left
                
                title = 'Bending Curve'  # Updated title
            else:
                # Single frame plotting
                spine_2d = project_to_plane(points_spine, ref_centroid, ref_e1, ref_e2)
                tdcr_2d = project_to_plane(points_tdcr, ref_centroid, ref_e1, ref_e2)
                
                # Apply selected transformations
                spine_2d = transform_2d_points(spine_2d, selected_transforms)
                tdcr_2d = transform_2d_points(tdcr_2d, selected_transforms)
                spine_x = spine_2d[:, 0]
                spine_y = spine_2d[:, 1]
                tdcr_x = tdcr_2d[:, 0]
                tdcr_y = tdcr_2d[:, 1]
                
                if display_check.get_status()[0]:
                    spine_scatter = main_ax.scatter(spine_x, spine_y, c='b', marker='o', s=40, label='Continuum Backbone')
                    legend_handles.append(spine_scatter)
                    legend_labels.append('Continuum Backbone')
                if display_check.get_status()[1]:
                    tdcr_scatter = main_ax.scatter(tdcr_x, tdcr_y, c='g', marker='o', s=40, label='ROI Points')
                    legend_handles.append(tdcr_scatter)
                    legend_labels.append('ROI Points')
                
                if show_spiral:
                    x = spine_2d[:, 0]
                    y = spine_2d[:, 1]
                    x0, y0, a, b, theta_off = fit_log_spiral_explicit(x, y)
                    theta_data = np.unwrap(np.arctan2(y - y0, x - x0))
                    theta_min, theta_max = np.min(theta_data), np.max(theta_data)
                    theta_fit = np.linspace(theta_min, theta_max, 200)
                    r_fit = a * safe_exp(b * (theta_fit + theta_off))
                    x_fit = x0 + r_fit * np.cos(theta_fit)
                    y_fit = y0 + r_fit * np.sin(theta_fit)
                    spiral_line = main_ax.plot(x_fit, y_fit, 'r-', lw=2, label='Bending Curve')
                    legend_handles.append(spiral_line[0])
                    legend_labels.append('Bending Curve')
                
                # Add legend for single frame 2D
                if show_legend and legend_handles:
                    main_ax.legend(handles=legend_handles, labels=legend_labels, loc='upper left', bbox_to_anchor=(-0.1, 1.2), fontsize=LEGEND_ITEM_FONT_SIZE)
                
                title = 'Bending Curve'
            
            main_ax.set_xlabel('X', fontsize=AXIS_LABEL_FONT_SIZE)  # Increased font size for axis labels
            main_ax.set_ylabel('Y', fontsize=AXIS_LABEL_FONT_SIZE)  # Increased font size for axis labels
            main_ax.tick_params(axis='both', which='major', labelsize=TICK_MAJOR_FONT_SIZE)
            main_ax.tick_params(axis='both', which='minor', labelsize=TICK_MINOR_FONT_SIZE)
            main_ax.set_title(title, fontsize=PLOT_TITLE_FONT_SIZE)
            if selected_transforms:  # If any transform is selected
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
            
            # 3D mode plotting
            legend_handles = []
            legend_labels = []
            
            if show_all_3d_points:
                # Show all points from all frames in 3D
                num_frames = len(considered_frames)
                colors = cm.viridis(np.linspace(0, 1, num_frames))
                alpha_val = 0.4  # Translucent for overlay
                
                # Collect L1 values for legend
                l1_values = np.array([df_spine.loc[i + IGNORE_FIRST_N_FRAMES, 'L1'] for i in considered_frames])
                min_l1_idx = np.argmin(l1_values)
                mid_l1_idx = len(l1_values) // 2
                max_l1_idx = np.argmax(l1_values)
                min_l1 = l1_values[min_l1_idx]
                mid_l1 = l1_values[mid_l1_idx]
                max_l1 = l1_values[max_l1_idx]
                
                # Get base point from reference frame (first point)
                base_x = df_spine.loc[REFERENCE_FRAME, x_cols_spine[0]]
                base_y = df_spine.loc[REFERENCE_FRAME, y_cols_spine[0]]
                base_z = df_spine.loc[REFERENCE_FRAME, z_cols_spine[0]]
                
                if show_workspace:
                    # Collect all original points
                    all_spine_points = []
                    all_tdcr_points = []
                    frame_colors = []
                    for j, i in enumerate(considered_frames):
                        actual_i = i + IGNORE_FIRST_N_FRAMES
                        xs_spine_i = df_spine.loc[actual_i, x_cols_spine].values
                        ys_spine_i = df_spine.loc[actual_i, y_cols_spine].values
                        zs_spine_i = df_spine.loc[actual_i, z_cols_spine].values
                        points_spine_i = np.vstack([xs_spine_i, ys_spine_i, zs_spine_i]).T
                        
                        # Apply displacement filtering
                        spine_displacements_i = calculate_point_displacements(points_spine_i)
                        if not np.all(spine_displacements_i == 0):
                            low_displacement_mask_i = spine_displacements_i < IGNORE_FIRST_X_DISPLACEMENT
                            xs_spine_i[low_displacement_mask_i] = df_spine.loc[REFERENCE_FRAME, x_cols_spine].values[low_displacement_mask_i]
                            ys_spine_i[low_displacement_mask_i] = df_spine.loc[REFERENCE_FRAME, y_cols_spine].values[low_displacement_mask_i]
                            zs_spine_i[low_displacement_mask_i] = df_spine.loc[REFERENCE_FRAME, z_cols_spine].values[low_displacement_mask_i]
                            points_spine_i = np.vstack([xs_spine_i, ys_spine_i, zs_spine_i]).T
                        
                        all_spine_points.append(points_spine_i)
                        
                        xs_tdcr_i = df_tdcr.loc[actual_i, x_cols_tdcr].values
                        ys_tdcr_i = df_tdcr.loc[actual_i, y_cols_tdcr].values
                        zs_tdcr_i = df_tdcr.loc[actual_i, z_cols_tdcr].values
                        points_tdcr_i = np.vstack([xs_tdcr_i, ys_tdcr_i, zs_tdcr_i]).T
                        all_tdcr_points.append(points_tdcr_i)
                        
                        frame_colors.append(colors[j])
                    
                    # Determine ranges for the line
                    all_coords = {
                        'X': np.concatenate([p[:,0] for p in all_spine_points] + [p[:,0] for p in all_tdcr_points]),
                        'Y': np.concatenate([p[:,1] for p in all_spine_points] + [p[:,1] for p in all_tdcr_points]),
                        'Z': np.concatenate([p[:,2] for p in all_spine_points] + [p[:,2] for p in all_tdcr_points])
                    }
                    min_coord = {ax: np.min(all_coords[ax]) for ax in 'XYZ'}
                    max_coord = {ax: np.max(all_coords[ax]) for ax in 'XYZ'}
                    
                    # Draw the downward line (rotation axis) through base point along selected axis
                    line_x = [base_x, base_x]
                    line_y = [base_y, base_y]
                    line_z = [base_z, base_z]
                    if rotation_axis == 'X':
                        line_x = [min_coord['X'], max_coord['X']]
                    elif rotation_axis == 'Y':
                        line_y = [min_coord['Y'], max_coord['Y']]
                    elif rotation_axis == 'Z':
                        line_z = [min_coord['Z'], max_coord['Z']]
                    main_ax.plot(line_x, line_y, line_z, 'r-', linewidth=3, label='Rotation Axis')
                    legend_handles.append(mlines.Line2D([], [], color='r', linewidth=3, label='Rotation Axis'))
                    legend_labels.append('Rotation Axis')
                    
                    # Revolve all points around the selected axis
                    num_rotations = 36  # Every 10 degrees
                    thetas = np.linspace(0, 2 * np.pi, num_rotations, endpoint=False)
                    
                    # For spine points
                    revolved_spine_x = []
                    revolved_spine_y = []
                    revolved_spine_z = []
                    revolved_spine_colors = []
                    
                    for j, points in enumerate(all_spine_points):
                        color = frame_colors[j]
                        num_pts = len(points)
                        for theta in thetas:
                            cos_theta = np.cos(theta)
                            sin_theta = np.sin(theta)
                            if rotation_axis == 'X':
                                # Rotate around X: fix X, rotate YZ
                                shift_y = points[:,1] - base_y
                                shift_z = points[:,2] - base_z
                                new_shift_y = shift_y * cos_theta - shift_z * sin_theta
                                new_shift_z = shift_y * sin_theta + shift_z * cos_theta
                                new_x = points[:,0]
                                new_y = new_shift_y + base_y
                                new_z = new_shift_z + base_z
                            elif rotation_axis == 'Y':
                                # Rotate around Y: fix Y, rotate XZ
                                shift_x = points[:,0] - base_x
                                shift_z = points[:,2] - base_z
                                new_shift_x = shift_x * cos_theta - shift_z * sin_theta
                                new_shift_z = shift_x * sin_theta + shift_z * cos_theta
                                new_x = new_shift_x + base_x
                                new_y = points[:,1]
                                new_z = new_shift_z + base_z
                            elif rotation_axis == 'Z':
                                # Rotate around Z: fix Z, rotate XY
                                shift_x = points[:,0] - base_x
                                shift_y = points[:,1] - base_y
                                new_shift_x = shift_x * cos_theta - shift_y * sin_theta
                                new_shift_y = shift_x * sin_theta + shift_y * cos_theta
                                new_x = new_shift_x + base_x
                                new_y = new_shift_y + base_y
                                new_z = points[:,2]
                            revolved_spine_x.extend(new_x)
                            revolved_spine_y.extend(new_y)
                            revolved_spine_z.extend(new_z)
                            revolved_spine_colors.extend([color] * num_pts)
                    
                    # For tdcr points
                    revolved_tdcr_x = []
                    revolved_tdcr_y = []
                    revolved_tdcr_z = []
                    revolved_tdcr_colors = []
                    
                    for j, points in enumerate(all_tdcr_points):
                        color = frame_colors[j]
                        num_pts = len(points)
                        for theta in thetas:
                            cos_theta = np.cos(theta)
                            sin_theta = np.sin(theta)
                            if rotation_axis == 'X':
                                shift_y = points[:,1] - base_y
                                shift_z = points[:,2] - base_z
                                new_shift_y = shift_y * cos_theta - shift_z * sin_theta
                                new_shift_z = shift_y * sin_theta + shift_z * cos_theta
                                new_x = points[:,0]
                                new_y = new_shift_y + base_y
                                new_z = new_shift_z + base_z
                            elif rotation_axis == 'Y':
                                shift_x = points[:,0] - base_x
                                shift_z = points[:,2] - base_z
                                new_shift_x = shift_x * cos_theta - shift_z * sin_theta
                                new_shift_z = shift_x * sin_theta + shift_z * cos_theta
                                new_x = new_shift_x + base_x
                                new_y = points[:,1]
                                new_z = new_shift_z + base_z
                            elif rotation_axis == 'Z':
                                shift_x = points[:,0] - base_x
                                shift_y = points[:,1] - base_y
                                new_shift_x = shift_x * cos_theta - shift_y * sin_theta
                                new_shift_y = shift_x * sin_theta + shift_y * cos_theta
                                new_x = new_shift_x + base_x
                                new_y = new_shift_y + base_y
                                new_z = points[:,2]
                            revolved_tdcr_x.extend(new_x)
                            revolved_tdcr_y.extend(new_y)
                            revolved_tdcr_z.extend(new_z)
                            revolved_tdcr_colors.extend([color] * num_pts)
                    
                    # Plot revolved points
                    if display_check.get_status()[0]:  # Spine
                        main_ax.scatter(revolved_spine_x, revolved_spine_y, revolved_spine_z, c=revolved_spine_colors, marker='o', s=10, alpha=0.1)
                    
                    if display_check.get_status()[1]:  # ROI_Points
                        main_ax.scatter(revolved_tdcr_x, revolved_tdcr_y, revolved_tdcr_z, c=revolved_tdcr_colors, marker='o', s=10, alpha=0.1)
                    
                    title = f'Workspace Visualization - All Considered Frames (Around {rotation_axis}-Axis)'
                else:
                    # Regular all points without workspace
                    for j, i in enumerate(considered_frames):
                        actual_i = i + IGNORE_FIRST_N_FRAMES
                        xs_spine_i = df_spine.loc[actual_i, x_cols_spine].values
                        ys_spine_i = df_spine.loc[actual_i, y_cols_spine].values
                        zs_spine_i = df_spine.loc[actual_i, z_cols_spine].values
                        points_spine_i = np.vstack([xs_spine_i, ys_spine_i, zs_spine_i]).T
                        
                        # Apply displacement filtering
                        spine_displacements_i = calculate_point_displacements(points_spine_i)
                        if not np.all(spine_displacements_i == 0):
                            low_displacement_mask_i = spine_displacements_i < IGNORE_FIRST_X_DISPLACEMENT
                            xs_spine_i[low_displacement_mask_i] = df_spine.loc[REFERENCE_FRAME, x_cols_spine].values[low_displacement_mask_i]
                            ys_spine_i[low_displacement_mask_i] = df_spine.loc[REFERENCE_FRAME, y_cols_spine].values[low_displacement_mask_i]
                            zs_spine_i[low_displacement_mask_i] = df_spine.loc[REFERENCE_FRAME, z_cols_spine].values[low_displacement_mask_i]
                        
                        xs_tdcr_i = df_tdcr.loc[actual_i, x_cols_tdcr].values
                        ys_tdcr_i = df_tdcr.loc[actual_i, y_cols_tdcr].values
                        zs_tdcr_i = df_tdcr.loc[actual_i, z_cols_tdcr].values
                        
                        color = colors[j]
                        
                        if display_check.get_status()[0]:  # Spine
                            main_ax.scatter(xs_spine_i, ys_spine_i, zs_spine_i, c=[color], marker='o', s=30, alpha=alpha_val)
                        
                        if display_check.get_status()[1]:  # ROI_Points
                            main_ax.scatter(xs_tdcr_i, ys_tdcr_i, zs_tdcr_i, c=[color], marker='o', s=30, alpha=alpha_val)
                    
                    title = 'All 3D Points - All Considered Frames'
                
                # Create legend for min, mid, max
                low = mlines.Line2D([], [], color=colors[min_l1_idx], marker='o', linestyle='None', label=f'Min Disp: {min_l1:.2f} mm')
                mid = mlines.Line2D([], [], color=colors[mid_l1_idx], marker='o', linestyle='None', label=f'Mid Disp: {mid_l1:.2f} mm')
                high = mlines.Line2D([], [], color=colors[max_l1_idx], marker='o', linestyle='None', label=f'Max Disp: {max_l1:.2f} mm')
                legend_handles.extend([low, mid, high])
                legend_labels.extend([f'Min Disp: {min_l1:.2f} mm', f'Mid Disp: {mid_l1:.2f} mm', f'Max Disp: {max_l1:.2f} mm'])
                
                if show_legend:
                    main_ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(-0.1, 1.2), fontsize=LEGEND_ITEM_FONT_SIZE, title='Tendon Displacement', title_fontsize=LEGEND_TITLE_FONT_SIZE)
                
                main_ax.set_title(title, fontsize=PLOT_TITLE_FONT_SIZE)
            else:
                # Single frame 3D plotting
                if display_check.get_status()[0]:
                    spine_scatter = main_ax.scatter(xs_spine, ys_spine, zs_spine, c='b', marker='o', s=40, label='Continuum Backbone')
                    legend_handles.append(spine_scatter)
                    legend_labels.append('Continuum Backbone')
                    
                if display_check.get_status()[1]:
                    tdcr_scatter = main_ax.scatter(xs_tdcr, ys_tdcr, zs_tdcr, c='g', marker='o', s=40, label='ROI Points')
                    legend_handles.append(tdcr_scatter)
                    legend_labels.append('ROI Points')
                
                if show_legend and legend_handles:
                    main_ax.legend(handles=legend_handles, labels=legend_labels, loc='upper left', bbox_to_anchor=(-0.1, 1.2), fontsize=LEGEND_ITEM_FONT_SIZE, frameon=True)
                
                main_ax.set_title(f'3D Plot - Frame {actual_idx + 1}', fontsize=PLOT_TITLE_FONT_SIZE)
            
            f = 12
            lp = 20
            main_ax.set_xlabel('X', fontsize=AXIS_LABEL_FONT_SIZE, labelpad=lp)
            main_ax.set_ylabel('Y', fontsize=AXIS_LABEL_FONT_SIZE, labelpad=lp)
            main_ax.set_zlabel('Z', fontsize=AXIS_LABEL_FONT_SIZE, labelpad=lp)
            main_ax.tick_params(axis='both', which='major', labelsize=TICK_MAJOR_FONT_SIZE)
            main_ax.tick_params(axis='both', which='minor', labelsize=TICK_MINOR_FONT_SIZE)
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
    
    plt.draw()


def update(val):
    display_status = display_check.get_status()
    modes_status = modes_check.get_status()
    multi_status = multi_check.get_status()
    legend_status = legend_check.get_status()
    transform_status = transform_check.get_status()
    trace_status = trace_check.get_status()
    
    mode_2d = modes_status[0]
    deviation_plot_mode = modes_status[1]
    show_spiral = display_status[2]
    show_all_deviations = multi_status[0]
    show_all_2d = multi_status[1]
    show_average = multi_status[2]
    show_legend = legend_status[0]
    show_all_3d_points = trace_status[0]
    show_workspace = trace_status[1]
    rotation_axis = axis_radio.value_selected
    
    # Get list of selected transforms
    selected_transforms = [transform_labels[i] for i, status in enumerate(transform_status) if status]
    
    # Hide slider when showing all/average in deviation mode, all in 2D mode, all points in 3D mode, or workspace
    hide_slider = (deviation_plot_mode and (show_all_deviations or show_average)) or \
                  (mode_2d and show_all_2d and not deviation_plot_mode) or \
                  (not mode_2d and not deviation_plot_mode and (show_all_3d_points or show_workspace))
    slider_ax.set_visible(not hide_slider)
    
    if hide_slider:
        plot_row(0, mode_2d, show_spiral, deviation_plot_mode, show_all_deviations, show_average, show_all_2d, selected_transforms, show_legend, show_all_3d_points, show_workspace, rotation_axis)
    else:
        plot_row(int(slider.val) - 1, mode_2d, show_spiral, deviation_plot_mode, show_all_deviations, show_average, show_all_2d, selected_transforms, show_legend, show_all_3d_points, show_workspace, rotation_axis)


def toggle_visibility(label):
    update(None)



display_check.on_clicked(toggle_visibility)
modes_check.on_clicked(toggle_visibility)
multi_check.on_clicked(toggle_visibility)
legend_check.on_clicked(toggle_visibility)
transform_check.on_clicked(toggle_visibility)
trace_check.on_clicked(toggle_visibility)
axis_radio.on_clicked(toggle_visibility)
slider.on_changed(update)



# Initial plot
plot_row(0, False, True, True, True, True, False, [], True, False, False, 'Y')
plt.show()
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.widgets import Slider, CheckButtons, RadioButtons
# from scipy.optimize import minimize
# import matplotlib.lines as mlines
# from matplotlib import cm

# # ------- CONFIG: Optimized for Speed and Accuracy -------
# IGNORE_FIRST_N_FRAMES = 0      
# IGNORE_FIRST_X_DISPLACEMENT = 0.0   
# EVERY_N_FRAMES = 1 
# REFERENCE_FRAME = 0  
# IGNORE_FRAMES = [0]  

# # UI Font Sizes
# SLIDER_LABEL_FONT_SIZE = 16
# CHECKBUTTON_LABEL_FONT_SIZE = 12
# GROUP_TITLE_FONT_SIZE = 14
# AXIS_LABEL_FONT_SIZE = 20
# TICK_MAJOR_FONT_SIZE = 20
# TICK_MINOR_FONT_SIZE = 18
# PLOT_TITLE_FONT_SIZE = 20
# LEGEND_ITEM_FONT_SIZE = 13
# LEGEND_TITLE_FONT_SIZE = 13

# # ------- VISUAL SETUP -------
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['cmr10']
# plt.rcParams["axes.formatter.use_mathtext"] = True
# plt.rcParams['xtick.labelsize'] = 28
# plt.rcParams['ytick.labelsize'] = 28

# # --- Load data ---
# try:
#     df_spine = pd.read_csv('tdcr_trunk_spine.csv')
#     df_tdcr = pd.read_csv('tdcr_trunk_output.csv')
# except FileNotFoundError:
#     print("Error: Ensure 'tdcr_trunk_spine.csv' and 'tdcr_trunk_output.csv' are in the directory.")
#     exit()

# x_cols_spine = [col for col in df_spine.columns if col.startswith('x')]
# y_cols_spine = [col for col in df_spine.columns if col.startswith('y')]
# z_cols_spine = [col for col in df_spine.columns if col.startswith('z')]
# x_cols_tdcr = [col for col in df_tdcr.columns if col.startswith('x')]
# y_cols_tdcr = [col for col in df_tdcr.columns if col.startswith('y')]
# z_cols_tdcr = [col for col in df_tdcr.columns if col.startswith('z')]

# # Global Limits
# all_x = np.concatenate([df_spine[x_cols_spine].values.flatten(), df_tdcr[x_cols_tdcr].values.flatten()])
# all_y = np.concatenate([df_spine[y_cols_spine].values.flatten(), df_tdcr[y_cols_tdcr].values.flatten()])
# all_z = np.concatenate([df_spine[z_cols_spine].values.flatten(), df_tdcr[z_cols_tdcr].values.flatten()])
# x_min, x_max = np.nanmin(all_x), np.nanmax(all_x)
# y_min, y_max = np.nanmin(all_y), np.nanmax(all_y)
# z_min, z_max = np.nanmin(all_z), np.nanmax(all_z)

# max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
# x_center, y_center, z_center = (x_max + x_min) / 2, (y_max + y_min) / 2, (z_max + z_min) / 2
# margin = 0.1
# x_lim = [x_center - max_range/2 - margin*max_range, x_center + max_range/2 + margin*max_range]
# y_lim = [y_center - max_range/2 - margin*max_range, y_center + max_range/2 + margin*max_range]
# z_lim = [z_center - max_range/2 - margin*max_range, z_center + max_range/2 + margin*max_range]

# def fit_plane(points):
#     points = points[~np.isnan(points).any(axis=1)]
#     if len(points) < 3: return np.mean(points, axis=0), np.array([0,0,1]), np.array([1,0,0]), np.array([0,1,0])
#     centroid = np.mean(points, axis=0)
#     _, _, vh = np.linalg.svd(points - centroid)
#     return centroid, vh[2, :], vh[0, :], vh[1, :]

# def project_to_plane(points, origin, e1, e2):
#     rel = points - origin
#     return np.stack([np.dot(rel, e1), np.dot(rel, e2)], axis=1)

# avg_points = []
# for idx in range(min(3, len(df_spine))):
#     p = np.vstack([df_spine.loc[idx, x_cols_spine], df_spine.loc[idx, y_cols_spine], df_spine.loc[idx, z_cols_spine]]).T
#     avg_points.extend(p)
# ref_centroid, ref_normal, ref_e1, ref_e2 = fit_plane(np.array(avg_points))

# spiral_fit_cache = {}

# # --- NEW: Fast Nelder-Mead Spiral Fitting ---
# def fit_log_spiral_explicit(x, y, maxiter=500):
#     mask = ~np.isnan(x) & ~np.isnan(y)
#     x, y = x[mask], y[mask]
#     if len(x) < 5: return (np.nan,) * 5

#     def linear_log_cost(center):
#         x0, y0 = center
#         dx, dy = x - x0, y - y0
#         r = np.sqrt(dx**2 + dy**2)
#         if np.any(r < 1e-5): return 1e10
#         theta = np.unwrap(np.arctan2(dy, dx))
#         # Linear regression: ln(r) = b*theta + intercept
#         A = np.vstack([theta, np.ones(len(theta))]).T
#         _, residuals, _, _ = np.linalg.lstsq(A, np.log(r), rcond=None)
#         return residuals[0] if len(residuals) > 0 else 1e10

#     # Start guess at the last point
#     res = minimize(linear_log_cost, [x[-1], y[-1]], method='Nelder-Mead', options={'maxiter': maxiter})
#     if not res.success: return (np.nan,) * 5

#     x0_fit, y0_fit = res.x
#     dx, dy = x - x0_fit, y - y0_fit
#     theta_final = np.unwrap(np.arctan2(dy, dx))
#     r_final = np.sqrt(dx**2 + dy**2)
#     b_fit, intercept = np.linalg.lstsq(np.vstack([theta_final, np.ones(len(theta_final))]).T, np.log(r_final), rcond=None)[0]
    
#     return x0_fit, y0_fit, np.exp(intercept), b_fit, 0

# def fit_log_spiral_explicit_cached(frame_idx):
#     actual_idx = frame_idx + IGNORE_FIRST_N_FRAMES
#     if actual_idx in spiral_fit_cache: return spiral_fit_cache[actual_idx]
    
#     p = np.vstack([df_spine.loc[actual_idx, x_cols_spine], df_spine.loc[actual_idx, y_cols_spine], df_spine.loc[actual_idx, z_cols_spine]]).T
#     spine_2d = project_to_plane(p, ref_centroid, ref_e1, ref_e2)
#     fit = fit_log_spiral_explicit(spine_2d[:, 0], spine_2d[:, 1])
#     spiral_fit_cache[actual_idx] = fit
#     return fit

# n_frames = len(df_spine) - IGNORE_FIRST_N_FRAMES
# considered_frames = [i for i in range(n_frames) if (i % EVERY_N_FRAMES == 0) and (i + IGNORE_FIRST_N_FRAMES not in IGNORE_FRAMES)]

# # --- GUI Setup ---
# fig = plt.figure(figsize=(12, 9))
# main_ax = fig.add_subplot(111, projection='3d')
# plt.subplots_adjust(left=0.1, right=0.75, bottom=0.2)

# deviation_ax = fig.add_axes([0.1, 0.2, 0.6, 0.7])
# deviation_ax.set_visible(False)

# slider_ax = fig.add_axes([0.2, 0.05, 0.5, 0.03])
# slider = Slider(slider_ax, 'Frame', 1, len(considered_frames), valinit=1, valstep=1)

# # Checkboxes
# display_ax = plt.axes([0.8, 0.7, 0.15, 0.2])
# display_check = CheckButtons(display_ax, ['Spine', 'Points', 'Spiral'], [True, True, True])

# modes_ax = plt.axes([0.8, 0.55, 0.15, 0.1])
# modes_check = CheckButtons(modes_ax, ['2D Mode', 'Deviations'], [False, False])

# def plot_row(idx):
#     main_ax.clear()
#     deviation_ax.clear()
    
#     mode_2d = modes_check.get_status()[0]
#     dev_mode = modes_check.get_status()[1]
#     show_spiral = display_check.get_status()[2]
    
#     deviation_ax.set_visible(dev_mode)
#     main_ax.set_visible(not dev_mode)

#     actual_idx = considered_frames[int(idx)-1] + IGNORE_FIRST_N_FRAMES
#     p_spine = np.vstack([df_spine.loc[actual_idx, x_cols_spine], df_spine.loc[actual_idx, y_cols_spine], df_spine.loc[actual_idx, z_cols_spine]]).T
    
#     if dev_mode:
#         x0, y0, a, b, _ = fit_log_spiral_explicit_cached(int(idx)-1)
#         spine_2d = project_to_plane(p_spine, ref_centroid, ref_e1, ref_e2)
#         theta = np.unwrap(np.arctan2(spine_2d[:,1]-y0, spine_2d[:,0]-x0))
#         r_model = a * np.exp(b * theta)
#         err = np.abs(np.sqrt((spine_2d[:,0]-x0)**2 + (spine_2d[:,1]-y0)**2) - r_model)
#         deviation_ax.plot(err, 'r-o')
#         deviation_ax.set_title(f"Spiral Deviation - Frame {actual_idx}")
#     elif mode_2d:
#         spine_2d = project_to_plane(p_spine, ref_centroid, ref_e1, ref_e2)
#         main_ax.view_init(elev=90, azim=-90) # Top down for 2D
#         main_ax.scatter(spine_2d[:,0], spine_2d[:,1], 0, c='blue')
#         if show_spiral:
#             x0, y0, a, b, _ = fit_log_spiral_explicit_cached(int(idx)-1)
#             t_plot = np.linspace(0, 10, 200)
#             rs = a * np.exp(b * t_plot)
#             main_ax.plot(x0 + rs*np.cos(t_plot), y0 + rs*np.sin(t_plot), 0, 'r--')
#     else:
#         main_ax.plot(p_spine[:,0], p_spine[:,1], p_spine[:,2], 'b-o')
#         main_ax.set_xlim(x_lim); main_ax.set_ylim(y_lim); main_ax.set_zlim(z_lim)

# def update(val):
#     plot_row(slider.val)
#     fig.canvas.draw_idle()

# slider.on_changed(update)
# display_check.on_clicked(update)
# modes_check.on_clicked(update)

# plot_row(1)
# plt.show()