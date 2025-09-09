import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons
import spiral_optimizer  # Our C++ module

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

# Global variables that will be controlled by sliders
IGNORE_FIRST_N_FRAMES = 750
IGNORE_FIRST_X_DISPLACEMENT = 6.0

# Convert DataFrame columns to NumPy arrays
spine_data = np.column_stack([df_spine[x_cols_spine].values, 
                             df_spine[y_cols_spine].values, 
                             df_spine[z_cols_spine].values])
tdcr_data = np.column_stack([df_tdcr[x_cols_tdcr].values, 
                            df_tdcr[y_cols_tdcr].values, 
                            df_tdcr[z_cols_tdcr].values])

n_spine_points = len(x_cols_spine)
n_tdcr_points = len(x_cols_tdcr)

def fit_plane(points):
    """Fit plane using SVD"""
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
    """Project points to plane"""
    rel = points - origin
    x = np.dot(rel, e1)
    y = np.dot(rel, e2)
    return np.column_stack([x, y])

def update_global_limits():
    """Recalculate global limits when frame ignore setting changes"""
    global x_lim, y_lim, z_lim, ref_centroid, ref_normal, ref_e1, ref_e2
    
    all_x = np.concatenate([df_spine[x_cols_spine].values.flatten()[IGNORE_FIRST_N_FRAMES:], 
                           df_tdcr[x_cols_tdcr].values.flatten()[IGNORE_FIRST_N_FRAMES:]])
    all_y = np.concatenate([df_spine[y_cols_spine].values.flatten()[IGNORE_FIRST_N_FRAMES:], 
                           df_tdcr[y_cols_tdcr].values.flatten()[IGNORE_FIRST_N_FRAMES:]])
    all_z = np.concatenate([df_spine[z_cols_spine].values.flatten()[IGNORE_FIRST_N_FRAMES:], 
                           df_tdcr[z_cols_tdcr].values.flatten()[IGNORE_FIRST_N_FRAMES:]])

    x_min, x_max = np.nanmin(all_x), np.nanmax(all_x)
    y_min, y_max = np.nanmin(all_y), np.nanmax(all_y)
    z_min, z_max = np.nanmin(all_z), np.nanmax(all_z)

    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
    x_center = (x_max + x_min) / 2
    y_center = (y_max + y_min) / 2
    z_center = (z_max + z_min) / 2

    margin = 0
    x_lim = [x_center - max_range/2, x_center + max_range/2]
    y_lim = [y_center - max_range/2, y_center + max_range/2]
    z_lim = [z_center - max_range/2, z_center + max_range/2]
    
    # Update reference plane
    ref_frame_data = spine_data[IGNORE_FIRST_N_FRAMES]
    xs0 = ref_frame_data[:n_spine_points]
    ys0 = ref_frame_data[n_spine_points:2*n_spine_points]  
    zs0 = ref_frame_data[2*n_spine_points:3*n_spine_points]
    points0 = np.column_stack([xs0, ys0, zs0])
    ref_centroid, ref_normal, ref_e1, ref_e2 = fit_plane(points0)

# Initialize
update_global_limits()

x2d_lim = (-150, 150)
y2d_lim = (-150, 150)

def calculate_point_displacements(points_current):
    """Calculate displacement of each point from reference frame"""
    ref_frame_data = spine_data[IGNORE_FIRST_N_FRAMES]
    ref_xs = ref_frame_data[:n_spine_points]
    ref_ys = ref_frame_data[n_spine_points:2*n_spine_points]
    ref_zs = ref_frame_data[2*n_spine_points:3*n_spine_points]
    ref_points = np.column_stack([ref_xs, ref_ys, ref_zs])
    
    if len(ref_points) != len(points_current):
        return np.zeros(len(points_current))
    
    diffs = points_current - ref_points
    distances = np.linalg.norm(diffs, axis=1)
    return distances

spiral_fit_cache = {}

def fit_log_spiral_explicit_cached(frame_idx):
    cache_key = (frame_idx, IGNORE_FIRST_N_FRAMES, IGNORE_FIRST_X_DISPLACEMENT)
    if cache_key in spiral_fit_cache:
        return spiral_fit_cache[cache_key]
    
    actual_idx = frame_idx + IGNORE_FIRST_N_FRAMES
    
    # Extract data using pre-converted arrays
    frame_data = spine_data[actual_idx]
    xs_spine = frame_data[:n_spine_points]
    ys_spine = frame_data[n_spine_points:2*n_spine_points]
    zs_spine = frame_data[2*n_spine_points:3*n_spine_points]
    points_spine = np.column_stack([xs_spine, ys_spine, zs_spine])
    
    spine_2d = project_to_plane(points_spine, ref_centroid, ref_e1, ref_e2)
    x = spine_2d[:, 0]
    y = spine_2d[:, 1]
    
    # Use ultra-fast C++ fitting
    spiral = spiral_optimizer.fit_spiral_fast(x, y)
    
    if spiral.valid:
        fit = (spiral.x0, spiral.y0, spiral.a, spiral.b, spiral.theta_off)
    else:
        fit = (np.nan, np.nan, np.nan, np.nan, np.nan)
    
    spiral_fit_cache[cache_key] = fit
    return fit

# Create figure with adjusted positioning  
fig = plt.figure(figsize=(12, 9))
main_ax = fig.add_subplot(111, projection='3d')

# Adjusted positioning for more sliders
plt.subplots_adjust(left=0.05, right=0.82, bottom=0.35, top=0.85)
error_ax = fig.add_axes([0.1, 0.25, 0.7, 0.55])
error_ax.set_visible(False)

# Slider positions
frame_slider_ax = fig.add_axes([0.15, 0.18, 0.4, 0.02])
n_frames_slider_ax = fig.add_axes([0.15, 0.14, 0.4, 0.02])
displacement_slider_ax = fig.add_axes([0.15, 0.10, 0.4, 0.02])

# Create sliders
frame_slider = Slider(frame_slider_ax, 'Frame', 1, len(df_spine) - IGNORE_FIRST_N_FRAMES, 
                     valinit=1, valstep=1)
n_frames_slider = Slider(n_frames_slider_ax, 'Skip N Frames', 0, 1500, 
                        valinit=IGNORE_FIRST_N_FRAMES, valstep=50)
displacement_slider = Slider(displacement_slider_ax, 'Min Displacement', 0, 20, 
                           valinit=IGNORE_FIRST_X_DISPLACEMENT, valfmt='%.1f')

for slider in [frame_slider, n_frames_slider, displacement_slider]:
    slider.label.set_fontsize(12)

# Checkboxes
check_ax = plt.axes([0.83, 0.3, 0.15, 0.4])
check_labels = ['Spine', 'ROI_Points', '2D Mode', 'Show Spiral', 'Error Plot Mode', 'Show All Frames', 'Show Average']
check_vals = [True, True, False, True, False, False, False]
check = CheckButtons(check_ax, check_labels, check_vals)

for text in check.labels:
    text.set_fontsize(11)

legend_ax = plt.axes([0.83, 0.75, 0.15, 0.15])
legend_ax.axis('off')

spine_scatter = None
tdcr_scatter = None
spiral_line = None

def plot_row(row_idx, mode_2d, show_spiral, error_plot_mode, show_all_errors, show_average):
    global spine_scatter, tdcr_scatter, spiral_line, main_ax, error_ax, legend_ax
    
    legend_ax.clear()
    legend_ax.axis('off')
    main_ax.clear()
    error_ax.clear()
    error_ax.set_visible(error_plot_mode)
    main_ax.set_visible(not error_plot_mode)
    
    actual_idx = row_idx + IGNORE_FIRST_N_FRAMES
    
    # Fast data extraction
    frame_data = spine_data[actual_idx]
    xs_spine = frame_data[:n_spine_points]
    ys_spine = frame_data[n_spine_points:2*n_spine_points] 
    zs_spine = frame_data[2*n_spine_points:3*n_spine_points]
    points_spine = np.column_stack([xs_spine, ys_spine, zs_spine])
    
    # Apply displacement filtering
    spine_displacements = calculate_point_displacements(points_spine)
    low_displacement_mask = spine_displacements < IGNORE_FIRST_X_DISPLACEMENT
    
    # Get reference frame data
    ref_frame_data = spine_data[IGNORE_FIRST_N_FRAMES]
    ref_xs = ref_frame_data[:n_spine_points]
    ref_ys = ref_frame_data[n_spine_points:2*n_spine_points]
    ref_zs = ref_frame_data[2*n_spine_points:3*n_spine_points]
    
    xs_spine[low_displacement_mask] = ref_xs[low_displacement_mask]
    ys_spine[low_displacement_mask] = ref_ys[low_displacement_mask]
    zs_spine[low_displacement_mask] = ref_zs[low_displacement_mask]
    points_spine = np.column_stack([xs_spine, ys_spine, zs_spine])
    
    # TDCR data
    tdcr_frame_data = tdcr_data[actual_idx]
    xs_tdcr = tdcr_frame_data[:n_tdcr_points]
    ys_tdcr = tdcr_frame_data[n_tdcr_points:2*n_tdcr_points]
    zs_tdcr = tdcr_frame_data[2*n_tdcr_points:3*n_tdcr_points]
    points_tdcr = np.column_stack([xs_tdcr, ys_tdcr, zs_tdcr])
    
    status = check.get_status()
    
    if error_plot_mode:
        # Ultra-fast error processing with C++
        if show_average or show_all_errors:
            n_frames = len(df_spine) - IGNORE_FIRST_N_FRAMES
            sample_step = max(1, n_frames // 50)  # Process more frames - C++ is fast!
            
            all_errors = []
            max_points = 0
            
            for i in range(0, n_frames, sample_step):
                actual_i = i + IGNORE_FIRST_N_FRAMES
                if actual_i >= len(spine_data):
                    break
                    
                frame_data_i = spine_data[actual_i]
                xs_i = frame_data_i[:n_spine_points]
                ys_i = frame_data_i[n_spine_points:2*n_spine_points]
                zs_i = frame_data_i[2*n_spine_points:3*n_spine_points]
                points_i = np.column_stack([xs_i, ys_i, zs_i])
                
                spine_2d_i = project_to_plane(points_i, ref_centroid, ref_e1, ref_e2)
                x_i = spine_2d_i[:, 0]
                y_i = spine_2d_i[:, 1]
                
                spiral = spiral_optimizer.fit_spiral_fast(x_i, y_i)
                
                if spiral.valid:
                    # Use C++ function for error calculation
                    distances_i = spiral_optimizer.calculate_spiral_errors(x_i, y_i, spiral.x0, spiral.y0, 
                                                                         spiral.a, spiral.b, spiral.theta_off)
                    
                    filter_idx = int(IGNORE_FIRST_X_DISPLACEMENT)
                    if filter_idx < len(distances_i):
                        filtered_distances_i = distances_i[filter_idx:]
                        all_errors.append(filtered_distances_i)
                        max_points = max(max_points, len(filtered_distances_i))
            
            if all_errors and show_average:
                # Pad arrays and calculate average
                padded_errors = []
                for err in all_errors:
                    padded = np.full(max_points, np.nan)
                    padded[:len(err)] = err
                    padded_errors.append(padded)
                
                avg_errors = np.nanmean(padded_errors, axis=0)
                point_indices_avg = np.arange(int(IGNORE_FIRST_X_DISPLACEMENT), 
                                            int(IGNORE_FIRST_X_DISPLACEMENT) + len(avg_errors))
                error_ax.plot(point_indices_avg, avg_errors, 'r-', linewidth=4, label='Average Error')
            
            if show_all_errors:
                fixed_color = 'tab:blue'
                alpha_val = 0.3 if show_average else 1.0
                
                for err in all_errors:
                    point_indices = np.arange(int(IGNORE_FIRST_X_DISPLACEMENT), 
                                            int(IGNORE_FIRST_X_DISPLACEMENT) + len(err))
                    error_ax.plot(point_indices, err, color=fixed_color, alpha=alpha_val)

        # Plot single frame
        if not show_all_errors and not show_average:
            x0, y0, a, b, theta_off = fit_log_spiral_explicit_cached(row_idx)
            if not np.isnan(x0):
                spine_2d = project_to_plane(points_spine, ref_centroid, ref_e1, ref_e2)
                x = spine_2d[:, 0]
                y = spine_2d[:, 1]
                
                distances = spiral_optimizer.calculate_spiral_errors(x, y, x0, y0, a, b, theta_off)
                point_indices = np.arange(len(distances))
                filter_idx = int(IGNORE_FIRST_X_DISPLACEMENT)
                filtered_indices = point_indices[filter_idx:]
                filtered_distances = distances[filter_idx:]
                error_ax.plot(filtered_indices, filtered_distances, 'b-o', label='Distance from Fitted Curve')

        error_ax.set_xlabel('Point Index (shifted)', fontsize=14)
        error_ax.set_ylabel('Distance', fontsize=14)
        error_ax.grid(True)
        error_ax.set_ylim(0, 100)
        
        if show_average and show_all_errors:
            error_ax.set_title('Error Plot - All Frames + Average (C++ Ultra-Fast)', fontsize=15)
        elif show_average:
            error_ax.set_title('Average Error Plot (C++ Ultra-Fast)', fontsize=15)
        elif show_all_errors:
            error_ax.set_title('Error Plot - All Frames (C++ Ultra-Fast)', fontsize=15)
        else:
            error_ax.set_title(f'Error Plot - Frame {actual_idx + 1}', fontsize=15)
        
        if show_average or (not show_all_errors and not show_average):
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
                if not np.isnan(x0):
                    theta_data = np.arctan2(y - y0, x - x0)
                    # Simple unwrap for display
                    for i in range(1, len(theta_data)):
                        while theta_data[i] - theta_data[i-1] > np.pi:
                            theta_data[i] -= 2*np.pi
                        while theta_data[i] - theta_data[i-1] < -np.pi:
                            theta_data[i] += 2*np.pi
                    
                    theta_min, theta_max = np.min(theta_data), np.max(theta_data)
                    theta_fit = np.linspace(theta_min, theta_max, 200)
                    r_fit = a * np.exp(np.clip(b * (theta_fit + theta_off), -20, 20))
                    x_fit = x0 + r_fit * np.cos(theta_fit)
                    y_fit = y0 + r_fit * np.sin(theta_fit)
                    spiral_line = main_ax.plot(x_fit, y_fit, 'r-', lw=3, label='Fitted Spiral (C++ Ultra-Fast)')
            
            main_ax.set_xlabel('Plane X', fontsize=14)
            main_ax.set_ylabel('Plane Y', fontsize=14)
            title = '2D Projection - C++ Ultra-Fast Processing'
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

def update_frame_slider_range():
    """Update frame slider range when N frames changes"""
    new_max = len(df_spine) - IGNORE_FIRST_N_FRAMES
    if new_max > 0:
        frame_slider.valmax = new_max
        frame_slider.ax.set_xlim(1, new_max)
        if frame_slider.val > new_max:
            frame_slider.set_val(1)

def update_parameters(val):
    global IGNORE_FIRST_N_FRAMES, IGNORE_FIRST_X_DISPLACEMENT, spiral_fit_cache
    
    # Update global parameters
    old_n_frames = IGNORE_FIRST_N_FRAMES
    old_displacement = IGNORE_FIRST_X_DISPLACEMENT
    
    IGNORE_FIRST_N_FRAMES = int(n_frames_slider.val)
    IGNORE_FIRST_X_DISPLACEMENT = displacement_slider.val
    
    # Clear cache if parameters changed
    if old_n_frames != IGNORE_FIRST_N_FRAMES or old_displacement != IGNORE_FIRST_X_DISPLACEMENT:
        spiral_fit_cache.clear()
        update_global_limits()
        update_frame_slider_range()
    
    # Update plot
    status = check.get_status()
    mode_2d = status[2]
    show_spiral = status[3]
    error_plot_mode = status[4]
    show_all_errors = status[5]
    show_average = status[6]
    
    # Hide frame slider when showing average or all errors
    frame_slider_ax.set_visible(not (error_plot_mode and (show_all_errors or show_average)))
    
    if error_plot_mode and (show_all_errors or show_average):
        plot_row(0, mode_2d, show_spiral, error_plot_mode, show_all_errors, show_average)
    else:
        current_frame = max(1, min(int(frame_slider.val), len(df_spine) - IGNORE_FIRST_N_FRAMES))
        plot_row(current_frame - 1, mode_2d, show_spiral, error_plot_mode, show_all_errors, show_average)

def toggle_visibility(label):
    update_parameters(None)

# Connect sliders and checkboxes
frame_slider.on_changed(update_parameters)
n_frames_slider.on_changed(update_parameters)
displacement_slider.on_changed(update_parameters)
check.on_clicked(toggle_visibility)

print("🚀🚀🚀 C++ Ultra-Fast Version Loading...")
print("⚡ Expected performance: 100-1000x faster than original!")
print("🔥 Machine code speed spiral fitting")
print("💨 Real-time error plot generation")
print("✨ Professional grade performance")

# Initial plot
plot_row(0, False, True, False, False, False)

plt.show()
