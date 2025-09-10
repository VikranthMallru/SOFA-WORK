import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.lines as mlines
from matplotlib import cm
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['cmr10']
plt.rcParams["axes.formatter.use_mathtext"] = True
# ------- CONFIG: Edit these to control frame skipping -------
IGNORE_FIRST_N_FRAMES = 0        
IGNORE_FIRST_X_DISPLACEMENT = 0.0   
EVERY_N_FRAMES = 1 #min 1
REFERENCE_FRAME = 0  # New: Fixed frame for plane fitting (e.g., 0 for initial state)
# ----------------------------------------------------------
# Load data
# Using file names from circle fit code
df_spine = pd.read_csv('tdcr_spine.csv')
df_tdcr = pd.read_csv('tdcr_output.csv')
# Column extraction
x_cols_spine = [col for col in df_spine.columns if col.startswith('x')]
y_cols_spine = [col for col in df_spine.columns if col.startswith('y')]
z_cols_spine = [col for col in df_spine.columns if col.startswith('z')]
x_cols_tdcr = [col for col in df_tdcr.columns if col.startswith('x')]
y_cols_tdcr = [col for col in df_tdcr.columns if col.startswith('y')]
z_cols_tdcr = [col for col in df_tdcr.columns if col.startswith('z')]
# Global axis limits
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
margin = 0.1
x_lim = [x_center - max_range/2 - margin*max_range, x_center + max_range/2 + margin*max_range]
y_lim = [y_center - max_range/2 - margin*max_range, y_center + max_range/2 + margin*max_range]
z_lim = [z_center - max_range/2 - margin*max_range, z_center + max_range/2 + margin*max_range]
# Plane fitting utilities from 1st code
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
# Circle fitting function from 2nd code
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
def generate_circle_points(center, radius, e1, e2, angle_start, angle_end, num_points=100):
    angles = np.linspace(angle_start, angle_end, num_points)
    circle_points = np.array([center + radius*np.cos(a)*e1 + radius*np.sin(a)*e2 for a in angles])
    return circle_points
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
def should_consider_frame(frame_idx):
    return (frame_idx % EVERY_N_FRAMES) == 0
# New rotation helper function
def rotate_2d(points_2d, reference_vector):
    """Rotate 2D points so that reference_vector points along positive x-axis."""
    if np.linalg.norm(reference_vector) == 0:
        return points_2d  # No rotation if zero vector
    angle = np.arctan2(reference_vector[1], reference_vector[0])
    rot_matrix = np.array([[np.cos(-angle), -np.sin(-angle)],
                           [np.sin(-angle), np.cos(-angle)]])
    return np.dot(points_2d, rot_matrix.T)
# New: Additional rotation function
def additional_rotate(points_2d, angle):
    """Apply an additional rotation by the given angle (in radians)."""
    rot_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                           [np.sin(angle), np.cos(angle)]])
    return np.dot(points_2d, rot_matrix.T)
# Cache for circle fitting results
circle_fit_cache = {}
# Precompute circle fits for frames
n_frames = len(df_spine) - IGNORE_FIRST_N_FRAMES
considered_frames = [i for i in range(n_frames) if should_consider_frame(i)]
for i in considered_frames:
    actual_i = i + IGNORE_FIRST_N_FRAMES
    points_spine = np.vstack([
        df_spine.loc[actual_i, x_cols_spine].values,
        df_spine.loc[actual_i, y_cols_spine].values,
        df_spine.loc[actual_i, z_cols_spine].values
    ]).T
    if len(points_spine) >= 3:
        center, radius, normal, e1, e2 = fit_circle_3d(points_spine[0], points_spine[len(points_spine)//2], points_spine[-1])
    else:
        center, radius, normal, e1, e2 = None, np.inf, None, None, None
    l1 = df_spine.loc[actual_i, 'L1'] if 'L1' in df_spine.columns else 0.0  # Assuming 'L1' column exists
    circle_fit_cache[actual_i] = (center, radius, normal, e1, e2, l1)
# Define plotting variables
fig = plt.figure(figsize=(10, 8))
main_ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(left=0.15, right=0.80, bottom=0.25, top=0.85)
# Axes for deviation plot and legend
deviation_ax = fig.add_axes([0.15, 0.15, 0.65, 0.65])
deviation_ax.set_visible(False)
legend_ax = plt.axes([0.85, 0.85, 0.13, 0.15])
legend_ax.axis('off')
# Slider
slider_ax = fig.add_axes([0.2, 0.08, 0.5, 0.03])
slider = Slider(slider_ax, 'Frame Slider', 1, len(considered_frames), valinit=1, valstep=1)
slider.label.set_fontsize(16)
# Check buttons grouped
# Group 1: Display
display_ax = plt.axes([0.85, 0.65, 0.13, 0.15])
display_labels = ['Spine', 'ROI_Points', 'Show Circle']
display_vals = [True, True, True]
display_check = CheckButtons(display_ax, display_labels, display_vals)
for text in display_check.labels:
    text.set_fontsize(12)
fig.text(0.85, 0.80, 'Display', fontsize=14, fontweight='bold')
# Group 2: Modes
modes_ax = plt.axes([0.85, 0.50, 0.13, 0.10])
modes_labels = ['2D Mode', 'Deviation Plot Mode']
modes_vals = [False, False]
modes_check = CheckButtons(modes_ax, modes_labels, modes_vals)
for text in modes_check.labels:
    text.set_fontsize(12)
fig.text(0.85, 0.60, 'Modes', fontsize=14, fontweight='bold')
# Group 3: Multi-frame
multi_ax = plt.axes([0.85, 0.35, 0.13, 0.15])
multi_labels = ['Show All Frames', 'Show All in 2D', 'Show Average']
multi_vals = [False, False, False]
multi_check = CheckButtons(multi_ax, multi_labels, multi_vals)
for text in multi_check.labels:
    text.set_fontsize(12)
fig.text(0.85, 0.50, 'Multi-Frame', fontsize=14, fontweight='bold')
# Group 4: 2D Options
options_ax = plt.axes([0.85, 0.25, 0.13, 0.05])
options_labels = ['Transform 2D']
options_vals = [False]
options_check = CheckButtons(options_ax, options_labels, options_vals)
for text in options_check.labels:
    text.set_fontsize(12)
fig.text(0.85, 0.30, '2D Options', fontsize=14, fontweight='bold')
# Globals for scatter and plots
spine_scatter = None
tdcr_scatter = None
circle_line = None
line_fit = None
spiral_line = None
# Helper function to get angle range for arc
def get_angle_range(center, e1, e2, points_spine):
    start_point = points_spine[0]
    end_point = points_spine[-1]
    start_2d = np.array([np.dot(start_point - center, e1), np.dot(start_point - center, e2)])
    end_2d = np.array([np.dot(end_point - center, e1), np.dot(end_point - center, e2)])
    angle_start = np.arctan2(start_2d[1], start_2d[0])
    angle_end = np.arctan2(end_2d[1], end_2d[0])
    # Ensure angle_end > angle_start
    if angle_end < angle_start:
        angle_end += 2 * np.pi
    return angle_start, angle_end
# Main plotting function
def plot_row(considered_idx, mode_2d, deviation_plot_mode, show_all_deviations, show_average, show_all_2d, show_transform, show_spiral):
    global spine_scatter, tdcr_scatter, circle_line, line_fit, main_ax, deviation_ax, legend_ax
    legend_ax.clear()
    legend_ax.axis('off')
    main_ax.clear()
    deviation_ax.clear()
    deviation_ax.set_visible(deviation_plot_mode)
    main_ax.set_visible(not deviation_plot_mode)
    if considered_idx < len(considered_frames):
        row_idx = considered_frames[considered_idx]
    else:
        row_idx = 0
    actual_idx = row_idx + IGNORE_FIRST_N_FRAMES
    xs_spine = df_spine.loc[actual_idx, x_cols_spine].values
    ys_spine = df_spine.loc[actual_idx, y_cols_spine].values
    zs_spine = df_spine.loc[actual_idx, z_cols_spine].values
    points_spine = np.vstack([xs_spine, ys_spine, zs_spine]).T
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
    # Get circle fit cached
    center, radius, normal, e1, e2, _ = circle_fit_cache[actual_idx]
    if deviation_plot_mode:
        legend_handles = []
        # Calculate deviations
        if show_average or show_all_deviations:
            all_deviations = []
            max_points = 0
            for i in considered_frames:
                idx = i + IGNORE_FIRST_N_FRAMES
                c, r, n, v1, v2, _ = circle_fit_cache[idx]
                points = np.vstack([
                    df_spine.loc[idx, x_cols_spine].values,
                    df_spine.loc[idx, y_cols_spine].values,
                    df_spine.loc[idx, z_cols_spine].values
                ]).T
                if c is None or r == np.inf:
                    continue
                # Project points to the fitted plane
                points_2d = project_to_plane(points, c, v1, v2)
                dists_3d = np.linalg.norm(points - c, axis=1)
                deviations = np.abs(dists_3d - r)
                all_deviations.append(deviations)
                max_points = max(max_points, len(deviations))
            # Padding for averaging
            padded_devs = []
            for dev in all_deviations:
                padded = np.full(max_points, np.nan)
                padded[:len(dev)] = dev
                padded_devs.append(padded)
            avg_devs = np.nanmean(padded_devs, axis=0)
            points_idx = np.arange(max_points)
        # Plot all individual deviation lines
        if show_all_deviations:
            num_frames = len(considered_frames)
            colors = cm.viridis(np.linspace(0, 1, num_frames))
            alpha_val = 0.3 if show_average else 1.0
            for j, i in enumerate(considered_frames):
                idx = i + IGNORE_FIRST_N_FRAMES
                c, r, n, v1, v2, _ = circle_fit_cache[idx]
                points = np.vstack([
                    df_spine.loc[idx, x_cols_spine].values,
                    df_spine.loc[idx, y_cols_spine].values,
                    df_spine.loc[idx, z_cols_spine].values
                ]).T
                if c is None or r == np.inf:
                    continue
                points_2d = project_to_plane(points, c, v1, v2)
                dists_3d = np.linalg.norm(points - c, axis=1)
                deviations = np.abs(dists_3d - r)
                x_vals = np.arange(len(deviations))
                deviation_ax.plot(x_vals, deviations, color=colors[j], alpha=alpha_val, linewidth=2)
        # Plot average deviation
        if show_average:
            avg_color = 'r' if show_all_deviations else 'tab:blue'
            avg_line, = deviation_ax.plot(points_idx, avg_devs, avg_color + '-', linewidth=5, label='Average Deviation')
            legend_handles.append(avg_line)
        # Plot single frame deviation line
        if not show_all_deviations and not show_average:
            if center is not None and radius != np.inf:
                dists_3d_single = np.linalg.norm(points_spine - center, axis=1)
                deviations_single = np.abs(dists_3d_single - radius)
                x_vals = np.arange(len(deviations_single))
                single_line, = deviation_ax.plot(x_vals, deviations_single, 'b-o', linewidth=2, label=f'Frame {actual_idx+1}')
                legend_handles.append(single_line)
        # Add min/mid/max disp to legend if show_all_deviations
        if show_all_deviations:
            l1_values = np.array([circle_fit_cache[i + IGNORE_FIRST_N_FRAMES][5] for i in considered_frames])
            min_idx = np.argmin(l1_values)
            mid_idx = len(l1_values) // 2
            max_idx = np.argmax(l1_values)
            low = mlines.Line2D([], [], color=colors[min_idx], linewidth=2, label=f'Min Disp: {l1_values[min_idx]:.2f}')
            mid = mlines.Line2D([], [], color=colors[mid_idx], linewidth=2, label=f'Mid Disp: {l1_values[mid_idx]:.2f}')
            high = mlines.Line2D([], [], color=colors[max_idx], linewidth=2, label=f'Max Disp: {l1_values[max_idx]:.2f}')
            legend_handles = [low, mid, high]
        deviation_ax.set_xlabel('Point Index', fontsize=24)
        deviation_ax.set_ylabel('Deviation (mm)', fontsize=24)
        deviation_ax.tick_params(axis='both', which='major', labelsize=24)
        deviation_ax.tick_params(axis='both', which='minor', labelsize=18)
        deviation_ax.grid(True)
        deviation_ax.set_ylim(0, 5)
        if show_average and show_all_deviations:
            deviation_ax.set_title('Deviation Plot', fontsize=20)
        elif show_average:
            deviation_ax.set_title('Average Deviation Plot - Considered Frames', fontsize=20)
        elif show_all_deviations:
            deviation_ax.set_title('Deviation Plot - All Considered Frames', fontsize=20)
        else:
            deviation_ax.set_title(f'Deviation Plot - Frame {actual_idx + 1}', fontsize=20)
        if legend_handles:
            deviation_ax.legend(handles=legend_handles, fontsize=12, loc='upper left')
    else:
        # Plot 2D or 3D visualization
        if mode_2d:
            if main_ax.name == '3d':
                fig.delaxes(main_ax)
                main_ax = fig.add_subplot(111)
            legend_handles = []
            legend_labels = []
            if show_all_2d:
                num_frames = len(considered_frames)
                colors = cm.viridis(np.linspace(0, 1, num_frames))
                alpha_val = 0.3
                for j, i in enumerate(considered_frames):
                    idx = i + IGNORE_FIRST_N_FRAMES
                    pts_sp = np.vstack([
                        df_spine.loc[idx, x_cols_spine].values,
                        df_spine.loc[idx, y_cols_spine].values,
                        df_spine.loc[idx, z_cols_spine].values
                    ]).T
                    pts_td = np.vstack([
                        df_tdcr.loc[idx, x_cols_tdcr].values,
                        df_tdcr.loc[idx, y_cols_tdcr].values,
                        df_tdcr.loc[idx, z_cols_tdcr].values
                    ]).T
                    c, r, n, v1, v2, _ = circle_fit_cache[idx]
                    if c is None or r == np.inf:
                        continue
                    sp_2d = project_to_plane(pts_sp, c, v1, v2)
                    td_2d = project_to_plane(pts_td, c, v1, v2)
                    # Align base to origin
                    base_2d = sp_2d[0].copy()
                    sp_2d -= base_2d
                    td_2d -= base_2d
                    # New: Rotational alignment using tip vector
                    tip_vector = sp_2d[-1]  # Vector from base to tip
                    sp_2d = rotate_2d(sp_2d, tip_vector)
                    td_2d = rotate_2d(td_2d, tip_vector)
                    # New: Apply rotation offset for each frame to make it look like bending forward
                    offset_angle = (j / (num_frames - 1)) * (np.pi / 2) if num_frames > 1 else 0  # Total 90 degrees fan
                    sp_2d = additional_rotate(sp_2d, offset_angle)
                    td_2d = additional_rotate(td_2d, offset_angle)
                    color = colors[j]
                    if show_transform:
                        sp_x, sp_y = sp_2d[:, 1], sp_2d[:, 0]
                        td_x, td_y = td_2d[:, 1], td_2d[:, 0]
                    else:
                        sp_x, sp_y = sp_2d[:, 0], sp_2d[:, 1]
                        td_x, td_y = td_2d[:, 0], td_2d[:, 1]
                    if display_check.get_status()[0]:
                        main_ax.scatter(sp_x, sp_y, c=[color], marker='o', s=40, alpha=alpha_val)
                    if display_check.get_status()[1]:
                        main_ax.scatter(td_x, td_y, c=[color], marker='o', s=40, alpha=alpha_val)
                    if show_spiral:
                        angle_start, angle_end = get_angle_range(c, v1, v2, pts_sp)
                        angles = np.linspace(angle_start, angle_end, 200)
                        x_fit_plane = r * np.cos(angles)
                        y_fit_plane = r * np.sin(angles)
                        fit_points = np.stack([x_fit_plane, y_fit_plane], axis=1)
                        fit_points -= base_2d  # Align base
                        fit_points = rotate_2d(fit_points, tip_vector)  # New: Rotate arc
                        fit_points = additional_rotate(fit_points, offset_angle)  # Apply offset
                        if show_transform:
                            x_fit_plot = fit_points[:, 1]
                            y_fit_plot = fit_points[:, 0]
                        else:
                            x_fit_plot = fit_points[:, 0]
                            y_fit_plot = fit_points[:, 1]
                        main_ax.plot(x_fit_plot, y_fit_plot, color=color, lw=2, alpha=alpha_val)
                # Legend for min/mid/max tendon displacement
                l1_values = np.array([circle_fit_cache[i + IGNORE_FIRST_N_FRAMES][5] for i in considered_frames])
                min_idx = np.argmin(l1_values)
                mid_idx = len(l1_values) // 2
                max_idx = np.argmax(l1_values)
                low = mlines.Line2D([], [], color=colors[min_idx], linewidth=2, label=f'Min Disp: {l1_values[min_idx]:.2f}')
                mid = mlines.Line2D([], [], color=colors[mid_idx], linewidth=2, label=f'Mid Disp: {l1_values[mid_idx]:.2f}')
                high = mlines.Line2D([], [], color=colors[max_idx], linewidth=2, label=f'Max Disp: {l1_values[max_idx]:.2f}')
                main_ax.legend(handles=[low, mid, high], loc='upper left', fontsize=12, title='Tendon Displacement')
                main_ax.set_xlabel('X', fontsize=20)
                main_ax.set_ylabel('Y', fontsize=20)
                main_ax.tick_params(axis='both', which='major', labelsize=24)
                main_ax.tick_params(axis='both', which='minor', labelsize=18)
                main_ax.set_title('2D Projection and Curve Fitting', fontsize=15)
            else:
                # Single frame 2D plot
                if center is not None and radius != np.inf:
                    sp_2d = project_to_plane(points_spine, center, e1, e2)
                    td_2d = project_to_plane(points_tdcr, center, e1, e2)
                    if show_transform:
                        sp_x, sp_y = sp_2d[:, 1], sp_2d[:, 0]
                        td_x, td_y = td_2d[:, 1], td_2d[:, 0]
                    else:
                        sp_x, sp_y = sp_2d[:, 0], sp_2d[:, 1]
                        td_x, td_y = td_2d[:, 0], td_2d[:, 1]
                    if display_check.get_status()[0]:
                        spine_scatter = main_ax.scatter(sp_x, sp_y, c='b', marker='o', s=40, label='Spine Points')
                    if display_check.get_status()[1]:
                        tdcr_scatter = main_ax.scatter(td_x, td_y, c='g', marker='o', s=40, label='ROI Points')
                    if show_spiral:
                        angle_start, angle_end = get_angle_range(center, e1, e2, points_spine)
                        angles = np.linspace(angle_start, angle_end, 200)
                        x_fit_plane = radius * np.cos(angles)
                        y_fit_plane = radius * np.sin(angles)
                        if show_transform:
                            x_fit_plot = y_fit_plane
                            y_fit_plot = x_fit_plane
                        else:
                            x_fit_plot = x_fit_plane
                            y_fit_plot = y_fit_plane
                        spiral_line = main_ax.plot(x_fit_plot, y_fit_plot, 'r-', lw=2, label='Fitted Circle')
                    main_ax.set_xlabel('X', fontsize=20)
                    main_ax.set_ylabel('Y', fontsize=20)
                    main_ax.tick_params(axis='both', which='major', labelsize=24)
                    main_ax.tick_params(axis='both', which='minor', labelsize=18)
                    main_ax.set_title('2D Projection onto Reference Plane', fontsize=15)
            main_ax.axis('equal')
            main_ax.grid(True)
        else:
            # 3D mode
            if main_ax.name != '3d':
                fig.delaxes(main_ax)
                main_ax = fig.add_subplot(111, projection='3d')
            legend_handles = []
            if display_check.get_status()[0]:
                spine_scatter = main_ax.scatter(xs_spine, ys_spine, zs_spine, c='b', marker='o', s=40, label='Spine Points')
                legend_handles.append(spine_scatter)
            if display_check.get_status()[1]:
                tdcr_scatter = main_ax.scatter(xs_tdcr, ys_tdcr, zs_tdcr, c='g', marker='o', s=40, label='ROI Points')
                legend_handles.append(tdcr_scatter)
            circle_line = None
            line_fit = None
            large_threshold = 1e6
            if radius == np.inf or radius > large_threshold or center is None:
                line_fit, = main_ax.plot([points_spine[0,0], points_spine[-1,0]],
                                         [points_spine[0,1], points_spine[-1,1]],
                                         [points_spine[0,2], points_spine[-1,2]],
                                         'r-', linewidth=2, label='Fitted Line')
                legend_handles.append(line_fit)
            else:
                angle_start, angle_end = get_angle_range(center, e1, e2, points_spine)
                circle_points = generate_circle_points(center, radius, e1, e2, angle_start, angle_end)
                segments = []
                for i in range(len(circle_points) - 1):
                    p1 = circle_points[i]
                    p2 = circle_points[i+1]
                    if (x_lim[0] <= p1[0] <= x_lim[1] and x_lim[0] <= p2[0] <= x_lim[1] and
                        y_lim[0] <= p1[1] <= y_lim[1] and y_lim[0] <= p2[1] <= y_lim[1] and
                        z_lim[0] <= p1[2] <= z_lim[1] and z_lim[0] <= p2[2] <= z_lim[1]):
                        segments.append([p1, p2])
                circle_line = Line3DCollection(segments, colors='r', linewidths=1.5, alpha=1.0)
                if display_check.get_status()[2] and display_check.get_status()[0]:
                    main_ax.add_collection3d(circle_line)
            if display_check.get_status()[2] and display_check.get_status()[0]:
                legend_handles.append(mlines.Line2D([], [], color='r', lw=5, label='Fitted Circle'))
            main_ax.set_xlabel('X', fontsize=14, labelpad=20)
            main_ax.set_ylabel('Y', fontsize=14, labelpad=20)
            main_ax.set_zlabel('Z', fontsize=14, labelpad=20)
            main_ax.tick_params(axis='both', which='major', labelsize=20)
            main_ax.tick_params(axis='both', which='minor', labelsize=14)
            radius_or_len = radius if radius != np.inf else np.linalg.norm(points_spine[-1] - points_spine[0])
            title_str = f"Radius: {radius_or_len:.4f} mm"
            main_ax.set_title(title_str, fontsize=16)
            main_ax.set_xlim(x_lim)
            main_ax.set_ylim(y_lim)
            main_ax.set_zlim(z_lim)
            try:
                main_ax.set_box_aspect([1,1,1])
            except Exception:
                pass
        if legend_handles:
            legend_ax.legend(legend_handles, [h.get_label() for h in legend_handles], loc='upper left', fontsize=13, frameon=True)
    plt.draw()
# Update function
def update(val):
    display_status = display_check.get_status()
    modes_status = modes_check.get_status()
    multi_status = multi_check.get_status()
    options_status = options_check.get_status()
    mode_2d = modes_status[0]
    deviation_plot_mode = modes_status[1]
    show_all_deviations = multi_status[0]
    show_all_2d = multi_status[1]
    show_average = multi_status[2]
    show_transform = options_status[0]
    show_spiral = display_status[2]
    hide_slider = deviation_plot_mode and (show_all_deviations or show_average) or (mode_2d and show_all_2d and not deviation_plot_mode)
    slider_ax.set_visible(not hide_slider)
    if hide_slider:
        plot_row(0, mode_2d, deviation_plot_mode, show_all_deviations, show_average, show_all_2d, show_transform, show_spiral)
    else:
        plot_row(int(slider.val) - 1, mode_2d, deviation_plot_mode, show_all_deviations, show_average, show_all_2d, show_transform, show_spiral)
def toggle_visibility(label):
    update(None)
display_check.on_clicked(toggle_visibility)
modes_check.on_clicked(toggle_visibility)
multi_check.on_clicked(toggle_visibility)
options_check.on_clicked(toggle_visibility)
slider.on_changed(update)
plot_row(0, False, False, False, False, False, False, True)
plt.show()
