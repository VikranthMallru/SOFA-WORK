# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.widgets import Slider, CheckButtons
# from scipy.optimize import minimize, Bounds, differential_evolution

# # ------- CONFIG: Edit these to control frame skipping -------
# IGNORE_FIRST_N_FRAMES = 750        
# IGNORE_FIRST_X_DISPLACEMENT = 6.0   

# # ------- VISUAL: Use Computer Modern (LaTeX) fonts -------
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['cmr10']
# plt.rcParams["axes.formatter.use_mathtext"] = True

# # --- Load data ---
# df_spine = pd.read_csv('tdcr_trunk_spine.csv')
# df_tdcr = pd.read_csv('tdcr_trunk_output.csv')

# x_cols_spine = [col for col in df_spine.columns if col.startswith('x')]
# y_cols_spine = [col for col in df_spine.columns if col.startswith('y')]
# z_cols_spine = [col for col in df_spine.columns if col.startswith('z')]

# x_cols_tdcr = [col for col in df_tdcr.columns if col.startswith('x')]
# y_cols_tdcr = [col for col in df_tdcr.columns if col.startswith('y')]
# z_cols_tdcr = [col for col in df_tdcr.columns if col.startswith('z')]

# # ------- VISUAL: Consistent global limits with small margin (considering only non-ignored frames) -------
# all_x = np.concatenate([df_spine[x_cols_spine].values.flatten()[IGNORE_FIRST_N_FRAMES:], df_tdcr[x_cols_tdcr].values.flatten()[IGNORE_FIRST_N_FRAMES:]])
# all_y = np.concatenate([df_spine[y_cols_spine].values.flatten()[IGNORE_FIRST_N_FRAMES:], df_tdcr[y_cols_tdcr].values.flatten()[IGNORE_FIRST_N_FRAMES:]])
# all_z = np.concatenate([df_spine[z_cols_spine].values.flatten()[IGNORE_FIRST_N_FRAMES:], df_tdcr[z_cols_tdcr].values.flatten()[IGNORE_FIRST_N_FRAMES:]])

# x_min, x_max = np.nanmin(all_x), np.nanmax(all_x)
# y_min, y_max = np.nanmin(all_y), np.nanmax(all_y)
# z_min, z_max = np.nanmin(all_z), np.nanmax(all_z)

# max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
# x_center = (x_max + x_min) / 2
# y_center = (y_max + y_min) / 2
# z_center = (z_max + z_min) / 2

# margin = 0
# x_lim = [x_center - max_range/2 - margin*max_range, x_center + max_range/2 + margin*max_range]
# y_lim = [y_center - max_range/2 - margin*max_range, y_center + max_range/2 + margin*max_range]
# z_lim = [z_center - max_range/2 - margin*max_range, z_center + max_range/2 + margin*max_range]

# def get_axis_limits():
#     return x_lim, y_lim, z_lim

# def fit_plane(points):
#     points = points[~np.isnan(points).any(axis=1)]
#     if len(points) < 3:
#         return np.mean(points, axis=0), np.array([0, 0, 1]), np.array([1, 0, 0]), np.array([0, 1, 0])
    
#     centroid = np.mean(points, axis=0)
#     centered = points - centroid
    
#     if np.allclose(centered.std(axis=0), 0):
#         return centroid, np.array([0, 0, 1]), np.array([1, 0, 0]), np.array([0, 1, 0])
    
#     centered += np.random.normal(0, 1e-10, centered.shape)
#     _, _, vh = np.linalg.svd(centered)
#     normal = vh[2, :]
#     e1 = vh[0, :]
#     e2 = vh[1, :]
    
#     return centroid, normal, e1, e2

# def project_to_plane(points, origin, e1, e2):
#     rel = points - origin
#     x = np.dot(rel, e1)
#     y = np.dot(rel, e2)
#     return np.stack([x, y], axis=1)

# xs0 = df_spine.loc[IGNORE_FIRST_N_FRAMES, x_cols_spine].values
# ys0 = df_spine.loc[IGNORE_FIRST_N_FRAMES, y_cols_spine].values
# zs0 = df_spine.loc[IGNORE_FIRST_N_FRAMES, z_cols_spine].values
# points0 = np.vstack([xs0, ys0, zs0]).T

# ref_centroid, ref_normal, ref_e1, ref_e2 = fit_plane(points0)

# x2d_lim = (-150, 150)
# y2d_lim = (-150, 150)

# def safe_exp(x):
#     x = np.clip(x, -20, 20)
#     return np.exp(x)

# def calculate_point_displacements(points_current):
#     """Calculate displacement of each point from reference frame"""
#     ref_points = np.vstack([
#         df_spine.loc[IGNORE_FIRST_N_FRAMES, x_cols_spine].values,
#         df_spine.loc[IGNORE_FIRST_N_FRAMES, y_cols_spine].values,
#         df_spine.loc[IGNORE_FIRST_N_FRAMES, z_cols_spine].values]).T
    
#     if len(ref_points) != len(points_current):
#         return np.zeros(len(points_current))
    
#     diffs = points_current - ref_points
#     distances = np.linalg.norm(diffs, axis=1)
#     return distances

# spiral_fit_cache = {}

# def fit_log_spiral_explicit(x, y, maxiter=20000, use_global_opt=False, iter_refine=2):
#     x = np.asarray(x).ravel()
#     y = np.asarray(y).ravel()
#     mask = ~np.isnan(x) & ~np.isnan(y)
#     x = x[mask]
#     y = y[mask]
    
#     if len(x) < 5:
#         return np.nan, np.nan, np.nan, np.nan, np.nan
    
#     centroid = np.array([np.median(x), np.median(y)])
#     dists = np.sqrt((x - centroid[0])**2 + (y - centroid[1])**2)
#     keep_mask = dists < np.percentile(dists, 90)
#     x = x[keep_mask]
#     y = y[keep_mask]
    
#     if len(x) < 5:
#         return np.nan, np.nan, np.nan, np.nan, np.nan
    
#     x0_init = np.median(x)
#     y0_init = np.median(y)
#     theta = np.unwrap(np.arctan2(y - y0_init, x - x0_init))
#     r = np.sqrt((x - x0_init)**2 + (y - y0_init)**2)
#     weights = r / np.max(r)
#     p = np.polyfit(theta, np.log(np.maximum(r,1e-8)), 1, w=weights)
#     a0 = np.exp(p[1])
#     b0 = p[0]
#     theta0 = 0
    
#     params0 = np.array([x0_init, y0_init, np.log(a0), b0, theta0], dtype=float)
    
#     min_x, max_x = np.min(x), np.max(x)
#     min_y, max_y = np.min(y), np.max(y)
#     bounds = Bounds([min_x, min_y, -10, -2, -2*np.pi], [max_x, max_y, 10, 2, 2*np.pi])
    
#     def spiral_cost(params):
#         x0, y0, loga, b, theta_off = params
#         theta_data = np.unwrap(np.arctan2(y - y0, x - x0))
#         r_data = np.sqrt((x - x0)**2 + (y - y0)**2)
        
#         if np.all(r_data == 0):
#             return 1e10
        
#         model_log_r = loga + b*(theta_data + theta_off)
#         log_r_data = np.log(np.maximum(r_data,1e-8))
#         reg = 0.01*(b**2 + theta_off**2)
        
#         return np.sum((log_r_data - model_log_r)**2) + reg
    
#     if use_global_opt:
#         bounds_list = [(min_x,max_x),(min_y,max_y),(-10,10),(-2,2),(-2*np.pi,2*np.pi)]
#         res = differential_evolution(spiral_cost, bounds_list, maxiter=maxiter, tol=1e-10)
#     else:
#         res = minimize(spiral_cost, params0, method='L-BFGS-B', bounds=bounds, options={'maxiter':maxiter,'ftol':1e-10})
    
#     x0_fit, y0_fit, loga_fit, b_fit, theta_off_fit = res.x
#     a_fit = np.exp(np.clip(loga_fit,-20,20))
    
#     for _ in range(iter_refine):
#         theta_data = np.unwrap(np.arctan2(y - y0_fit, x - x0_fit))
#         r_data = np.sqrt((x - x0_fit)**2 + (y - y0_fit)**2)
#         r_model = a_fit * safe_exp(b_fit * (theta_data + theta_off_fit))
#         residuals = np.abs(r_data - r_model)
#         keep_mask = residuals < np.percentile(residuals, 80)
        
#         if np.sum(keep_mask) < 5:
#             break
        
#         x_refine = x[keep_mask]
#         y_refine = y[keep_mask]
        
#         x0_init = np.median(x_refine)
#         y0_init = np.median(y_refine)
#         theta = np.unwrap(np.arctan2(y_refine - y0_init, x_refine - x0_init))
#         r = np.sqrt((x_refine - x0_init)**2 + (y_refine - y0_init)**2)
#         p = np.polyfit(theta, np.log(np.maximum(r,1e-8)), 1, w=(r/np.max(r)))
#         a0 = np.exp(p[1])
#         b0 = p[0]
#         theta0 = 0
        
#         params0 = np.array([x0_init, y0_init, np.log(a0), b0, theta0], dtype=float)
        
#         if use_global_opt:
#             res = differential_evolution(spiral_cost, bounds_list, maxiter=maxiter, tol=1e-10)
#         else:
#             res = minimize(spiral_cost, params0, method='L-BFGS-B', bounds=bounds, options={'maxiter':maxiter,'ftol':1e-10})
        
#         x0_fit, y0_fit, loga_fit, b_fit, theta_off_fit = res.x
#         a_fit = np.exp(np.clip(loga_fit,-20,20))
    
#     return x0_fit, y0_fit, a_fit, b_fit, theta_off_fit

# def fit_log_spiral_explicit_cached(frame_idx):
#     actual_idx = frame_idx + IGNORE_FIRST_N_FRAMES
#     if actual_idx in spiral_fit_cache:
#         return spiral_fit_cache[actual_idx]
    
#     xs_spine = df_spine.loc[actual_idx, x_cols_spine].values
#     ys_spine = df_spine.loc[actual_idx, y_cols_spine].values
#     zs_spine = df_spine.loc[actual_idx, z_cols_spine].values
#     points_spine = np.vstack([xs_spine, ys_spine, zs_spine]).T
    
#     spine_2d = project_to_plane(points_spine, ref_centroid, ref_e1, ref_e2)
#     x = spine_2d[:, 0]
#     y = spine_2d[:, 1]
    
#     fit = fit_log_spiral_explicit(x, y)
#     spiral_fit_cache[actual_idx] = fit
#     return fit

# def calculate_displacement(points):
#     ref_points = np.vstack([
#         df_spine.loc[IGNORE_FIRST_N_FRAMES, x_cols_spine].values,
#         df_spine.loc[IGNORE_FIRST_N_FRAMES, y_cols_spine].values,
#         df_spine.loc[IGNORE_FIRST_N_FRAMES, z_cols_spine].values]).T
    
#     curr_points = points
    
#     if len(ref_points) != len(curr_points):
#         return 0
    
#     diffs = curr_points - ref_points
#     distances = np.linalg.norm(diffs, axis=1)
#     return np.max(distances)

# # Create figure with adjusted positioning
# fig = plt.figure(figsize=(10, 8))
# main_ax = fig.add_subplot(111, projection='3d')

# # Fixed UI positioning to prevent overlapping
# plt.subplots_adjust(left=0.05, right=0.85, bottom=0.25, top=0.85)  # Increased bottom margin
# error_ax = fig.add_axes([0.1, 0.15, 0.75, 0.65])  # Moved up and made smaller
# error_ax.set_visible(False)

# slider_ax = fig.add_axes([0.2, 0.08, 0.5, 0.03])  # Moved slider up
# slider = Slider(slider_ax, 'Frame Slider', 1, len(df_spine) - IGNORE_FIRST_N_FRAMES, valinit=1, valstep=1)
# slider.label.set_fontsize(16)

# check_ax = plt.axes([0.85, 0.3, 0.13, 0.4])
# # Added new "Show Average" toggle
# check_labels = ['Spine', 'ROI_Points', '2D Mode', 'Show Spiral', 'Error Plot Mode', 'Show All Frames', 'Show Average']
# check_vals = [True, True, False, True, False, False, False]
# check = CheckButtons(check_ax, check_labels, check_vals)

# for text in check.labels:
#     text.set_fontsize(12)

# legend_ax = plt.axes([0.85, 0.75, 0.13, 0.15])
# legend_ax.axis('off')

# spine_scatter = None
# tdcr_scatter = None
# spiral_line = None

# def plot_row(row_idx, mode_2d, show_spiral, error_plot_mode, show_all_errors, show_average):
#     global spine_scatter, tdcr_scatter, spiral_line, main_ax, error_ax, legend_ax
    
#     legend_ax.clear()
#     legend_ax.axis('off')
#     main_ax.clear()
#     error_ax.clear()
#     error_ax.set_visible(error_plot_mode)
#     main_ax.set_visible(not error_plot_mode)
    
#     actual_idx = row_idx + IGNORE_FIRST_N_FRAMES

#     xs_spine = df_spine.loc[actual_idx, x_cols_spine].values
#     ys_spine = df_spine.loc[actual_idx, y_cols_spine].values
#     zs_spine = df_spine.loc[actual_idx, z_cols_spine].values
#     points_spine = np.vstack([xs_spine, ys_spine, zs_spine]).T

#     # Apply displacement filtering - set coordinates to 0 if displacement < threshold
#     spine_displacements = calculate_point_displacements(points_spine)
#     low_displacement_mask = spine_displacements < IGNORE_FIRST_X_DISPLACEMENT
#     xs_spine[low_displacement_mask] = df_spine.loc[IGNORE_FIRST_N_FRAMES, x_cols_spine].values[low_displacement_mask]
#     ys_spine[low_displacement_mask] = df_spine.loc[IGNORE_FIRST_N_FRAMES, y_cols_spine].values[low_displacement_mask]
#     zs_spine[low_displacement_mask] = df_spine.loc[IGNORE_FIRST_N_FRAMES, z_cols_spine].values[low_displacement_mask]
#     points_spine = np.vstack([xs_spine, ys_spine, zs_spine]).T

#     xs_tdcr = df_tdcr.loc[actual_idx, x_cols_tdcr].values
#     ys_tdcr = df_tdcr.loc[actual_idx, y_cols_tdcr].values
#     zs_tdcr = df_tdcr.loc[actual_idx, z_cols_tdcr].values
#     points_tdcr = np.vstack([xs_tdcr, ys_tdcr, zs_tdcr]).T

#     status = check.get_status()
    
#     if error_plot_mode:
#         # Calculate average data if needed
#         if show_average:
#             n_frames = len(df_spine) - IGNORE_FIRST_N_FRAMES
#             all_errors = []
#             max_points = 0
            
#             for i in range(n_frames):
#                 actual_i = i + IGNORE_FIRST_N_FRAMES
#                 x0_i, y0_i, a_i, b_i, theta_off_i = fit_log_spiral_explicit_cached(i)
#                 xs_spine_i = df_spine.loc[actual_i, x_cols_spine].values
#                 ys_spine_i = df_spine.loc[actual_i, y_cols_spine].values
#                 zs_spine_i = df_spine.loc[actual_i, z_cols_spine].values
#                 points_spine_i = np.vstack([xs_spine_i, ys_spine_i, zs_spine_i]).T
#                 spine_2d_i = project_to_plane(points_spine_i, ref_centroid, ref_e1, ref_e2)
#                 x_i = spine_2d_i[:, 0]
#                 y_i = spine_2d_i[:, 1]
#                 theta_data_i = np.unwrap(np.arctan2(y_i - y0_i, x_i - x0_i))
#                 r_data_i = np.sqrt((x_i - x0_i)**2 + (y_i - y0_i)**2)
#                 r_model_i = a_i * safe_exp(b_i * (theta_data_i + theta_off_i))
#                 distances_i = np.abs(r_data_i - r_model_i)
                
#                 filter_idx = int(IGNORE_FIRST_X_DISPLACEMENT)
#                 filtered_distances_i = distances_i[filter_idx:]
#                 all_errors.append(filtered_distances_i)
#                 max_points = max(max_points, len(filtered_distances_i))
            
#             padded_errors = []
#             for err in all_errors:
#                 padded = np.full(max_points, np.nan)
#                 padded[:len(err)] = err
#                 padded_errors.append(padded)
            
#             avg_errors = np.nanmean(padded_errors, axis=0)
#             point_indices_avg = np.arange(int(IGNORE_FIRST_X_DISPLACEMENT), int(IGNORE_FIRST_X_DISPLACEMENT) + len(avg_errors))

#         # Plot all individual frames
#         if show_all_errors:
#             fixed_color = 'tab:blue'
#             alpha_val = 0.3 if show_average else 1.0
#             n_frames = len(df_spine) - IGNORE_FIRST_N_FRAMES
#             for i in range(n_frames):
#                 actual_i = i + IGNORE_FIRST_N_FRAMES
#                 x0_i, y0_i, a_i, b_i, theta_off_i = fit_log_spiral_explicit_cached(i)
#                 xs_spine_i = df_spine.loc[actual_i, x_cols_spine].values
#                 ys_spine_i = df_spine.loc[actual_i, y_cols_spine].values
#                 zs_spine_i = df_spine.loc[actual_i, z_cols_spine].values
#                 points_spine_i = np.vstack([xs_spine_i, ys_spine_i, zs_spine_i]).T
#                 spine_2d_i = project_to_plane(points_spine_i, ref_centroid, ref_e1, ref_e2)
#                 x_i = spine_2d_i[:, 0]
#                 y_i = spine_2d_i[:, 1]
#                 theta_data_i = np.unwrap(np.arctan2(y_i - y0_i, x_i - x0_i))
#                 r_data_i = np.sqrt((x_i - x0_i)**2 + (y_i - y0_i)**2)
#                 r_model_i = a_i * safe_exp(b_i * (theta_data_i + theta_off_i))
#                 distances_i = np.abs(r_data_i - r_model_i)
#                 point_indices_i = np.arange(len(distances_i))
#                 filter_idx = int(IGNORE_FIRST_X_DISPLACEMENT)
#                 filtered_indices_i = point_indices_i[filter_idx:]
#                 filtered_distances_i = distances_i[filter_idx:]
#                 error_ax.plot(filtered_indices_i, filtered_distances_i, color=fixed_color, alpha=alpha_val)

#         # Plot average line on top
#         if show_average:
#             error_ax.plot(point_indices_avg, avg_errors, 'r-', linewidth=4, label='Average Error')

#         # Plot single frame if neither is selected
#         if not show_all_errors and not show_average:
#             x0, y0, a, b, theta_off = fit_log_spiral_explicit_cached(row_idx)
#             spine_2d = project_to_plane(points_spine, ref_centroid, ref_e1, ref_e2)
#             x = spine_2d[:, 0]
#             y = spine_2d[:, 1]
#             theta_data = np.unwrap(np.arctan2(y - y0, x - x0))
#             r_data = np.sqrt((x - x0)**2 + (y - y0)**2)
#             r_model = a * safe_exp(b * (theta_data + theta_off))
#             distances = np.abs(r_data - r_model)
#             point_indices = np.arange(len(distances))
#             filter_idx = int(IGNORE_FIRST_X_DISPLACEMENT)
#             filtered_indices = point_indices[filter_idx:]
#             filtered_distances = distances[filter_idx:]
#             error_ax.plot(filtered_indices, filtered_distances, 'b-o', label='Distance from Fitted Curve')

#         error_ax.set_xlabel('Point Index (shifted)', fontsize=14)
#         error_ax.set_ylabel('Distance', fontsize=14)
#         error_ax.grid(True)
#         error_ax.set_ylim(0, 100)
        
#         if show_average and show_all_errors:
#             error_ax.set_title('Error Plot - All Frames + Average', fontsize=15)
#         elif show_average:
#             error_ax.set_title('Average Error Plot - All Frames', fontsize=15)
#         elif show_all_errors:
#             error_ax.set_title('Error Plot - All Frames (shifted)', fontsize=15)
#         else:
#             error_ax.set_title(f'Error Plot - Frame {actual_idx + 1} (shifted)', fontsize=15)
        
#         if show_average or (not show_all_errors and not show_average):
#             error_ax.legend(fontsize=12)

#     else:
#         if mode_2d:
#             if main_ax.name == '3d':
#                 fig.delaxes(main_ax)
#                 main_ax = fig.add_subplot(111)
            
#             spine_2d = project_to_plane(points_spine, ref_centroid, ref_e1, ref_e2)
#             tdcr_2d = project_to_plane(points_tdcr, ref_centroid, ref_e1, ref_e2)
            
#             if status[0]:
#                 spine_scatter = main_ax.scatter(spine_2d[:, 0], spine_2d[:, 1], c='b', marker='o', s=40, label='Spine Points')
#             if status[1]:
#                 tdcr_scatter = main_ax.scatter(tdcr_2d[:, 0], tdcr_2d[:, 1], c='g', marker='o', s=40, label='ROI Points')
            
#             if show_spiral:
#                 x = spine_2d[:, 0]
#                 y = spine_2d[:, 1]
#                 x0, y0, a, b, theta_off = fit_log_spiral_explicit_cached(row_idx)
#                 theta_data = np.unwrap(np.arctan2(y - y0, x - x0))
#                 theta_min, theta_max = np.min(theta_data), np.max(theta_data)
#                 theta_fit = np.linspace(theta_min, theta_max, 200)
#                 r_fit = a * safe_exp(b * (theta_fit + theta_off))
#                 x_fit = x0 + r_fit * np.cos(theta_fit)
#                 y_fit = y0 + r_fit * np.sin(theta_fit)
#                 spiral_line = main_ax.plot(x_fit, y_fit, 'r-', lw=2, label='Fitted Spiral')
            
#             main_ax.set_xlabel('Plane X', fontsize=14)
#             main_ax.set_ylabel('Plane Y', fontsize=14)
#             title = '2D Projection onto Reference Plane'
#             main_ax.set_title(title, fontsize=15)
#             main_ax.set_xlim(x2d_lim)
#             main_ax.set_ylim(y2d_lim)
#             main_ax.axis('equal')
#             main_ax.grid(True)
#         else:
#             if main_ax.name != '3d':
#                 fig.delaxes(main_ax)
#                 main_ax = fig.add_subplot(111, projection='3d')
            
#             if status[0]:
#                 spine_scatter = main_ax.scatter(xs_spine, ys_spine, zs_spine, c='b', marker='o', s=40, label='Spine Points')
#             if status[1]:
#                 tdcr_scatter = main_ax.scatter(xs_tdcr, ys_tdcr, zs_tdcr, c='g', marker='o', s=40, label='ROI Points')
            
#             f = 12
#             lp = 20
#             main_ax.set_xlabel('X', fontsize=f, labelpad=lp)
#             main_ax.set_ylabel('Y', fontsize=f, labelpad=lp)
#             main_ax.set_zlabel('Z', fontsize=f, labelpad=lp)
#             main_ax.tick_params(axis='both', which='major', labelsize=20)
#             main_ax.tick_params(axis='both', which='minor', labelsize=14)
#             main_ax.tick_params(axis='x', pad=10)
#             main_ax.tick_params(axis='y', pad=10)
#             main_ax.tick_params(axis='z', pad=10)
#             main_ax.set_xlim(x_lim)
#             main_ax.set_ylim(y_lim)
#             main_ax.set_zlim(z_lim)
            
#             try:
#                 main_ax.set_box_aspect([1, 1, 1])
#             except Exception:
#                 pass
        
#         legend_handles = []
#         legend_labels = []
#         if status[1] and tdcr_scatter is not None:
#             legend_handles.append(tdcr_scatter)
#             legend_labels.append('ROI Points')
#         if status[0] and spine_scatter is not None:
#             legend_handles.append(spine_scatter)
#             legend_labels.append('Spine Points')
#         if legend_handles:
#             legend_ax.legend(legend_handles, legend_labels, loc='upper left', fontsize=13, frameon=True)
    
#     plt.draw()

# def update(val):
#     status = check.get_status()
#     mode_2d = status[2]
#     show_spiral = status[3]
#     error_plot_mode = status[4]
#     show_all_errors = status[5]
#     show_average = status[6]
    
#     # Hide slider when showing average or all errors
#     slider_ax.set_visible(not (error_plot_mode and (show_all_errors or show_average)))
    
#     if error_plot_mode and (show_all_errors or show_average):
#         plot_row(0, mode_2d, show_spiral, error_plot_mode, show_all_errors, show_average)
#     else:
#         plot_row(int(slider.val) - 1, mode_2d, show_spiral, error_plot_mode, show_all_errors, show_average)

# def toggle_visibility(label):
#     update(None)

# # Initial plot with new parameter
# plot_row(0, False, True, False, False, False)

# slider.on_changed(update)
# check.on_clicked(toggle_visibility)

# plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons
from scipy.optimize import minimize, Bounds, differential_evolution

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

def safe_exp(x):
    x = np.clip(x, -20, 20)
    return np.exp(x)

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
    x_lim = [x_center - max_range/2 - margin*max_range, x_center + max_range/2 + margin*max_range]
    y_lim = [y_center - max_range/2 - margin*max_range, y_center + max_range/2 + margin*max_range]
    z_lim = [z_center - max_range/2 - margin*max_range, z_center + max_range/2 + margin*max_range]
    
    # Update reference plane
    xs0 = df_spine.loc[IGNORE_FIRST_N_FRAMES, x_cols_spine].values
    ys0 = df_spine.loc[IGNORE_FIRST_N_FRAMES, y_cols_spine].values
    zs0 = df_spine.loc[IGNORE_FIRST_N_FRAMES, z_cols_spine].values
    points0 = np.vstack([xs0, ys0, zs0]).T
    ref_centroid, ref_normal, ref_e1, ref_e2 = fit_plane(points0)

# Initialize
update_global_limits()

x2d_lim = (-150, 150)
y2d_lim = (-150, 150)

def calculate_point_displacements(points_current):
    """Calculate displacement of each point from reference frame"""
    ref_points = np.vstack([
        df_spine.loc[IGNORE_FIRST_N_FRAMES, x_cols_spine].values,
        df_spine.loc[IGNORE_FIRST_N_FRAMES, y_cols_spine].values,
        df_spine.loc[IGNORE_FIRST_N_FRAMES, z_cols_spine].values]).T
    
    if len(ref_points) != len(points_current):
        return np.zeros(len(points_current))
    
    diffs = points_current - ref_points
    distances = np.linalg.norm(diffs, axis=1)
    return distances

spiral_fit_cache = {}

def fast_outlier_detection(x, y):
    """Fast outlier detection using simple distance thresholding"""
    center_x, center_y = np.median(x), np.median(y)
    distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    threshold = np.percentile(distances, 85)  # Keep 85% of points (faster than MAD)
    return distances < threshold

def fit_log_spiral_explicit(x, y, maxiter=15000, use_global_opt=False, iter_refine=2):
    """Optimized spiral fitting - faster but still robust"""
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]
    
    if len(x) < 6:  # Reduced from 8 to 6
        return np.nan, np.nan, np.nan, np.nan, np.nan
    
    # Fast outlier detection
    keep_mask = fast_outlier_detection(x, y)
    x = x[keep_mask]
    y = y[keep_mask]
    
    if len(x) < 6:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    
    # Quick initial parameter estimation
    x0_init = np.median(x)
    y0_init = np.median(y)
    
    theta = np.unwrap(np.arctan2(y - y0_init, x - x0_init))
    r = np.sqrt((x - x0_init)**2 + (y - y0_init)**2)
    
    # Simple filtering - keep points with reasonable radius
    min_r = np.percentile(r, 5)
    valid_r_mask = r > min_r
    
    if np.sum(valid_r_mask) < 5:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    
    theta_fit = theta[valid_r_mask]
    r_fit = r[valid_r_mask]
    
    # Fast weighted fitting
    try:
        weights = np.sqrt(r_fit) / np.max(r_fit)
        p = np.polyfit(theta_fit, np.log(np.maximum(r_fit, 1e-8)), 1, w=weights)
        a0 = np.exp(p[1])
        b0 = p[0]
    except:
        a0 = np.median(r_fit)
        b0 = 0.1
    
    # Reduced to 2 attempts max, with simpler starting points
    best_cost = np.inf
    best_params = None
    
    for attempt in range(2):  # Reduced from 3 to 2
        theta0 = 0 if attempt == 0 else np.random.uniform(-np.pi/2, np.pi/2)
        x0_try = x0_init if attempt == 0 else x0_init + np.random.normal(0, np.std(x)/20)
        y0_try = y0_init if attempt == 0 else y0_init + np.random.normal(0, np.std(y)/20)
        
        params0 = np.array([x0_try, y0_try, np.log(np.maximum(a0, 1e-8)), b0, theta0], dtype=float)
        
        # Tighter bounds for faster convergence
        x_range = np.max(x) - np.min(x)
        y_range = np.max(y) - np.min(y)
        bounds = Bounds([np.min(x) - 0.05*x_range, np.min(y) - 0.05*y_range, -10, -2.5, -2*np.pi], 
                       [np.max(x) + 0.05*x_range, np.max(y) + 0.05*y_range, 10, 2.5, 2*np.pi])
        
        def spiral_cost(params):
            x0, y0, loga, b, theta_off = params
            try:
                theta_data = np.unwrap(np.arctan2(y - y0, x - x0))
                r_data = np.sqrt((x - x0)**2 + (y - y0)**2)
                
                if np.all(r_data < 1e-8):
                    return 1e10
                
                model_log_r = loga + b*(theta_data + theta_off)
                log_r_data = np.log(np.maximum(r_data, 1e-8))
                
                # Simple L2 loss for speed (instead of Huber)
                residuals = log_r_data - model_log_r
                loss = np.sum(residuals**2)
                
                # Light regularization
                reg = 0.001*(b**2 + theta_off**2)
                
                return loss + reg
            except:
                return 1e10
        
        try:
            # Use global optimization only on first attempt and only if explicitly requested
            if use_global_opt and attempt == 0:
                bounds_list = [(np.min(x) - 0.05*x_range, np.max(x) + 0.05*x_range),
                              (np.min(y) - 0.05*y_range, np.max(y) + 0.05*y_range),
                              (-10, 10), (-2.5, 2.5), (-2*np.pi, 2*np.pi)]
                res = differential_evolution(spiral_cost, bounds_list, maxiter=maxiter//5, 
                                           tol=1e-8, seed=42)
            else:
                res = minimize(spiral_cost, params0, method='L-BFGS-B', bounds=bounds, 
                             options={'maxiter': maxiter//2, 'ftol': 1e-8})
            
            if res.success and res.fun < best_cost:
                best_cost = res.fun
                best_params = res.x.copy()
                # Early stopping if cost is good enough
                if best_cost < 1.0:
                    break
        except:
            continue
    
    if best_params is None:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    
    x0_fit, y0_fit, loga_fit, b_fit, theta_off_fit = best_params
    a_fit = np.exp(np.clip(loga_fit, -20, 20))
    
    # Reduced refinement iterations
    for iteration in range(iter_refine):
        try:
            theta_data = np.unwrap(np.arctan2(y - y0_fit, x - x0_fit))
            r_data = np.sqrt((x - x0_fit)**2 + (y - y0_fit)**2)
            r_model = a_fit * safe_exp(b_fit * (theta_data + theta_off_fit))
            residuals = np.abs(r_data - r_model)
            
            # Fast percentile-based outlier detection
            threshold = np.percentile(residuals, 75)
            keep_mask = residuals < threshold
            
            if np.sum(keep_mask) < 6:
                break
            
            x_refine = x[keep_mask]
            y_refine = y[keep_mask]
            
            if len(x_refine) < 6:
                break
            
            # Quick re-estimation
            x0_init = np.median(x_refine)
            y0_init = np.median(y_refine)
            theta = np.unwrap(np.arctan2(y_refine - y0_init, x_refine - x0_init))
            r = np.sqrt((x_refine - x0_init)**2 + (y_refine - y0_init)**2)
            
            valid_mask = r > np.percentile(r, 5)
            if np.sum(valid_mask) < 4:
                break
                
            weights = np.sqrt(r[valid_mask]) / np.max(r[valid_mask])
            p = np.polyfit(theta[valid_mask], np.log(np.maximum(r[valid_mask], 1e-8)), 1, w=weights)
            a0 = np.exp(p[1])
            b0 = p[0]
            
            params0 = np.array([x0_init, y0_init, np.log(np.maximum(a0, 1e-8)), b0, theta_off_fit], dtype=float)
            
            res = minimize(spiral_cost, params0, method='L-BFGS-B', bounds=bounds, 
                         options={'maxiter': maxiter//4, 'ftol': 1e-8})
            
            if res.success:
                x0_fit, y0_fit, loga_fit, b_fit, theta_off_fit = res.x
                a_fit = np.exp(np.clip(loga_fit, -20, 20))
            else:
                break
        except:
            break
    
    return x0_fit, y0_fit, a_fit, b_fit, theta_off_fit

def fit_log_spiral_explicit_cached(frame_idx):
    cache_key = (frame_idx, IGNORE_FIRST_N_FRAMES, IGNORE_FIRST_X_DISPLACEMENT)
    if cache_key in spiral_fit_cache:
        return spiral_fit_cache[cache_key]
    
    actual_idx = frame_idx + IGNORE_FIRST_N_FRAMES
    xs_spine = df_spine.loc[actual_idx, x_cols_spine].values
    ys_spine = df_spine.loc[actual_idx, y_cols_spine].values
    zs_spine = df_spine.loc[actual_idx, z_cols_spine].values
    points_spine = np.vstack([xs_spine, ys_spine, zs_spine]).T
    
    spine_2d = project_to_plane(points_spine, ref_centroid, ref_e1, ref_e2)
    x = spine_2d[:, 0]
    y = spine_2d[:, 1]
    
    # Use global optimization sparingly - only for every 10th frame or when requested
    use_global = (frame_idx % 10 == 0)
    fit = fit_log_spiral_explicit(x, y, use_global_opt=use_global)
    spiral_fit_cache[cache_key] = fit
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

# Create figure with adjusted positioning
fig = plt.figure(figsize=(12, 9))
main_ax = fig.add_subplot(111, projection='3d')

# Adjusted positioning for more sliders
plt.subplots_adjust(left=0.05, right=0.82, bottom=0.35, top=0.85)
error_ax = fig.add_axes([0.1, 0.25, 0.7, 0.55])
error_ax.set_visible(False)

# Slider positions - moved down to make room for new sliders
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
    xs_spine = df_spine.loc[actual_idx, x_cols_spine].values
    ys_spine = df_spine.loc[actual_idx, y_cols_spine].values
    zs_spine = df_spine.loc[actual_idx, z_cols_spine].values
    points_spine = np.vstack([xs_spine, ys_spine, zs_spine]).T
    
    # Apply displacement filtering
    spine_displacements = calculate_point_displacements(points_spine)
    low_displacement_mask = spine_displacements < IGNORE_FIRST_X_DISPLACEMENT
    xs_spine[low_displacement_mask] = df_spine.loc[IGNORE_FIRST_N_FRAMES, x_cols_spine].values[low_displacement_mask]
    ys_spine[low_displacement_mask] = df_spine.loc[IGNORE_FIRST_N_FRAMES, y_cols_spine].values[low_displacement_mask]
    zs_spine[low_displacement_mask] = df_spine.loc[IGNORE_FIRST_N_FRAMES, z_cols_spine].values[low_displacement_mask]
    points_spine = np.vstack([xs_spine, ys_spine, zs_spine]).T
    
    xs_tdcr = df_tdcr.loc[actual_idx, x_cols_tdcr].values
    ys_tdcr = df_tdcr.loc[actual_idx, y_cols_tdcr].values
    zs_tdcr = df_tdcr.loc[actual_idx, z_cols_tdcr].values
    points_tdcr = np.vstack([xs_tdcr, ys_tdcr, zs_tdcr]).T
    
    status = check.get_status()
    
    if error_plot_mode:
        # Calculate average data if needed
        if show_average:
            n_frames = len(df_spine) - IGNORE_FIRST_N_FRAMES
            all_errors = []
            max_points = 0
            
            # Sample every 5th frame for speed when calculating average
            sample_step = max(1, n_frames // 100)  # Process max 100 frames for average
            
            for i in range(0, n_frames, sample_step):
                actual_i = i + IGNORE_FIRST_N_FRAMES
                x0_i, y0_i, a_i, b_i, theta_off_i = fit_log_spiral_explicit_cached(i)
                if np.isnan(x0_i):
                    continue
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
                all_errors.append(filtered_distances_i)
                max_points = max(max_points, len(filtered_distances_i))
            
            if all_errors:
                padded_errors = []
                for err in all_errors:
                    padded = np.full(max_points, np.nan)
                    padded[:len(err)] = err
                    padded_errors.append(padded)
                
                avg_errors = np.nanmean(padded_errors, axis=0)
                point_indices_avg = np.arange(int(IGNORE_FIRST_X_DISPLACEMENT), int(IGNORE_FIRST_X_DISPLACEMENT) + len(avg_errors))
        
        # Plot all individual frames (heavily sampled for performance)
        if show_all_errors:
            fixed_color = 'tab:blue'
            alpha_val = 0.3 if show_average else 1.0
            n_frames = len(df_spine) - IGNORE_FIRST_N_FRAMES
            sample_step = max(1, n_frames // 150)  # Sample max 150 frames
            
            for i in range(0, n_frames, sample_step):
                actual_i = i + IGNORE_FIRST_N_FRAMES
                x0_i, y0_i, a_i, b_i, theta_off_i = fit_log_spiral_explicit_cached(i)
                if np.isnan(x0_i):
                    continue
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
                error_ax.plot(filtered_indices_i, filtered_distances_i, color=fixed_color, alpha=alpha_val)

        # Plot average line on top
        if show_average and 'avg_errors' in locals():
            error_ax.plot(point_indices_avg, avg_errors, 'r-', linewidth=4, label='Average Error')

        # Plot single frame if neither is selected
        if not show_all_errors and not show_average:
            x0, y0, a, b, theta_off = fit_log_spiral_explicit_cached(row_idx)
            if not np.isnan(x0):
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
                error_ax.plot(filtered_indices, filtered_distances, 'b-o', label='Distance from Fitted Curve')

        error_ax.set_xlabel('Point Index (shifted)', fontsize=14)
        error_ax.set_ylabel('Distance', fontsize=14)
        error_ax.grid(True)
        error_ax.set_ylim(0, 100)
        
        if show_average and show_all_errors:
            error_ax.set_title('Error Plot - All Frames + Average (sampled)', fontsize=15)
        elif show_average:
            error_ax.set_title('Average Error Plot - All Frames (sampled)', fontsize=15)
        elif show_all_errors:
            error_ax.set_title('Error Plot - All Frames (sampled)', fontsize=15)
        else:
            error_ax.set_title(f'Error Plot - Frame {actual_idx + 1} (shifted)', fontsize=15)
        
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
                    theta_data = np.unwrap(np.arctan2(y - y0, x - x0))
                    theta_min, theta_max = np.min(theta_data), np.max(theta_data)
                    theta_fit = np.linspace(theta_min, theta_max, 200)
                    r_fit = a * safe_exp(b * (theta_fit + theta_off))
                    x_fit = x0 + r_fit * np.cos(theta_fit)
                    y_fit = y0 + r_fit * np.sin(theta_fit)
                    spiral_line = main_ax.plot(x_fit, y_fit, 'r-', lw=3, label='Fitted Spiral')
            
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

# Initial plot
plot_row(0, False, True, False, False, False)

plt.show()
