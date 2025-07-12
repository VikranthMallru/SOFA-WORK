import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['cmr10']
plt.rcParams["axes.formatter.use_mathtext"] = True

# --- Load data ---
df_spine = pd.read_csv('tdcr_trunk_spine.csv')
x_cols_spine = [col for col in df_spine.columns if col.startswith('x')]
y_cols_spine = [col for col in df_spine.columns if col.startswith('y')]
z_cols_spine = [col for col in df_spine.columns if col.startswith('z')]

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

# --- Compute error for all frames ---
all_rms_errors = []
L1_disp = []
for row_idx in range(len(df_spine)):
    xs_spine = df_spine.loc[row_idx, x_cols_spine].values
    ys_spine = df_spine.loc[row_idx, y_cols_spine].values
    zs_spine = df_spine.loc[row_idx, z_cols_spine].values
    points_spine = np.vstack([xs_spine, ys_spine, zs_spine]).T
    centroid, normal, e1, e2 = fit_plane(points_spine)
    points_2d = project_to_plane(points_spine, centroid, e1, e2)
    if is_straight(points_2d):
        all_rms_errors.append(0.0)
    else:
        x = points_2d[:, 0]
        y = points_2d[:, 1]
        x0, y0, a, b, theta_off = fit_log_spiral_explicit(x, y)
        if not (np.isnan(a) or np.isinf(a) or np.isnan(b) or np.isinf(b)):
            percent_error = calculate_rms_percent_error(x, y, x0, y0, a, b, theta_off)
        else:
            percent_error = np.nan
        all_rms_errors.append(percent_error)
    # Cable displacement (L1)
    if 'L1' in df_spine.columns:
        L1 = df_spine.loc[row_idx, 'L1']
    else:
        L1 = row_idx
    L1_disp.append(L1)

all_rms_errors = np.array(all_rms_errors)
L1_disp = np.array(L1_disp)

# --- Compute differentiation (finite difference) curve ---
diff_curve = np.gradient(all_rms_errors)

# --- Only plot from a given frame onward ---
start_idx = 300 # 0-based index for 300th frame; edit this value to change starting frame

# --- Mask out points where differentiation > 5 ---
mask = np.abs(diff_curve[start_idx:]) <= 0.01

plt.figure(figsize=(6, 6))
plt.plot(
    L1_disp[start_idx:][mask],
    all_rms_errors[start_idx:][mask],
    'bo-', linewidth=2, markersize=2, label='RMS % Error'
)
plt.plot(
    L1_disp[start_idx:][mask],
    diff_curve[start_idx:][mask],
    'r-', linewidth=2, label='d(Error)/dFrame'
)
plt.axhline(15, color='gray', linestyle='--', linewidth=1, label='15% threshold')
plt.xlabel('Cable Displacement (L1)', fontsize=14)
plt.ylabel('Value', fontsize=14)
plt.title(f'RMS Error (%)', fontsize=15)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
