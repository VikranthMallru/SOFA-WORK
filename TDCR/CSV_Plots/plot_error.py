# import pandas as pd
# import numpy as np

# def fit_circle_3d(p1, p2, p3):
#     v1 = p2 - p1
#     v2 = p3 - p1
#     normal = np.cross(v1, v2)
#     norm_normal = np.linalg.norm(normal)
#     if norm_normal == 0:
#         return None, np.inf, None, None, None
#     normal = normal / norm_normal
#     e1 = v1 / np.linalg.norm(v1)
#     e2 = np.cross(normal, e1)

#     def to_2d(p):
#         return np.array([np.dot(p - p1, e1), np.dot(p - p1, e2)])

#     p1_2d = to_2d(p1)
#     p2_2d = to_2d(p2)
#     p3_2d = to_2d(p3)

#     A = np.array([
#         [2*(p2_2d[0]-p1_2d[0]), 2*(p2_2d[1]-p1_2d[1])],
#         [2*(p3_2d[0]-p1_2d[0]), 2*(p3_2d[1]-p1_2d[1])]
#     ])

#     b = np.array([
#         p2_2d[0]**2 + p2_2d[1]**2 - p1_2d[0]**2 - p1_2d[1]**2,
#         p3_2d[0]**2 + p3_2d[1]**2 - p1_2d[0]**2 - p1_2d[1]**2
#     ])

#     try:
#         center_2d = np.linalg.solve(A, b)
#     except np.linalg.LinAlgError:
#         return None, np.inf, None, None, None

#     radius = np.linalg.norm(center_2d - p1_2d)
#     center_3d = p1 + center_2d[0]*e1 + center_2d[1]*e2  # Fixed broadcasting issue

#     return center_3d, radius, normal, e1, e2

# def calculate_error_percent(points_spine, center, radius, first_point, last_point):
#     large_radius_threshold = 1e6
#     if radius == np.inf or radius > large_radius_threshold or center is None:
#         line_vec = last_point - first_point
#         line_len = np.linalg.norm(line_vec)
#         if line_len == 0:
#             return 0.0
#         line_unit = line_vec / line_len
#         diffs = points_spine - first_point
#         proj_lengths = np.dot(diffs, line_unit)
#         proj_points = first_point + np.outer(proj_lengths, line_unit)
#         dists = np.linalg.norm(points_spine - proj_points, axis=1)
#         total_error = np.sum(np.abs(dists))
#         percent_error = (total_error / (len(points_spine) * line_len)) * 100
#         return percent_error
#     else:
#         distances = np.linalg.norm(points_spine - center, axis=1)
#         total_error = np.sum(np.abs(distances - radius))
#         percent_error = (total_error / (len(points_spine) * radius)) * 100 if radius != 0 else 0
#         return percent_error

# def main():
#     df_spine = pd.read_csv('tdcr_spine.csv')

#     x_cols_spine = [col for col in df_spine.columns if col.startswith('x')]
#     y_cols_spine = [col for col in df_spine.columns if col.startswith('y')]
#     z_cols_spine = [col for col in df_spine.columns if col.startswith('z')]

#     for idx, row in df_spine.iterrows():
#         xs_spine = row[x_cols_spine].values
#         ys_spine = row[y_cols_spine].values
#         zs_spine = row[z_cols_spine].values
#         points_spine = np.vstack([xs_spine, ys_spine, zs_spine]).T
#         first_point = points_spine[0]
#         middle_point = points_spine[len(points_spine)//2]
#         last_point = points_spine[-1]
#         center, radius, normal, e1, e2 = fit_circle_3d(first_point, middle_point, last_point)
#         error_percent = calculate_error_percent(points_spine, center, radius, first_point, last_point)
#         print(f'Index: {idx}, Error%: {error_percent:.4f}')

# if __name__ == '__main__':
#     main()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------- Matplotlib global styling ----------
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"]  = ["cmr10"]
plt.rcParams["axes.formatter.use_mathtext"] = True

# ---------- Size controls ----------
BASE_FONTSIZE = 40          # bigger fonts
LINEW         = 10           # thicker lines
MARKERSZ      = 10           # larger points

plt.rcParams.update({
    "axes.titlesize":   BASE_FONTSIZE + 2,
    "axes.labelsize":   BASE_FONTSIZE,
    "xtick.labelsize":  BASE_FONTSIZE - 2,
    "ytick.labelsize":  BASE_FONTSIZE - 2,
    "legend.fontsize":  BASE_FONTSIZE - 2,
    "figure.titlesize": BASE_FONTSIZE + 4
})

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

def calculate_error_percent(points_spine, center, radius, first_point, last_point):
    large_radius_threshold = 1e6
    if radius == np.inf or radius > large_radius_threshold or center is None:
        line_vec = last_point - first_point
        line_len = np.linalg.norm(line_vec)
        if line_len == 0:
            return 0.0
        line_unit = line_vec / line_len
        diffs = points_spine - first_point
        proj_lengths = np.dot(diffs, line_unit)
        proj_points = first_point + np.outer(proj_lengths, line_unit)
        dists = np.linalg.norm(points_spine - proj_points, axis=1)
        total_error = np.sum(np.abs(dists))
        percent_error = (total_error / (len(points_spine) * line_len)) * 100
        return percent_error
    else:
        distances = np.linalg.norm(points_spine - center, axis=1)
        total_error = np.sum(np.abs(distances - radius))
        percent_error = (total_error / (len(points_spine) * radius)) * 100 if radius != 0 else 0
        return percent_error

def main():
    df_spine = pd.read_csv('tdcr_spine.csv')

    x_cols_spine = [col for col in df_spine.columns if col.startswith('x')]
    y_cols_spine = [col for col in df_spine.columns if col.startswith('y')]
    z_cols_spine = [col for col in df_spine.columns if col.startswith('z')]

    # Use 'L1' as cable displacement
    if 'L1' in df_spine.columns:
        disp = df_spine['L1'].values
    else:
        print("Warning: 'L1' column not found. Using row indices as x-axis.")
        disp = np.arange(len(df_spine))

    all_rms = []
    for idx, row in df_spine.iterrows():
        xs_spine = row[x_cols_spine].values
        ys_spine = row[y_cols_spine].values
        zs_spine = row[z_cols_spine].values
        points_spine = np.vstack([xs_spine, ys_spine, zs_spine]).T
        first_point = points_spine[0]
        middle_point = points_spine[len(points_spine)//2]
        last_point = points_spine[-1]

        center, radius, normal, e1, e2 = fit_circle_3d(first_point, middle_point, last_point)
        error_percent = calculate_error_percent(points_spine, center, radius, first_point, last_point)

        all_rms.append(error_percent)

    all_rms = np.asarray(all_rms)
    disp    = np.asarray(disp)
    diff    = np.gradient(all_rms)

    # ---------- Plot ----------
    START = 0
    mask  = np.abs(diff[START:]) <= 0.01

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(disp[START:][mask], all_rms[START:][mask],
            "o-", color="royalblue", linewidth=LINEW,
            markersize=MARKERSZ, label="RMS % Error")
    ax.axhline(15, color="firebrick", linestyle="--",
               linewidth=LINEW-2, label="15% threshold")

    ax.set_xlabel("Cable Displacement(mm)", fontsize=BASE_FONTSIZE)
    ax.set_ylabel("Value(%)",             fontsize=BASE_FONTSIZE)
    ax.set_title("RMS Error (%)",      fontsize=BASE_FONTSIZE + 2)

    ax.tick_params(labelsize=BASE_FONTSIZE)
    ax.grid(True, which="both", linestyle=":")
    ax.legend()
    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
