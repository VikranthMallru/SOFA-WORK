import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds

# ---------- Matplotlib global styling ----------
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"]  = ["cmr10"]
plt.rcParams["axes.formatter.use_mathtext"] = True

# ---------- Size controls ----------
BASE_FONTSIZE = 40          # ❶ bigger fonts
LINEW         = 10           # ❷ thicker lines
MARKERSZ      = 10           # ❸ larger points

plt.rcParams.update({
    "axes.titlesize":   BASE_FONTSIZE + 2,
    "axes.labelsize":   BASE_FONTSIZE,
    "xtick.labelsize":  BASE_FONTSIZE - 2,
    "ytick.labelsize":  BASE_FONTSIZE - 2,
    "legend.fontsize":  BASE_FONTSIZE - 2,
    "figure.titlesize": BASE_FONTSIZE + 4
})

# ---------- Load data ----------
df_spine = pd.read_csv("tdcr_trunk_spine.csv")
x_cols   = [c for c in df_spine.columns if c.startswith("x")]
y_cols   = [c for c in df_spine.columns if c.startswith("y")]
z_cols   = [c for c in df_spine.columns if c.startswith("z")]

# ---------- Helper functions ----------
def fit_plane(pts):
    centroid = np.mean(pts, axis=0)
    _, _, vh = np.linalg.svd(pts - centroid)
    return centroid, vh[2], vh[0], vh[1]

def project_to_plane(pts, origin, e1, e2):
    rel = pts - origin
    return np.column_stack([rel @ e1, rel @ e2])

def safe_exp(x):
    return np.exp(np.clip(x, -20, 20))

def fit_log_spiral_explicit(x, y, maxiter=5_000):
    if x.size < 5: return [np.nan]*5
    x0, y0 = np.mean(x), np.mean(y)
    theta  = np.unwrap(np.arctan2(y - y0, x - x0))
    r      = np.hypot(x - x0, y - y0).clip(min=1e-6)
    p      = np.polyfit(theta, np.log(r), 1)
    params0 = [x0, y0, np.log(np.exp(p[1])), p[0], 0.0]
    bnds = Bounds([x.min(), y.min(), -10, -2, -2*np.pi],
                  [x.max(), y.max(),  10,  2,  2*np.pi])
    def cost(p):
        x0, y0, loga, b, t_off = p
        td = np.unwrap(np.arctan2(y - y0, x - x0))
        rd = np.hypot(x - x0, y - y0)
        return ((np.log(rd) - (loga + b*(td + t_off)))**2).sum()
    res = minimize(cost, params0, method="L-BFGS-B",
                   bounds=bnds, options={"maxiter": maxiter})
    x0f, y0f, loga, b, t_off = res.x
    return x0f, y0f, safe_exp(loga), b, t_off

def rms_percent_error(x, y, x0, y0, a, b, t_off):
    theta = np.unwrap(np.arctan2(y - y0, x - x0))
    r     = np.hypot(x - x0, y - y0)
    model = a * safe_exp(b*(theta + t_off))
    mean_r = r.mean()
    return np.nan if mean_r < 1e-8 else 100*np.sqrt(((r - model)**2).mean())/mean_r

def is_straight(pts2d, ang_thresh=0.5, svd_ratio=20):
    pts_c = pts2d - pts2d.mean(axis=0)
    _, s, _ = np.linalg.svd(pts_c)
    if s[1] == 0: return True
    ratio   = s[0]/s[1]
    ang_span= np.ptp(np.arctan2(pts2d[:,1]-pts2d[:,1].mean(),
                                pts2d[:,0]-pts2d[:,0].mean()))
    return ratio > svd_ratio or ang_span < ang_thresh

# ---------- Core processing ----------
all_rms, disp = [], []
for idx in range(len(df_spine)):
    xs, ys, zs = [df_spine.loc[idx, cols].values for cols in (x_cols, y_cols, z_cols)]
    pts        = np.column_stack([xs, ys, zs])
    centroid, _, e1, e2 = fit_plane(pts)
    pts2d = project_to_plane(pts, centroid, e1, e2)
    if is_straight(pts2d):
        all_rms.append(0.0)
    else:
        x, y = pts2d[:,0], pts2d[:,1]
        parms = fit_log_spiral_explicit(x, y)
        all_rms.append(rms_percent_error(x, y, *parms))
    disp.append(df_spine.loc[idx, "L1"] if "L1" in df_spine else idx)

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
# ax.plot(disp[START:][mask], diff[START:][mask], "-", color="firebrick",
#         linewidth=LINEW, label="d(Error)/dFrame")
ax.axhline(15, color="firebrick", linestyle="--",
           linewidth=LINEW-2, label="15% threshold")

ax.set_xlabel("Cable Displacement", fontsize=BASE_FONTSIZE)
ax.set_ylabel("Value(%)",             fontsize=BASE_FONTSIZE)
ax.set_title("RMS Error (%)",      fontsize=BASE_FONTSIZE + 2)

ax.tick_params(labelsize=BASE_FONTSIZE)
ax.grid(True, which="both", linestyle=":")
ax.legend()
fig.tight_layout()
plt.show()
