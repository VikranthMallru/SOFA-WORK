import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import os
import numpy as np

# --- Set matplotlib to use Computer Modern (default LaTeX) font and large size ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['cmr10']  # Computer Modern Roman 10pt
plt.rcParams['axes.formatter.use_mathtext'] = True
plt.rcParams['font.size'] = 32  # Set global font size very large for all elements

summary_file = "tdcr_theta_optim_summary_res3.csv"

if os.path.exists(summary_file):
    df = pd.read_csv(summary_file)
    if "theta" not in df.columns or "RMS_diff" not in df.columns:
        raise ValueError("CSV must contain 'theta' and 'RMS_diff' columns.")
    theta = df["theta"]
    rms_diff = df["RMS_diff"]
    best_idx = df["RMS_diff"].idxmin()
    best_theta = df.loc[best_idx, "theta"]
    best_rms = df.loc[best_idx, "RMS_diff"]
else:
    # Simulate data for demonstration if file is missing
    theta = np.linspace(0, 360, 121)
    rms_diff = 10 + 40 * np.abs(np.sin(np.radians(theta*3)))
    best_theta = theta[np.argmin(rms_diff)]
    best_rms = rms_diff.min()

plt.figure(figsize=(14, 10))  # Even larger figure for clarity

plt.plot(
    theta, rms_diff, 
    marker='o', 
    label="All thetas", 
    linewidth=5, 
    markersize=12
)

plt.xlabel("$\\theta$ (degrees) ", fontsize=54, labelpad=28)  # Increased font size here
plt.ylabel("RMS$_{\\mathrm{diff}}$ (Newtons)", fontsize=54, labelpad=28)  # Increased font size here
plt.ylim(0, max(50, np.max(rms_diff)*1.1))

plt.scatter(
    [best_theta], [best_rms],
    s=60,
    facecolors='none', 
    edgecolors='red', 
    linewidths=10,
    zorder=10, 
    label="Best value",
    path_effects=[path_effects.withStroke(linewidth=3, foreground='black')]
)

plt.title(
    f"Cable Force Equality (RMS$_{{\\mathrm{{diff}}}}$) vs Theta\n"
    f"Best: $\\theta$={best_theta}, RMS$_{{\\mathrm{{diff}}}}$={best_rms:.3f} N",
    fontsize=44, pad=38
)

plt.legend(fontsize=32, loc='best')
plt.grid(True, linewidth=3, alpha=0.7)

plt.xticks(fontsize=34)
plt.yticks(fontsize=34)

plt.tight_layout()
plt.show()
