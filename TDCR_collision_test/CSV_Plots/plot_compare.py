import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button

# --- User-defined variables ---
DeltaLv = 8.0  
alpha = 90.0    

def rms(values):
    """Calculate root mean square of a list or array of 3 values."""
    arr = np.array(values)
    return np.sqrt(np.mean(arr**2))

def get_cos_rms(alpha_deg):
    """Calculate RMS of cos(alpha), cos(120-alpha), cos(240-alpha) (alpha in degrees)."""
    angles = [alpha_deg, 120 - alpha_deg, 240 - alpha_deg]
    cosines = [np.cos(np.radians(a)) for a in angles]
    return rms(cosines)

# Load the five CSV files
no_object_df = pd.read_csv('data1.csv')
object_6k_df = pd.read_csv('data2.csv')
object_60k_df = pd.read_csv('data3.csv')
object_600k_df = pd.read_csv('data4.csv')
rigid_object_df = pd.read_csv('data5.csv')

# Calculate RMS of forces for each row
no_object_df['force_rms'] = no_object_df[['F1', 'F2', 'F3']].apply(rms, axis=1)
object_6k_df['force_rms'] = object_6k_df[['F1', 'F2', 'F3']].apply(rms, axis=1)
object_60k_df['force_rms'] = object_60k_df[['F1', 'F2', 'F3']].apply(rms, axis=1)
object_600k_df['force_rms'] = object_600k_df[['F1', 'F2', 'F3']].apply(rms, axis=1)
rigid_object_df['force_rms'] = rigid_object_df[['F1', 'F2', 'F3']].apply(rms, axis=1)

# Calculate RMS of cosines (same for all rows, since alpha is fixed)
cos_rms = get_cos_rms(alpha)

# Calculate Fv for all dataframes
no_object_df['Fv'] = no_object_df['force_rms'] / cos_rms
object_6k_df['Fv'] = object_6k_df['force_rms'] / cos_rms
object_60k_df['Fv'] = object_60k_df['force_rms'] / cos_rms
object_600k_df['Fv'] = object_600k_df['force_rms'] / cos_rms
rigid_object_df['Fv'] = rigid_object_df['force_rms'] / cos_rms

# Calculate differences (align on index)
min_len = min(len(no_object_df), len(object_6k_df), len(object_60k_df), len(object_600k_df), len(rigid_object_df))
Fv_data = [
    (no_object_df, 'No Object'),
    (object_6k_df, 'Object 6,000 kPa'),
    (object_60k_df, 'Object 60,000 kPa'),
    (object_600k_df, 'Object 600,000 kPa'),
    (rigid_object_df, 'Rigid Object')
]

def plot_all(log_scale=False):
    plt.clf()
    plt.figure(figsize=(10, 6))
    for df, label in Fv_data:
        y = df['Fv'][:min_len]
        if log_scale:
            y = np.log10(np.clip(y, a_min=1e-8, a_max=None))
        plt.plot(df.index[:min_len], y, label=label)
    plt.xlabel('DeltaLv Step')
    plt.ylabel('log10(Fv)' if log_scale else 'Fv')
    plt.title(f'Fv vs DeltaLv Step\nalpha = {alpha}°, DeltaLv = {DeltaLv}' + (" (log scale)" if log_scale else ""))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.draw()

# --- Matplotlib interactive plot with toggle button ---
fig = plt.figure(figsize=(10, 6))
ax_button = plt.axes([0.8, 0.01, 0.15, 0.06])
toggle_button = Button(ax_button, 'Toggle log/linear')
log_state = {'log': False}

def on_toggle(event):
    log_state['log'] = not log_state['log']
    plot_all(log_state['log'])

plot_all(log_state['log'])
toggle_button.on_clicked(on_toggle)
plt.show()
