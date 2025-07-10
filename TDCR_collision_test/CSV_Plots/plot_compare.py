import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, CheckButtons

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

# --- Matplotlib interactive plot with toggle buttons ---
fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(left=0.25, bottom=0.22)
ax_button = plt.axes([0.7, 0.01, 0.13, 0.06])
toggle_button = Button(ax_button, 'Toggle log/linear')
ax_exp_button = plt.axes([0.84, 0.01, 0.13, 0.06])
exp_button = Button(ax_exp_button, 'Toggle exp/normal')

# CheckButtons for toggling each plot
labels = [label for _, label in Fv_data]
visibility = [True] * len(labels)
check_ax = plt.axes([0.01, 0.25, 0.18, 0.25])
check = CheckButtons(check_ax, labels, visibility)

log_state = {'log': False}
exp_state = {'exp': False}
plot_visibility = {label: True for label in labels}

def plot_all():
    ax.clear()
    for (df, label) in Fv_data:
        if plot_visibility[label]:
            y = df['Fv'][:min_len]
            if log_state['log']:
                y = np.log10(np.clip(y, a_min=1e-8, a_max=None))
            elif exp_state['exp']:
                y = np.exp(np.clip(y, a_min=None, a_max=50))
            ax.plot(df.index[:min_len], y, label=label)
    ax.set_xlabel('DeltaLv Step')
    if log_state['log']:
        ax.set_ylabel('log10(Fv)')
    elif exp_state['exp']:
        ax.set_ylabel('exp(Fv)')
    else:
        ax.set_ylabel('Fv')
    ax.set_title(f'Fv vs DeltaLv Step\nalpha = {alpha}°, DeltaLv = {DeltaLv}' +
                 (" (log scale)" if log_state['log'] else "") +
                 (" (exp scale)" if exp_state['exp'] else ""))
    ax.legend()
    ax.grid(True)
    fig.canvas.draw_idle()

def on_toggle(event):
    log_state['log'] = not log_state['log']
    if log_state['log']:
        exp_state['exp'] = False  # Disable exp if log is enabled
    plot_all()

def on_exp_toggle(event):
    exp_state['exp'] = not exp_state['exp']
    if exp_state['exp']:
        log_state['log'] = False  # Disable log if exp is enabled
    plot_all()

def func(label):
    plot_visibility[label] = not plot_visibility[label]
    plot_all()

toggle_button.on_clicked(on_toggle)
exp_button.on_clicked(on_exp_toggle)
check.on_clicked(func)

plot_all()
plt.show()
