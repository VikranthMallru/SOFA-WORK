import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- User-defined variables ---
DeltaLv = 15.0  
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

# Load the two CSV files
no_object_df = pd.read_csv('data1.csv')
object_present_df = pd.read_csv('data2.csv')

# Calculate RMS of forces for each row
no_object_df['force_rms'] = no_object_df[['F1', 'F2', 'F3']].apply(rms, axis=1)
object_present_df['force_rms'] = object_present_df[['F1', 'F2', 'F3']].apply(rms, axis=1)

# Calculate RMS of cosines (same for all rows, since alpha is fixed)
cos_rms = get_cos_rms(alpha)

# Calculate Fv for both dataframes
no_object_df['Fv'] = no_object_df['force_rms'] / cos_rms
object_present_df['Fv'] = object_present_df['force_rms'] / cos_rms

# Calculate difference (align on index)
min_len = min(len(no_object_df), len(object_present_df))
Fv_diff = object_present_df['Fv'].iloc[:min_len].values - no_object_df['Fv'].iloc[:min_len].values

# Plotting all in the same plot
plt.figure(figsize=(10, 6))
plt.plot(no_object_df.index, no_object_df['Fv'], label='No Object')
plt.plot(object_present_df.index, object_present_df['Fv'], label='Object Present')

# plt.plot(range(min_len), Fv_diff, label='Fv Difference (Object - No Object)', color='purple')

plt.xlabel('DeltaLv Step')
plt.ylabel('Fv / Fv Difference')
plt.title(f'Fv and Difference vs DeltaLv Step\nalpha = {alpha}°, DeltaLv = {DeltaLv}')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
