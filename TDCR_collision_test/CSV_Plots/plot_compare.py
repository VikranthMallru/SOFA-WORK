import pandas as pd
import matplotlib.pyplot as plt

# Load the two CSV files
no_object_df = pd.read_csv('data1.csv')
object_present_df = pd.read_csv('data2.csv')

# Calculate proper RMS for each row (root mean square of F1, F2, F3)
no_object_df['force_rms'] = (no_object_df[['F1', 'F2', 'F3']]**2).mean(axis=1)**0.5
object_present_df['force_rms'] = (object_present_df[['F1', 'F2', 'F3']]**2).mean(axis=1)**0.5

# Calculate difference in force RMS
force_rms_diff = object_present_df['force_rms'] - no_object_df['force_rms']

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(no_object_df['L1'], no_object_df['force_rms'], label='No Object')
plt.plot(object_present_df['L1'], object_present_df['force_rms'], label='Object Present')
plt.plot(no_object_df['L1'], force_rms_diff, label='Difference', linestyle='--')

plt.xlabel('L1')
plt.ylabel('Force RMS')
plt.title('L1 vs Force RMS Comparison (Proper RMS)')
plt.legend()
plt.grid(True)
plt.show()
