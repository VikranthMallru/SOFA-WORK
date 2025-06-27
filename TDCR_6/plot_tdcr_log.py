import csv
import matplotlib.pyplot as plt

csv_path = "/home/ci/my_files/TDCR_6/CSV_Plots/tdcr_log.csv"

displacements = []
avg_forces = []

with open(csv_path, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            disp = float(row["displacement_cable1"])
            force = float(row["avg_force"])
            displacements.append(disp)
            avg_forces.append(force)
        except (ValueError, KeyError):
            continue

plt.figure(figsize=(8, 5))
plt.plot(displacements, avg_forces, marker='o', linestyle='-')
plt.xlabel("Displacement")
plt.ylabel("Average Force")
plt.title("Displacement vs Average Force")
plt.grid(True)
plt.tight_layout()
plt.show()