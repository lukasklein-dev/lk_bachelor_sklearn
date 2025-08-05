import pandas as pd

# Dateipfade
sampled_path = "evaluation/experiment_data/performance_influence_models/div_dist_based/sample_size_10/sampled_configurations.csv"
measurements_path = "evaluation/experiment_data/performance_measurements/sklearn_experiment_dtc/measurements.csv"
output_path = "evaluation/experiment_data/performance_measurements/sklearn_experiment_dtc/measurements_sorted.csv"

# Einlesen der Header
with open(sampled_path, "r") as f:
    sampled_header = f.readline().strip().split(',')

# DataFrame laden
df = pd.read_csv(measurements_path)

# Performance-Spalte merken und entfernen
perf_col = "Performance"
if perf_col in df.columns:
    perf = df[perf_col]
    df = df.drop(columns=[perf_col])
else:
    perf = None

# Neue Spaltenreihenfolge (nur die, die auch in df sind)
new_order = [col for col in sampled_header if col in df.columns]
if perf is not None:
    df = df[new_order]
    df[perf_col] = perf
else:
    df = df[new_order]

# Speichern
df.to_csv(output_path, index=False)
print(f"Gespeichert als {output_path}")