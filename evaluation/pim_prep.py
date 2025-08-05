import os
import json
import pandas as pd
import re
import ast
import argparse

# Argumente parsen
parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true", help="Speichere auch die config_id-Spalte")
args = parser.parse_args()

# Paths: all folder (-> alle measurements vom Cluster), sampled_configurations.csv (-> vara-feature ./case_study & sample_size={ALL}), configurations.yml (-> vara-feature ./confg-generator) anpassen
clf = "gbc" # TODO
sampled_csv = f"evaluation/experiment_data/performance_measurements/sklearn_experiment_{clf}/sampled_configurations.csv"
configs_path = f"evaluation/experiment_data/performance_measurements/sklearn_experiment_{clf}/configurations.yml"
all_dir = f"evaluation/experiment_data/performance_measurements/sklearn_experiment_{clf}/all"
output_csv = f"evaluation/experiment_data/performance_measurements/sklearn_experiment_{clf}/sampled_configurations_performance.csv"
logging_file = f"evaluation/experiment_data/performance_measurements/sklearn_experiment_{clf}/log.txt"

# Lade CSV
df = pd.read_csv(sampled_csv)
header = list(df.columns)
header_ohne_erstes = header[1:]  # z.B. ohne "decision_tree_classifier"

# Check auf doppelte Zeilen
duplicates = df.duplicated()
if duplicates.any():
    print("Error: Es gibt doppelte Zeilen in sampled_configurations.csv!")
    print(df[duplicates])
else:
    print("Success: Keine doppelten Zeilen in sampled_configurations.csv gefunden.")

# Extrahiere alle Konfigurationen aus der .case_study
with open(configs_path, "r") as f:
    lines = f.readlines()

case_configs = []
case_ids = []
for line in lines:
    m = re.match(r'^\s*(\d+):\s+\'(\[.*\])\'', line)
    if m:
        cid = m.group(1)
        config_list = ast.literal_eval(m.group(2))
        case_ids.append(cid)
        case_configs.append(config_list)

# Hilfsfunktion: Konfigurations-String in 0/1-Vektor (als Liste von ints) umwandeln
def config_to_vector(config, header_ohne_erstes):
    # config: z.B. ["0", "criterion", "entropy", "min_samples_split", "2", "splitter", "best"]
    # header_ohne_erstes: alle Spaltennamen außer der ersten
    config_set = set(config)
    #print(config_set)
    #print(header_ohne_erstes)
    return [1 if str(h) in config_set else 0 for h in header_ohne_erstes]

# Alle measurement-Files merken
all_txt_files = set(f for f in os.listdir(all_dir) if f.endswith("_success.txt"))
used_txt_files = set()

# Performance-Spalte initialisieren
performance_col = [None] * len(df)
config_id_col = [None] * len(df)
missing_configs = set()

for cid, config in zip(case_ids, case_configs):
    vec = config_to_vector(config, header_ohne_erstes)
    #print(vec)
    # Suche Zeile in df, die exakt diesem Vektor entspricht (ohne erste Spalte)
    match_idx = None
    for idx, row in df.iterrows():
        row_vec = [int(row[h]) for h in header_ohne_erstes]
        if row_vec == vec:
            match_idx = idx
            break
    if match_idx is None:
        print(f"Warnung: Keine passende Zeile für case_study config {cid} gefunden!")
        continue

    # Measurement-File suchen
    pattern = f"_config-{cid}_success.txt"
    txt_files = [f for f in all_txt_files if f.endswith(pattern)]
    if not txt_files:
        print(f"Warnung: Kein measurement-File für config_id {cid} gefunden!")
        continue
    txt_file = txt_files[0]
    used_txt_files.add(txt_file)
    txt_path = os.path.join(all_dir, txt_file)
    with open(txt_path, "r") as f:
        data = json.load(f)
    cid_str = str(cid)
    if data == {} or cid_str not in data:
        print(f"Warnung: Leeres measurement-File für config_id {cid} gefunden! --- Measurement Error") # Falls ein config nicht richtig gemessen wurde
        missing_configs.add(cid)
        continue
    rep_dict = data[cid_str]
    accuracies = []
    for rep in rep_dict.values():
        perf = rep.get("performance", {})
        if "accuracy" in perf:
            accuracies.append(perf["accuracy"])
    if accuracies:
        avg_acc = round(sum(accuracies) / len(accuracies), 4)
        performance_col[match_idx] = avg_acc
        config_id_col[match_idx] = cid
    else:
        print(f"Warnung: Keine accuracy für config_id {cid} gefunden!")

# LOGGING
print("-------------------- Logging --------------------")
print(f"Anzahl der Konfigurationen in sampled_configurations.csv: {len(df)}")
print(f"Anzahl der Konfigurationen in configurations.yml: {len(case_configs)}")
print(f"Anzahl der measurement-Dateien im 'all' Ordner: {len(all_txt_files)}")
print("-------------------------------------------------")

# Check: Welche config_ids haben fehlerhafte oder fehlende all txt measurement files?
if missing_configs:
    print(f"Error: {len(missing_configs)} config_ids haben fehlerhafte oder fehlende measurement-Dateien (siehe logging.txt).")
    # Output der fehlenden config_ids in log.txt
    with open(logging_file, "w") as log_file:
        for cid in missing_configs:
            log_file.write(f"{cid}\n")
        prefix = f"sklearn_experiment_{clf}-python_projects@56f85d6917"
        for cid in missing_configs:
            log_file.write(f"'{prefix},{cid}'\n")
else:
    print("Success: Alle config_ids haben gültige measurement-Dateien.")

# Check: Welche CSV-Zeilen wurden nicht durch eine configurations.yml-Konfiguration abgedeckt?
used_rows = set(idx for idx in range(len(df)) if config_id_col[idx] is not None)
all_rows = set(range(len(df)))
not_covered = all_rows - used_rows

if not_covered:
    print(f"Error: {len(not_covered)} Konfigurationen aus sampled_configurations.csv (Zeilenindizes) wurden nicht korrekt abgearbeitet (Probleme beim matching oder durch eine fehlerhafte measurement Datei):")
    # Output der nicht abgedeckten Zeilenindizes in log.txt
    if False:
        with open(logging_file, "a") as log_file:
            log_file.write("Nicht abgedeckte Zeilenindizes:\n")
            for idx in sorted(not_covered):
                log_file.write(f"{idx}\n")
else:
    print("Success: Alle Konfigurationen aus sampled_configurations.csv wurden abgedeckt.")

# Nicht verwendete measurement-Dateien ausgeben
unused_txt_files = all_txt_files - used_txt_files
if unused_txt_files:
    print("Error: Nicht verwendete measurement-Dateien:")
    for fname in sorted(unused_txt_files):
        print(fname)
else:
    print("Success: Alle measurement-Dateien wurden verwendet.")

# Neue Spalten anhängen und speichern (je nach debug flag)
df["Performance"] = performance_col
if args.debug:
    df["config_id"] = config_id_col
    print("Debug-Modus: config_id-Spalte wird zusätzlich gespeichert.")
else:
    if "config_id" in df.columns:
        df = df.drop(columns=["config_id"])

df.to_csv(output_csv, index=False)
print(f"\nFertig! Gespeichert als {output_csv}")