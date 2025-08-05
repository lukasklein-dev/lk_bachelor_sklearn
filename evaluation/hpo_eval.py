import pandas as pd
import json
import os

# TODO: SETUP
clf = "mnb"
sample_strategy = "diversified-distance" # diversified-distance or random
sample_size = 1
hyperopt_best_accuracy = 0.8357

csv_path = f"/home/lukas/Schreibtisch/repos/lk_bachelor_sklearn/evaluation/experiment_data/performance_influence_models/{sample_strategy}/{clf}/sample_size_{sample_size}/performance_predictions.csv"
json_path = "/home/lukas/Schreibtisch/repos/lk_bachelor_sklearn/evaluation/experiment_data/evaluation_results/hpo_results_top3_complete.json"

# Lade CSV
df = pd.read_csv(csv_path)

# Sortiere nach "Performance Prediction" und bei Gleichstand nach "Performance", beide absteigend
top_pred_df = df.sort_values(["Performance Prediction", "Performance"], ascending=[False, False]).head(3)

# Speichere den sortierten DataFrame zur Kontrolle
top_pred_df.to_csv(
    "/home/lukas/Schreibtisch/repos/lk_bachelor_sklearn/evaluation/experiment_data/evaluation_results/top3_performance_predictions.csv",
    index=False
)

# Listen extrahieren
best_predicted = top_pred_df["Performance Prediction"].tolist()
true_performance = top_pred_df["Performance"].tolist()

# Berechne Differenz: positiv, wenn unser Ansatz besser ist als hyperopt-sklearn
best_true_acc = max(true_performance)
diff = best_true_acc - hyperopt_best_accuracy

# Key für die JSON-Struktur
main_key = f"{clf}/{sample_strategy}/{sample_size}"

# Lade ggf. bestehende Ergebnisse
if os.path.exists(json_path):
    with open(json_path, "r") as f:
        try:
            all_results = json.load(f)
        except json.JSONDecodeError:
            all_results = {}
else:
    all_results = {}

# Update/Setze die Ergebnisse für diesen Lauf
all_results[main_key] = {
    "best_predicted": best_predicted,
    "true_performance": true_performance,
    "best_true_accuracy": best_true_acc,
    "hyperopt_best_accuracy": hyperopt_best_accuracy,
    "accuracy_difference": round(diff, 4)
}

# Schreibe die aktualisierte JSON zurück
with open(json_path, "w") as f:
    json.dump(all_results, f, indent=2)

# Ausgabe zur Kontrolle
print("-----| Logging |-----")
print("Best predicted accuracies:", best_predicted)
print("True performance for these configs:", true_performance)
print(f"Best true accuracy: {best_true_acc}")
print(f"Hyperopt-sklearn best accuracy: {hyperopt_best_accuracy}")
print(f"Difference (best_true_accuracy - hyperopt_best_accuracy): {round(diff, 4)}")
print(f"Results saved to {json_path}")