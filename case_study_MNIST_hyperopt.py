# Experiment Main Skript Part 2 (RQ2): Hyperopt-sklearn optimizes models (with restricted search space)

"""
Example usage (How to):
python case_study_MNIST_hyperopt.py --estimator {mnb}
"""

from hpsklearn import HyperoptEstimator, multinomial_nb, decision_tree_classifier, gradient_boosting_classifier, sgd_classifier, k_neighbors_classifier, random_forest_classifier
from sklearn.metrics import accuracy_score
from hyperopt import hp, tpe, rand
from scripts.io.mnist.hyperopt_spaces import *
from case_study_MNIST import load_mnist_data
import pandas as pd
import argparse
import json
import os
import time

# %%% Preparation %%%

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--estimator", choices=["mnb", "dtc", "gbc", "sgd", "knn", "rfc"], help="Set the sklearn estimator that hyperopt-sklearn should optimize. Choose: ['mnb', 'dtc', 'gbc', 'sgd', 'knn', 'rfc']", required=True)
args = parser.parse_args()
output_path = "scripts/io/mnist/hyperopt_output/hyperopt_results_10%.json"

# Setup hyperopt-sklearn estim
algorithm = tpe.suggest # tpe.suggest / rand.suggest
algorithm_str = "tpe.suggest" # tpe.suggest / rand.suggest
evals = 3 # 5%, 10%
timeout = 1800 # 300, 600

# Classifier -> Search Space
clf_search_space = {
    'mnb': multinomial_nb('clf', **mnb_search_space),
    'dtc': decision_tree_classifier('clf', **dtc_search_space),
    'gbc': gradient_boosting_classifier('clf', **gbc_search_space),
    'sgd': sgd_classifier('clf', **sgd_search_space),
    'knn': k_neighbors_classifier('clf', **knn_search_space),
    'rfc': random_forest_classifier('clf', **rfc_search_space),
}
clf = clf_search_space[args.estimator]

# Logging
print(f"-----| Hyperopt-sklearn started: {args.estimator}.")

# %%% Training %%%

estim = HyperoptEstimator(
    algo=algorithm,
    max_evals=evals,
    trial_timeout=timeout,
    classifier=clf,
    preprocessing=[],
    seed=42,
)

X_train, y_train, X_test, y_test = load_mnist_data()

# Logging
start_time = time.time()
print("Starting training ...")

# Training
estim.fit(X_train, y_train)

# Logging
elapsed = time.time() - start_time
print(f"Training completed in {elapsed:.2f} seconds")

# Predicting
y_pred = estim.predict(X_test)

# %%% Logging %%%

print(f"-----| Hyperopt-sklearn finished: {args.estimator}.")

# Print results:
best_model = estim.best_model()
parameters = best_model['learner'].get_params()
accuracy = accuracy_score(y_test, y_pred)
print(f"Hyperopt-sklearn estim: algo={algorithm_str}, max_evals={str(evals)}, trial_timeout={str(timeout)}")
print("Best model parameters:", parameters)
print("Accuracy:", accuracy)

# Save the results to JSON file with dictionary structure:
results_dict = {}
if os.path.exists(output_path):
    try:
        with open(output_path, 'r') as file:
            results_dict = json.load(file)
    except json.JSONDecodeError:
            print("Warning: Could not parse existing JSON file, creating new one.")
    
# Add/update current classifier results
results_dict[args.estimator] = {
    'hyperopt_sklearn_estim': f"algo={algorithm_str}, max_evals={str(evals)}, trial_timeout={str(timeout)}",
    'best_model_parameters': parameters,
    'training_time_sec': round(elapsed, 2),
    'accuracy': accuracy,
}
    
# Write to JSON file
with open(output_path, 'w') as file:
    json.dump(results_dict, file, indent=2)

print(f"-----| Results saved to {output_path}.")