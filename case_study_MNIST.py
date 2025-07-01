#!/usr/bin/env python3

# Experiment Main Skript Part 1 (RQ1): Collect sklearn classifier performance measurements

from sklearn.svm import SVC
import idx2numpy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score, hamming_loss, zero_one_loss
import click
import json
import os

"""
Bachelor Thesis: Case study using the MNIST data set
    RQ1: Collect sklearn performance measurements

Description:
    This script generates an estimator for each allowed/predefined configuration of an sklearn classifier. It will be called during the evaluation pipeline of the VaRA Tool-Suite, providing each hyperparameter configuration for a specified sklearn estimator. It trains and predicts each model, in order to finally collect all performance measurements in an output file.

Example usage (How to):
    Type in the following command in the terminal to run the script: (inside the root directory of the repo) (Note: The estimator class and params must be adjusted according to your needs)
python case_study_MNIST.py --estimator-[x] [--config_id, --params_1, --params_2, ...]
python case_study_MNIST.py --estimator-sgd --0 --average --early_stopping --fit_intercept --loss --huber --n_jobs ---1 --penalty --None --warm_start
python case_study_MNIST.py --estimator-sgd --1 --loss --huber --n_jobs ---1 --penalty --None --warm_start
case_study_MNIST.py --estimator-dtc 1 criterion entropy min_samples_split 2 splitter random
./case_study_MNIST.py --estimator-dtc 1 criterion entropy min_samples_split 2 splitter random
"""

# CONFIGURATION
output = "scripts/io/mnist/cs_output/perf_measurements.json" # Output file path for the performance measurements

estimator_modules = {
    '--estimator-sgd': "sklearn.linear_model.SGDClassifier",
    '--estimator-knn': "sklearn.neighbors.KNeighborsClassifier",
    '--estimator-svc': "sklearn.svm.SVC",
    '--estimator-rfc': "sklearn.ensemble.RandomForestClassifier",
    '--estimator-bgm': "sklearn.mixture.BayesianGaussianMixture",
    '--estimator-dtc': "sklearn.tree.DecisionTreeClassifier",
    '--estimator-etc': "sklearn.ensemble.ExtraTreesClassifier",
    '--estimator-gpc': "sklearn.gaussian_process.GaussianProcessClassifier",
    '--estimator-gbc': "sklearn.ensemble.GradientBoostingClassifier",
    '--estimator-mnb': "sklearn.naive_bayes.MultinomialNB",
}

def load_mnist_data():
    ### PREPARATION/PREPROCESSING ###

    # Load MNIST
    X_train = './data/MNIST/train-images.idx3-ubyte'
    y_train = './data/MNIST/train-labels.idx1-ubyte'
    X_test = './data/MNIST/t10k-images.idx3-ubyte'
    y_test = './data/MNIST/t10k-labels.idx1-ubyte'

    # Converting IDX files to numpy arrays
    X_train = idx2numpy.convert_from_file(X_train)
    y_train = idx2numpy.convert_from_file(y_train)
    X_test = idx2numpy.convert_from_file(X_test)
    y_test = idx2numpy.convert_from_file(y_test)

    # Flatten the images - Reshape the 3d array (60000, 28, 28) into a 2d array (60000, 28 x 28)
    X_train = X_train.reshape(-1, 28 * 28)
    X_test = X_test.reshape(-1, 28 * 28)

    # Normalize the data - 255 since it's a 8-bit gray-scale, resulting in [0;1]
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    return X_train, y_train, X_test, y_test

def model_training(model, X_train, y_train, X_test, y_test):
    ### TRAIN AND PREDICT ###
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    ### METRICS ###
    # NOTE: Discuss which metrics to use in the end.
    # Only store four floating point numbers (e.g. 0.1234) in the output file.
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted').round(4) # weighted ?
    recall = recall_score(y_test, y_pred, average='weighted').round(4) # weighted ?
    f1 = f1_score(y_test, y_pred, average='weighted').round(4) # weighted ?
    fbeta = fbeta_score(y_test, y_pred, beta=0.5, average='weighted').round(4) # weighted ?
    hamming = hamming_loss(y_test, y_pred).round(4)
    zero_one = zero_one_loss(y_test, y_pred).round(4)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'fbeta_score': fbeta,
        'hamming_loss': hamming,
        'zero_one_loss': zero_one
    }

    ### LOGGING ###
    print(f"Estimator {model.get_params()}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"F-beta Score: {fbeta}")
    print(f"Hamming Loss: {hamming}")
    print(f"Zero-One Loss: {zero_one_loss(y_test, y_pred)}")
    print("\n########################################\n")

    return model, metrics

@click.command(context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True,
))
@click.pass_context
def main(ctx):
    ### MODEL TRAINING (for each hyperparameter config) ###

    # First argument is the estimator, e.g. --estimator-sgd
    if len(ctx.args) < 1 or not ctx.args[0].startswith('--estimator-'):
        raise click.UsageError("You must specify an estimator, e.g. --estimator-sgd")
    estimator = ctx.args[0]
    if estimator not in estimator_modules:
        raise click.UsageError(f"Unknown estimator: {estimator}. Allowed estimators are: {', '.join(estimator_modules.keys())}")
    # Second argument is the config_id, e.g. 0 for configuration ID 0
    if len(ctx.args) < 2:
        raise click.UsageError("You must specify a configuration ID, e.g. 0")
    config_id = ctx.args[1]

    # Dynamically import estimator class
    module_name, class_name = estimator_modules[estimator].rsplit('.', 1)
    module = __import__(module_name, fromlist=[class_name])
    EstClass = getattr(module, class_name)
    default = EstClass()
    default_params = default.get_params()

    # Parse all hyperparameters (and for a non-binary hp its respective value)
    param_args = ctx.args
    est_kwargs = {}
    seen_keys = set()
    i = 2  # Start after estimator and config_id
    while i < len(param_args):
        # Prepare the argument or skip if it is not a valid parameter (only the estimator starts with '--', but for safety we still check for it):
        arg = param_args[i]
        if arg.startswith('--'):
            arg = arg[2:]
        if arg not in default_params:
            i += 1
            continue

        default_val = default_params[arg]
        seen_keys.add(arg)
        # For binary hyperparameters, handle bools as flags (present = True):
        if isinstance(default_val, bool):
            est_kwargs[arg] = True
            i += 1
        else:
            # For non-binary hyperparameters, read its selected value next in the list:
            if i + 1 < len(param_args):
                val = param_args[i + 1]
                if val.startswith('--'):
                    val = val[2:]
                # Convert string input to correct type
                if val == 'None':
                    est_kwargs[arg] = None
                elif val == 'True':
                    est_kwargs[arg] = True
                elif val == 'False':
                    est_kwargs[arg] = False
                elif isinstance(default_val, int) or isinstance(default_val, type(None)): # NOTE: default=None and value space: float, might be a problem.
                    try:
                        est_kwargs[arg] = int(val)
                    except ValueError:
                        est_kwargs[arg] = val
                elif isinstance(default_val, float):
                    try:
                        est_kwargs[arg] = float(val)
                    except ValueError:
                        est_kwargs[arg] = val
                else:
                    est_kwargs[arg] = val
                i += 2
            else:
                i += 1
    # Set all bools not seen to False
    for k, v in default_params.items():
        if isinstance(v, bool) and k not in seen_keys:
            est_kwargs[k] = False

    # Create and train the estimator instance with the parsed parameters
    estimator_instance = EstClass(**est_kwargs)

    X_train, y_train, X_test, y_test = load_mnist_data()
    performance = model_training(estimator_instance, X_train, y_train, X_test, y_test)

    # Save the performance measurements of the estimator to the output file
    result = {
        #"estimator": estimator,
        #"params": performance[0].get_params(),
        "performance": performance[1]
    }
    
    # Load existing results if file exists
    if os.path.exists(output):
        with open(output, "r") as f:
            all_results = json.load(f)
    else:
        all_results = {}

    all_results[str(config_id)] = result

    with open(output, "w") as f:
        json.dump(all_results, f, indent=2)
    
if __name__ == "__main__":
    main()