# Experiment Main Skript Part 1 (RQ1): Collect sklearn classifier performance measurements

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import numpy as np
import idx2numpy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, fbeta_score, hamming_loss, zero_one_loss
import click
import json
import os
import ast

"""
Bachelor Thesis: Case study using the MNIST data set
    RQ1: Collect sklearn performance measurements

Description:
    This script generates an estimator for each allowed/predefined configuration of an sklearn classifier. It will be called during the evaluation pipeline of the VaRA Tool-Suite, providing each hyperparameter configuration for a specified sklearn estimator. It trains and predicts each model, in order to finally collect all performance measurements in an output file.

Example usage (How to):
    Type in the following command in the terminal to run the script: (inside the root directory of the repo) (Note: The estimator class and params must be adjusted according to your needs)
python case_study_MNIST.py -config_id=0 --average --early_stopping --fit_intercept --loss --huber --n_jobs ---1 --penalty --None --warm_start
python case_study_MNIST.py -config_id=1 --loss --huber --n_jobs ---1 --penalty --None --warm_start
"""

# MANUAL CONFIGURATION -> For each classifier, change the estimator class and its case_study output json file. TODO: Maybe create a revision for each, then use the different revisions in the evaluation pipeline.
estimator = "sklearn.linear_model.SGDClassifier" # Example: sklearn.linear_model.SGDClassifier
output = "scripts/io/mnist/cs_output/cs00_measurements.json" # Example: scripts/io/mnist/cs_output/cs00_measurements.json
#config_id = 0 # TODO: Provide the current config ID as a cmd line argument, additionally to all parameters of the config?
# TODO: After clarifying the two Todos above, push to the repo.

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
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted') # weighted ?
    recall = recall_score(y_test, y_pred, average='weighted') # weighted ?
    f1 = f1_score(y_test, y_pred, average='weighted') # weighted ?
    #conf_matrix = confusion_matrix(y_test, y_pred)
    fbeta = fbeta_score(y_test, y_pred, beta=0.5, average='weighted') # weighted ?
    hamming = hamming_loss(y_test, y_pred)
    zero_one = zero_one_loss(y_test, y_pred)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        #'confusion_matrix': conf_matrix.tolist(),  # Convert to list for JSON serialization
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
    #print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"F-beta Score: {fbeta}")
    print(f"Hamming Loss: {hamming}")
    print(f"Zero-One Loss: {zero_one_loss(y_test, y_pred)}")
    print("\n######################################################\n")

    return model, metrics

@click.command(context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True,
))
#@click.option('--estimator', required=True, help='Full class path, e.g. sklearn.linear_model.SGDClassifier')
@click.option('-config_id', required=True, type=int, help='Configuration ID for this run')
#@click.option('--params', required=True, help='Parameter list as a Python literal, e.g. \'["loss", "hinge", "n_jobs", "-1", "penalty", "l1"]\'')
#@click.option('--output', required=True, help='Path to output file for results')
@click.pass_context
def main(ctx, config_id):
    ### MODEL TRAINING (for each hyperparameter config) ###

    # Dynamically import estimator class
    module_name, class_name = estimator.rsplit('.', 1)
    module = __import__(module_name, fromlist=[class_name])
    EstClass = getattr(module, class_name)
    default = EstClass()
    default_params = default.get_params()

    # Parse all hyperparameters (and for a non-binary hp its respective value)
    param_args = ctx.args
    est_kwargs = {}
    seen_keys = set()
    i = 0
    while i < len(param_args):
        # Prepare the argument or skip if it is not a valid parameter
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

    # Save the final model and its performance to the output file (-> NOTE: same filepath as stated in sklearn project from my oot TS repo)
    # NOTE: Might exclude estimator and params later. See what is really needed in the end.
    result = {
        "estimator": estimator,
        "params": performance[0].get_params(),
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