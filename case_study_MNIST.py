from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import numpy as np
import idx2numpy
from sklearn.metrics import accuracy_score

##### HELPER #####

def model_training(model, X_train, y_train, X_test, y_test):
    # Train and Predict
    model[1].fit(X_train, y_train)
    y_pred = model[1].predict(X_test)

    # LOGGING
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Configuration {model[0]}: {model[1].get_params()} - Accuracy: {accuracy}")

    return model[0], accuracy

####################

# Case study using the MNIST data set

# --- Preparation/Preprocessing ---

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

# --- Model Training (for each Hyperparameter config) ---

# RQ1: Sklearn performance measurements

# TESTING BY ONLY CONSIDERING SGD
if False:
    from scripts.io.mnist.generated_sklearn_estimators import estims # import my generated_estimators

    estims_evaluated = [] # store tuple: (model_id, accuracy)

    for estimator in estims:
        #model_training(estimator, X_train, y_train, X_test, y_test)
        estims_evaluated.append(model_training(estimator, X_train, y_train, X_test, y_test))

# RQ2: Hyperopt-sklearn optimized models (with restricted search space)
if True:
    from hpsklearn import HyperoptEstimator, k_neighbors_classifier, sgd_classifier
    from hyperopt import hp, tpe
    from scripts.io.mnist.hyperopt_spaces import *
    
    # %%% Models %%%

    sgd_search_space = {
        'loss': hp.choice('loss', sgd_space['loss']),
        'penalty': hp.choice('penalty', sgd_space['penalty']),
        'fit_intercept': hp.choice('fit_intercept', sgd_space['fit_intercept']),
        'shuffle': hp.choice('shuffle', sgd_space['shuffle']),
        'early_stopping': hp.choice('early_stopping', sgd_space['early_stopping']),
        'warm_start': hp.choice('warm_start', sgd_space['warm_start']),
        'average': hp.choice('average', sgd_space['average']),
        'n_jobs': hp.choice('n_jobs', sgd_space['n_jobs']),
    }

    # TODO: Add all other classifier search spaces

    # %%% Training %%%
    search_space = sgd_search_space # TODO: Enter the search space of the classifier you want to use 

    estim = HyperoptEstimator(
        algo=tpe.suggest,
        max_evals=3,
        #trial_timeout=300,
        classifier=sgd_classifier('clf', **search_space),
        preprocessing=[]
    )

    estim.fit(X_train, y_train)
    y_pred = estim.predict(X_test)

    # LOGGING
    best_model = estim.best_model()
    print("Best Model:", best_model)
    print("Hyperopt-Sklearn Accuracy Score:", accuracy_score(y_test, y_pred))

    # TODO: Inwieweit macht es überhaupt Sinn hyperopt für diesen (sehr stark) limitierten search space zu verwenden?