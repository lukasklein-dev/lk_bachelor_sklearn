# model_hp_spaces: classifier -> hp -> value space (NOTE: manually declared)

from hyperopt import hp

# %%% Feature Models %%%

mnb_fm = {
    'alpha': [0.01, 0.1, 0.5, 1.0, 10.0],
    'force_alpha': [True, False], # NOTE: not in hyperopt-sklearn
    'fit_prior': [True, False],
}

dtc_fm = {
    'criterion': ['gini', 'entropy', 'log_loss'],
    'splitter': ['best', 'random'],
    'min_samples_split': [2, 5, 10, 50],
    'random_state': [0, 1, 2, 3, 4],
}

gbc_fm = {
    'learning_rate': [0.05, 0.1, 0.3],
    'n_estimators': [50, 100, 500],
    'subsample': [0.7, 1.0],
    'criterion': ['friedman_mse', 'squared_error'],
    'min_samples_split': [2, 5, 10, 50],
    'warm_start': [True, False],
    'random_state': [0, 1, 2, 3, 4],
}

sgd_fm = {
    'loss': ['hinge', 'log_loss', 'modified_huber', 'squared_error', 'huber', 'epsilon_insensitive', 'perceptron'],
    'penalty': ['l2', 'l1', 'elasticnet', None],
    'fit_intercept': [True, False],
    'shuffle': [True, False],
    'early_stopping': [True, False],
    'warm_start': [True, False],
    'average': [True, False],
    'random_state': [0, 1, 2, 3, 4],
}

knn_fm = {
    'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'weights': ['uniform', 'distance', None],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'leaf_size': [1, 2, 4, 8, 16, 30, 32, 64],
}

rfc_fm = {
    'n_estimators': [100, 120, 150],
    'criterion': ['gini', 'entropy', 'log_loss'],
    'min_samples_split': [2, 3],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2', None],
    'class_weight': [None, 'balanced', 'balanced_subsample'],
    'bootstrap': [True],
    'oob_score': [True, False],
    'warm_start': [True, False],
    'random_state': [0, 1, 2, 3, 4],
}

# %%% Search Spaces (FM + default value, if HP not included) %%%

mnb_search_space = {
    'alpha': hp.choice('alpha', mnb_fm['alpha']),
    'fit_prior': hp.choice('fit_prior', mnb_fm['fit_prior']),
    #'force_alpha': hp.choice('force_alpha', mnb_fm['force_alpha']), # NOTE: not in hyperopt-sklearn
    # default:
    'class_prior': hp.choice('class_prior', [None]),
}

dtc_search_space = {
    'criterion': hp.choice('criterion', dtc_fm['criterion']),
    'splitter': hp.choice('splitter', dtc_fm['splitter']),
    'min_samples_split': hp.choice('min_samples_split', dtc_fm['min_samples_split']),
    'random_state': hp.choice('random_state', dtc_fm['random_state']),
    # default:
    'max_depth': hp.choice('max_depth', [None]),
    'min_samples_leaf': hp.choice('min_samples_leaf', [1]),
    'min_weight_fraction_leaf': hp.choice('min_weight_fraction_leaf', [0.0]),
    'max_features': hp.choice('max_features', [None]),
    'max_leaf_nodes': hp.choice('max_leaf_nodes', [None]),
    'min_impurity_decrease': hp.choice('min_impurity_decrease', [0.0]),
    'class_weight': hp.choice('class_weight', [None]),
    'ccp_alpha': hp.choice('ccp_alpha', [0.0]),
    #'monotonic_cst': hp.choice('monotonic_cst', [None]), # NOTE: not in hyperopt-sklearn
}

gbc_search_space = {
    'learning_rate': hp.choice('learning_rate', gbc_fm['learning_rate']),
    'n_estimators': hp.choice('n_estimators', gbc_fm['n_estimators']),
    'subsample': hp.choice('subsample', gbc_fm['subsample']),
    'criterion': hp.choice('criterion', gbc_fm['criterion']),
    'min_samples_split': hp.choice('min_samples_split', gbc_fm['min_samples_split']),
    'warm_start': hp.choice('warm_start', gbc_fm['warm_start']),
    'random_state': hp.choice('random_state', gbc_fm['random_state']),
    # default:
    'loss': hp.choice('loss', ['log_loss']),
    'min_samples_leaf': hp.choice('min_samples_leaf',[1]),
    'min_weight_fraction_leaf': hp.choice('min_weight_fraction_leaf', [0.0]),
    'max_depth': hp.choice('max_depth', [3]),
    'min_impurity_decrease': hp.choice('min_impurity_decrease', [0.0]),
    'init': hp.choice('init', [None]),
    'max_features': hp.choice('max_features', [None]),
    'verbose': hp.choice('verbose', [0]),
    'max_leaf_nodes': hp.choice('max_leaf_nodes', [None]),
    'validation_fraction': hp.choice('validation_fraction', [0.1]),
    'n_iter_no_change': hp.choice('n_iter_no_change', [None]),
    'tol': hp.choice('tol', [0.0001]),
    'ccp_alpha': hp.choice('ccp_alpha', [0.0]),
}

sgd_search_space = {
    'loss': hp.choice('loss', sgd_fm['loss']),
    'penalty': hp.choice('penalty', sgd_fm['penalty']),
    'fit_intercept': hp.choice('fit_intercept', sgd_fm['fit_intercept']),
    'shuffle': hp.choice('shuffle', sgd_fm['shuffle']),
    'early_stopping': hp.choice('early_stopping', sgd_fm['early_stopping']),
    'warm_start': hp.choice('warm_start', sgd_fm['warm_start']),
    'average': hp.choice('average', sgd_fm['average']),
    'random_state': hp.choice('random_state', sgd_fm['random_state']),
    # default:
    'alpha': hp.choice('alpha', [0.0001]),
    'l1_ratio': hp.choice('l1_ratio', [0.15]),
    'max_iter': hp.choice('max_iter', [1000]),
    'tol': hp.choice('tol', [0.001]),
    'verbose': hp.choice('verbose', [0]),
    #'epsilon': hp.choice('epsilon', [0.1]), # NOTE: combination problem
    'n_jobs': hp.choice('n_jobs', [None]),
    'learning_rate': hp.choice('learning_rate', ['optimal']),
    'eta0': hp.choice('eta0', [0.0]),
    'power_t': hp.choice('power_t', [0.5]),
    #'validation_fraction': hp.choice('validation_fraction', [0.1]), # NOTE: combination problem
    'n_iter_no_change': hp.choice('n_iter_no_change', [5]),
    'class_weight': hp.choice('class_weight', [None]),
}

knn_search_space = {
    'n_neighbors': hp.choice('n_neighbors', knn_fm['n_neighbors']),
    'weights': hp.choice('weights', knn_fm['weights']),
    'algorithm': hp.choice('algorithm', knn_fm['algorithm']),
    'leaf_size': hp.choice('leaf_size', knn_fm['leaf_size']),
    # default:
    'p': hp.choice('p', [2]),
    'metric': hp.choice('metric', ['minkowski']),
    'metric_params': hp.choice('metric_params', [None]),
    'n_jobs': hp.choice('n_jobs', [None]),
}

rfc_search_space = {
    'n_estimators': hp.choice('n_estimators', rfc_fm['n_estimators']),
    'criterion': hp.choice('criterion', rfc_fm['criterion']),
    'min_samples_split': hp.choice('min_samples_split', rfc_fm['min_samples_split']),
    'min_samples_leaf': hp.choice('min_samples_leaf', rfc_fm['min_samples_leaf']),
    'max_features': hp.choice('max_features', rfc_fm['max_features']),
    'class_weight': hp.choice('class_weight', rfc_fm['class_weight']),
    'bootstrap': hp.choice('bootstrap', rfc_fm['bootstrap']),
    'oob_score': hp.choice('oob_score', rfc_fm['oob_score']),
    'warm_start': hp.choice('warm_start', rfc_fm['warm_start']),
    'random_state': hp.choice('random_state', rfc_fm['random_state']),
    # default:
    'max_depth': hp.choice('max_depth', [None]),
    'min_weight_fraction_leaf': hp.choice('min_weight_fraction_leaf', [0.0]),
    'max_leaf_nodes': hp.choice('max_leaf_nodes', [None]),
    'min_impurity_decrease': hp.choice('min_impurity_decrease', [0.0]),
    'n_jobs': hp.choice('n_jobs', [None]),
    'verbose': hp.choice('verbose', [0]),
    'ccp_alpha': hp.choice('ccp_alpha', [0.0]),
    'max_samples': hp.choice('max_samples', [None]),
    #'monotonic_cst': hp.choice('monotonic_cst', [None]), # NOTE: not in hyperopt-sklearn
}