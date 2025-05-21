# model_hp_spaces: classifier -> hp -> value space (NOTE: manually declared)

sgd_space = {
    'loss': ['log_loss', 'modified_huber', 'squared_error', 'huber', 'epsilon_insensitive', 'perceptron', 'hinge'],
    'penalty': ['l1', 'elasticnet', 'l2', None],
    'fit_intercept': [True, False],
    'shuffle': [True, False],
    'early_stopping': [True, False],
    'warm_start': [True, False],
    'average': [True, False],
    'n_jobs': [-1],
}

k_neighbors_space = {
    'n_neighbors': [1,2,3,4,5,6,7,8,9,10],
    'algorithm': ['auto','ball_tree','kd_tree','brute'],
    'leaf_size': [1,2,4,8,16,30,32,64],
    'weights': ['uniform','distance',None],
    'n_jobs': [-1],
}

# TODO: Add all other classifier search spaces