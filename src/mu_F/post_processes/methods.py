"""Regressor data-processing helpers for post-process estimators."""


def estimator_regressor_data_function(candidates, constraints, desired_node_index):
    """
    Process regressor data for a convex estimator: builds the input/output
    pairs used to fit the estimator at the desired node.
    """
    inputs = candidates[:,:-1]
    outputs = (candidates[:,-1].reshape(-1,1) - constraints[desired_node_index].reshape(-1,1))**2
    return inputs, outputs



post_process_regressor_data_function = {"tablet_press": lambda x, y, z: (x, y),
"serial_mechanism_batch": lambda x, y, z: (x, y),
"convex_estimator": estimator_regressor_data_function,
"convex_underestimator": estimator_regressor_data_function,
"estimator": estimator_regressor_data_function,
"affine_study": lambda x, y, z: (x, y)
}