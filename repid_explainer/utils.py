from sklearn.inspection import partial_dependence
from sklearn.base import BaseEstimator
from typing import Tuple
import pandas as pd
import numpy as np

def generate_ice(
    model: BaseEstimator,
    X: pd.DataFrame,
    feature: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate ICE curve from the given machine learning model and the chosen feature (for x axis)

    Args:
        model (BaseEstimator): sklearn model object (already fit)
        X (pd.DataFrame): the original data for fitting the sklearn model
        feature (str): the selected feature for ICE curve

    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - ice_curve (np.ndarray): values of ICE curve where each row corresponds to each sample
                                      and each column corresponds to each grid value
            - grid_values (np.array): grid values for the selected feature on ICE curve
    """
    # get ice curve object from sklearn
    ice_data = partial_dependence(model, X, feature, kind="individual")
    
    # get actual ice curves and its x values
    grid_values = ice_data["grid_values"]
    ice_curve = ice_data["individual"][0]
    
    # mean centered ice cruves
    ice_curve = ice_curve - np.mean(ice_curve, axis=1, keepdims=True)
    
    return (ice_curve, grid_values)