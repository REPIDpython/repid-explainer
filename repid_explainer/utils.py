from sklearn.inspection import partial_dependence
from sklearn.base import BaseEstimator
from typing import Tuple, Union
import pandas as pd
import numpy as np


def SS_L2(ice_curve: np.ndarray) -> float:
    """calculate sum of L2 residual

    Args:
        ice_curve (np.ndarray): group of ice curves to calcualte total L2 loss

    Returns:
        float: total L2 loss
    """
    y_pred = np.mean(ice_curve, axis=0)
    sqr_diff = np.sum((ice_curve - y_pred)**2)
    return sqr_diff

def split_node(
    data: Union[np.ndarray, pd.DataFrame],
    ice_curve: np.ndarray,
    optimizer: callable,
    min_node_size: int = 10
) -> dict:
    """Compute the best split for the given ice curves

    Args:
        data (Union[np.ndarray, pd.DataFrame]): features that can be splitted on
        ice_curve (np.ndarray): collection of the current ice curves
        optimizer (callable): function to calcuate objective value
        min_node_size (int, optional): minimum size of each node after splitting. Defaults to 10.

    Returns:
        dict: output index of the best column, the splitting value, new overall objective value of the split
    """
    # in case we receive pd.DataFrame
    data = np.array(data)
    
    splits = np.apply_along_axis(find_best_split, 0, data, *[ice_curve, min_node_size, optimizer])
    
    # find minimum objective value
    obj_vals = [split[1] for split in splits]
    min_ind = obj_vals.index(min(obj_vals))
    
    return {
        "column_index": min_ind,
        "split_val": splits[min_ind][0],
        "new_tot_obj": splits[min_ind][1]
    }

def find_best_split(
    feature: np.ndarray,
    ice_curve: np.ndarray,
    min_node_size: int,
    objective: callable
) -> Tuple:
    """Given the selected feature and ice curve, find the best split (lowest objective value)

    Args:
        feature (np.ndarray): feature that is being splitted on
        ice_curve (np.ndarray): collection of the current ice curves
        min_node_size (int): minimum size of each node after splitting
        objective (callable): function to calcuate objective value

    Returns:
        Tuple: two elements 1). the best split point 2). the best objective value
    """
    # TODO: add option for categorical data
    candidates = generate_split_candidates_numeric(feature, n_quantiles=100, min_node_size=min_node_size)
    perform_split_vectorize = np.vectorize(perform_split)
    new_obj = perform_split_vectorize(candidates, feature, ice_curve, min_node_size, objective)
    
    # get split that has the minimum objective value
    min_ind = np.argmin(new_obj)
    return (candidates[min_ind], new_obj[min_ind])

def right_of_split(
    split_point: float,
    feature: np.ndarray
) -> np.ndarray:
    """Determine which index of data point is the right side of the splitting point

    Args:
        split_point (float): splitting point
        feature (np.ndarray): feature that is being splitted on

    Returns:
        np.ndarray: array of boolean (true for right side)
    """
    return np.array([(val > split_point) for val in feature])


def perform_split(
    split_point: float,
    feature: np.ndarray,
    ice_curve: np.ndarray,
    min_node_size: int,
    objective: callable
) -> float:
    """Perform split on the selected point to check the new objective value

    Args:
        split_point (float): splitting point we wish to split
        feature (np.ndarray): feature that is being splitted on
        ice_curve (np.ndarray): collection of the current ice curves
        min_node_size (int): minimum size of each node after splitting
        objective (callable): function to calcuate objective value

    Returns:
        float: new total objective value after the split
    """
    right_side = right_of_split(split_point, feature)
    
    # ignore invalide split by giving them inf cost
    if (sum(right_side) < min_node_size) or (sum(right_side) > len(feature) - min_node_size):
        return np.inf
    
    ice_right = ice_curve[right_side]
    ice_left = ice_curve[~right_side]
    
    # calcualte new objective value
    return objective(ice_left) + objective(ice_right)
    

def generate_split_candidates_numeric(
    data: np.ndarray,
    n_quantiles: int = None,
    min_node_size: int = 10
) -> np.ndarray:
    """Generate array of possible splitting points that other functions will try (for numerical data)

    Args:
        data (np.ndarray): Array of numerical data
        n_quantiles (int, optional): Number of quantiles users want to split data into. Defaults to None.
        min_node_size (int, optional): After splitting, each child node must have at least min_node_size in itself. Defaults to 10.

    Returns:
        np.ndarray: Possible splitting points to perform split on
    """
    
    # Check if all values are numeric
    all_numeric = np.all(np.array([np.issubdtype(x, np.number) for x in data]))
    if not all_numeric:
        raise ValueError("Input data must array of be float on int.")
    
    data.sort()
    
    # indices to split data that have min_node_size in each chuck (not guaranteed)
    idx = np.array(range(min_node_size, len(data) - min_node_size, min_node_size))
    
    split_point = data[idx]
    
    if isinstance(n_quantiles, type(None)):
        split_point = np.unique(split_point)
    else:
        # speed up the computation by choosing quantiles of the split_point
        qprobs = np.linspace(0, 1, num=n_quantiles+1)
        split_point = np.unique(np.percentile(split_point, qprobs * 100))

    return split_point

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

def node_to_split(node, 
                  max_obj_node: dict,
                  ice_curve: np.ndarray,
                  data: Union[np.ndarray, pd.DataFrame],
                  objective: callable,
                  gamma: float,
                  min_split_size: int = 10
) -> None:
    """Recursive function to find a node with the best improvement in nonsymmetric tree

    Args:
        node (Node): a node in REPID's nonsymmetric tree structure
        max_obj_node (dict): dictionary of current best node with the corresponding interaction importance (intTip)
        ice_curve (np.ndarray): collection of all ice curves
        data (Union[np.ndarray, pd.DataFrame]): features that can be splitted on (original features when training the model)
        objective (callable): function to calculate objective value
        gamma (float): stopping criteria factor (split only if intImp(split_nodes) >= gamma * intImp(parent_node))
        min_split_size (int, optional): minimum size of each node after splitting. Defaults to 10.
    """
    if node is not None:
        
        # aggregate all conditions
        condition = node.improvement_met \
                    | node.stop_criteria_met \
                    | (node.children["left"] is not None) \
                    | (node.childten["right"] is not None)
        
        # consider only nodes with no children and both improvement_met and stop_criteria_met are false        
        if not condition:
            node.computeSplit(data, ice_curve, objective, gamma, min_split_size=min_split_size)
            if node.intImp > max_obj_node["intImp"]:
                max_obj_node["intImp"] = node.intImp
                max_obj_node["node"] = node
            node.revertSplit()
            
        # recursively traverse the left and right children
        node_to_split(node.children["left"],
                      max_obj_node,
                      ice_curve,
                      data,
                      objective,
                      gamma,
                      min_split_size=min_split_size)
        node_to_split(node.children["right"],
                      max_obj_node,
                      ice_curve,
                      data,
                      objective,
                      gamma,
                      min_split_size=min_split_size)