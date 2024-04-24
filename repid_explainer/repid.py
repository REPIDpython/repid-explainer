import numpy as np
import pandas as pd
from utils import *

class Node():
    
    def __init__(self, 
                 depth=None, 
                 subset_idx=None,
                 obj_val=None,
                 obj_val_root=None,
                 child_type=None,
                 split_feature=None,
                 split_val=None,
                 children={"left": None, "right": None},
                 stop_criteria_met=False,
                 improvement_met=False,
                 intImp=None):
        self.depth = depth
        self.subset_idx = subset_idx
        self.obj_val = obj_val
        self.obj_val_root = obj_val_root
        self.child_type = child_type
        self.split_feature = split_feature
        self.split_val = split_val
        self.children = children
        self.stop_criteria_met = stop_criteria_met
        self.improvement_met = improvement_met
        self.intImp = intImp
        
    def computeSplit(
        self, 
        data: Union[np.ndarray, pd.DataFrame],
        ice_curve: np.ndarray,
        objective: callable,
        gamma: float,
        min_split_size: int = 10
    ) -> None:
        """Compute the best split (if plausible) given the current node

        Args:
            data (Union[np.ndarray, pd.DataFrame]): features that can be splitted on (original features when training the model)
            ice_curve (np.ndarray): collection of ice curves
            objective (callable): function to calculate objective value
            gamma (float): stopping criteria factor (split only if intImp(split_nodes) >= gamma * intImp(parent_node))
            min_split_size (int, optional): minimum size of each node after splitting. Defaults to 10.
        """
        
        # check stop criteria
        if (len(self.subset_idx) < min_split_size) | self.improvement_met:
            self.stop_criteria_met = True
            return None
        
        self.obj_val_root = objective(ice_curve)
        self.obj_val = objective(ice_curve[self.subset_idx])
        
        # perform splitting
        split = split_node(data[self.subset_idx],
                           ice_curve,
                           find_best_split,
                           min_node_size=min_split_size)
        
        if isinstance(self.intImp, type(None)):
            self.intImp = 0
        
        # calculate interaction importance and the threshold (stopping criteria)
        intImp = (self.obj_val - split["new_tot_obj"]) / self.obj_val_parent
        threshold = gamma if self.intImp == 0 else self.intImp * gamma
        
        # store split information if improvement is bigger than the threshold
        if intImp < threshold:
            self.improvement_met = True
        else:
            self.split_feature = split["column_index"]
            self.split_val = split["split_val"]
            self.intImp = intImp
            self.obj_val_parent = self.obj_val
            self.obj_val = split["new_tot_obj"]
        
        return None
    
    def revertSplit(self) -> None:
        """
        Revert any change made in the computeSplit method.
        """
        self.stop_criteria_met = False
        self.improvement_met = False
        self.split_feature = None
        self.split_val = None
        self.intImp = None
        self.obj_val = None
        self.obj_val_root = None
        self.obj_val_parent = None
    
    def computeChildren(
        self,
        data: Union[np.ndarray, pd.DataFrame]
    ) -> None:
        """Store information for children node

        Args:
            data (Union[np.ndarray, pd.DataFrame]): features that can be splitted on (original features when training the model)

        Raises:
            ValueError: to add information on children, program must have splitted the node first
        """
        
        # children remain as None if the stopping criteria are met
        if self.improvement_met | self.stop_criteria_met:
            return None
        
        # must have splitting information to move forward
        if self.split_feature is None:
            raise ValueError("Please compute the split first via computeSplit().")
        
        # get index of left and right children
        ind_left = np.where(np.array(data)[:, self.split_feature] <= self.split_val)[0]
        ind_right = np.where(np.array(data)[:, self.split_feature] > self.split_val)[0]

        # construct left and right children
        left_child = Node(depth=self.depth + 1,
                        subset_idx=ind_left,
                        child_type="left", 
                        improvement_met=self.improvement_met,
                        intImp=self.intImp,
                        stop_criteria_met=self.stop_criteria_met)
        right_child = Node(depth=self.depth + 1,
                        subset_idx=ind_right,
                        child_type="right", 
                        improvement_met=self.improvement_met,
                        intImp=self.intImp,
                        stop_criteria_met=self.stop_criteria_met)
        
        self.children = {"left": left_child,
                        "right": right_child}
        
        return None
   
    
class Repid():
    
    def __init__(self,
                 depth: int = 3,
                 n_split: int = 5,
                 method: str = "nonsymmetric",
                 intImp: float = 0.1
                 ) -> None:
        # check input validity
        if method not in ["symmetric", "nonsymmetric"]:
            raise ValueError("Argument 'method' must be either 'symmetric' or 'nonsymmetric'.")
        
        if not isinstance(n_split, int):
            raise ValueError("Argument 'n_split' must be a positive integer.")
        
        if not isinstance(depth, int):
            raise ValueError("Argument 'depth' must be a positive integer.")
        
        if n_split < 1:
            raise ValueError("Argument 'n_split' must be a positive integer.")
        
        if depth < 1:
            raise ValueError("Argument 'depth' must be a positive integer.")
        
        if not isinstance(intImp, (int, float)):
            raise ValueError("Argument 'intImp' must be a positive real number or zero.")
        
        if intImp < 0:
            raise ValueError("Argument 'intImp' must be a positive real number or zero.")
        
        self.depth = depth
        self.n_split = n_split
        self.method = method
        self.intImp = intImp
        
    def fit(self,
            model,
            data: Union[np.ndarray, pd.DataFrame],
            feature: Union[str, int],
            gamma: float = 0.1,
            min_split_size: int = 10,
            categorical_features: list = None):
        
        # get ice_curve
        ice_curve, grid_values = generate_ice(model, data, feature)
        
        # create root node
        root = Node(depth=self.depth,
                    subset_idx=np.arange(len(data)),
                    intImp=self.intImp)
        
        if self.method == "nonsymmetric":
            # use n_split for nonsymmetric method
            for split in range(self.n_split):
                max_obj_node = {"node": None, "intImp": -np.inf}
                node_to_split(root,
                              max_obj_node,
                              ice_curve,
                              data,
                              SS_L2,
                              gamma,
                              min_split_size=min_split_size)
        elif self.method == "symmetric":
            pass
        else:
            raise ValueError("Argument 'method' must be either 'symmetric' or 'nonsymmetric'.")