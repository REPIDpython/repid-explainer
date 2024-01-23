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
                 children=dict(),
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
    
    def computeChildren(
        self,
        data: Union[np.ndarray, pd.DataFrame]
    ) -> None:
        
        # store None as children if the stopping criteria are met
        if self.improvement_met | self.stop_criteria_met:
            self.children = {"left": None, 
                             "right": None}
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