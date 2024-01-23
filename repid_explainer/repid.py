import numpy as np
import pandas as pd
from utils import *

class Node():
    
    def __init__(self, 
                 id=None,
                 depth=None, 
                 subset_idx=None,
                 obj_val=None,
                 obj_val_parent=None,
                 parent_id=None,
                 child_type=None,
                 split_feature=None,
                 split_val=None,
                 children=[],
                 stop_criteria_met=False,
                 improvement_met=False,
                 intImp=None):
        self.id = id
        self.depth = depth
        self.subset_idx = subset_idx
        self.obj_val = obj_val
        self.obj_val_parent = obj_val_parent
        self.parent_id = parent_id
        self.child_type = child_type
        self.split_feature = split_feature
        self.split_val = split_val
        self.children = children
        self.stop_criteria_met = stop_criteria_met
        self.improvement_met = improvement_met
        self.intImp = intImp
        
    def computesplit(
        self, 
        data: np.ndarray,
        ice_curve: np.ndarray,
        objective: callable,
        gamma: float,
        min_split_size: int = 10
    ) -> None:
        
        # check stop criteria
        if (len(self.subset_idx) < min_split_size) | self.improvement_met:
            self.stop_criteria_met = True
            return None
        
        self.obj_val_parent = objective(ice_curve) # objective value of the root node
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