"""Main module."""

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