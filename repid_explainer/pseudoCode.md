IMPORT required libraries

DEFINE R6 Class Node with following attributes and methods:

    Attributes:
        - ID of the node
        - Depth of the node in the tree
        - Subset of data indices associated with this node
        - Objective value at the node and its parent
        - Information about the parent (ID and child type)
        - Split information (splitting feature and value)
        - List of children of this node
        - Flags indicating if stopping criterion and improvement thresholds are met
        - Integrated improvement score

    Methods:
        - Initialize: Setup a new node with given parameters
        - computeSplit: 
            * Calculate objective value and compare with predefined criteria to decide if a split is possible.
            * If split is possible, determine the feature and value for the split using an optimization function.
        - computeChildren: 
            * Create left and right child nodes based on the split determined in computeSplit.
            * Add children to current node's children list.

DEFINE function compute_tree that:
    
    Based on the objective specified:
        - Choose a specific function for splitting nodes
        - Compute input data for ICE splitting
    
    Initialize the parent node of the tree.

    Loop through each depth level (up to n.split times):
        - For each leaf node at the current depth:
            * Compute a split for the leaf node if feasible
            * If split computed, create children (left and right) for the leaf node
        - Add the children to the tree structure for the next depth

    Return the constructed tree structure.

DEFINE function compute_data_for_ice_splitting that:

    * Takes in an effect object and test data.
    * Processes data to return a structured dataset where each row corresponds to an ICE curve.
