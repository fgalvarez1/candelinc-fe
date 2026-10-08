import numpy as np

class P1DesignMatrix:
    def __init__(self, var_nodes):
        self.var_nodes = var_nodes



    def build(self, var_values):
        """
        Constructs the design matrix D for P1 finite elements.

        var_values: 1D array of length A (observed values of certain variable, one per acquisition)
        var_nodes: 1D array of length K (finite element node locations)
        
        Returns:
        D: Array of shape (A, K)
        """
        A = len(var_values)
        K = len(self.var_nodes)
        D = np.zeros((A, K))
        
        for i, var_value in enumerate(var_values):
            # Handle edge cases (extrapolation if the value is outside the nodes)
            if var_value <= self.var_nodes[0].item():
                D[i, 0] = 1.0
            elif var_value >= self.var_nodes[-1].item():
                D[i, -1] = 1.0
            else:
                # Find which interval [node_left, node_right] the value falls into
                idx = np.searchsorted(self.var_nodes, var_value) - 1
                n_left, n_right = self.var_nodes[idx].item(), self.var_nodes[idx+1].item()
                
                # P1 (Linear) interpolation weights
                dist = n_right - n_left
                w_left = (n_right - var_value) / dist
                w_right = (var_value - n_left) / dist
                
                D[i, idx] = w_left
                D[i, idx + 1] = w_right
                
        return D