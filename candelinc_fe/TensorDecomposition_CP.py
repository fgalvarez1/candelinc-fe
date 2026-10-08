import numpy as np

import tensorly as tl
from tensorly.decomposition import parafac

class CP:
    def __init__(self,):
        pass



    def get_normalized_cp_decomposition(self, data_tensor, rank):
        # CP for initial guess
        weights, factors = parafac(data_tensor, rank=rank, verbose=False)
        spatial_cp = factors[0] # n_p x n_modes
        temp_cp    = factors[1]
        acq_cp     = factors[2]

        assert set(weights) == {1}

        norm_spatial = np.linalg.norm(spatial_cp, axis=0, keepdims=True)
        # norm_temp    = np.linalg.norm(temp_cp, axis=0, keepdims=True)
        norm_acq     = np.linalg.norm(acq_cp, axis=0, keepdims=True)

        normalized_spatial_cp = spatial_cp / norm_spatial
        # normalized_temp_cp    = temp_cp / norm_temp
        scaled_temp_cp        = temp_cp * norm_spatial * norm_acq
        normalized_acq_cp     = acq_cp / norm_acq

        # Determine descending order based on column norms of scaled_temp_cp
        col_norms = np.linalg.norm(scaled_temp_cp, axis=0)
        sort_idx = np.argsort(col_norms)[::-1]

        normalized_spatial_cp = normalized_spatial_cp[:, sort_idx]
        scaled_temp_cp        = scaled_temp_cp[:, sort_idx]
        normalized_acq_cp     = normalized_acq_cp[:, sort_idx]

        X_cp   = tl.cp_to_tensor((weights, factors))
        mse_cp = np.mean((X_cp - data_tensor)**2) / np.mean(data_tensor**2)
        print(f"NMSE with CP: {mse_cp:.4f}")

        normalized_cp = {"normalized_spatial_cp" : normalized_spatial_cp,
                         "scaled_temp_cp"        : scaled_temp_cp,
                         "normalized_acq_cp"     : normalized_acq_cp}
        
        return normalized_cp, mse_cp