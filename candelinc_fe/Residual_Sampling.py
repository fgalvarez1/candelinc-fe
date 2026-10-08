import numpy as np

from .DesignMatrix_P1 import P1DesignMatrix
from .TensorDecomposition_CP import CP

class MultivariateNormal:
    def __init__(self, mean, covariance_matrix, seed=None):
        self.mean = mean
        self.covariance_matrix = covariance_matrix
        self.rng = np.random.default_rng(seed)

    def sample(self):
        return self.rng.multivariate_normal(self.mean, self.covariance_matrix)



class ResidualSampling:
    def __init__(self, X_observed, U_s, U_t, D, B, variables_combo_selected, nodes_dict, R_residuals=None, seed=None):
        self.X_observed = np.asarray(X_observed, dtype=float)
        self.U_s = np.asarray(U_s, dtype=float)
        self.U_t = np.asarray(U_t, dtype=float)
        self.D = D
        self.B = B
        self.variables_combo_selected = variables_combo_selected
        self.nodes_dict = nodes_dict
        self.R_residuals = R_residuals
        self.seed = seed

        # self.mvn, self.Residuals = self.get_MultiVarNormal_for_sampling_residuals_using_CandelincFE_modes()
        self.mvn, self.Residuals = self.get_MultiVarNormal_for_sampling_residuals_using_new_CP_modes(R_residuals=self.R_residuals)



    def get_D_new(self, var_dict_new):
        D_lst = []
        for var in self.variables_combo_selected:
            if var.count("-") == 1:
                var1, var2 = var.split("-")

                var1_values = var_dict_new[var1]
                var2_values = var_dict_new[var2]
                if var2 == "Position":
                    mask_SUP2 = (np.asarray(var2_values) == 0)
                    mask_PRO1 = (np.asarray(var2_values) == 1)

                    var1_SUP2 = np.asarray(var1_values)[mask_SUP2]
                    var1_PRO1 = np.asarray(var1_values)[mask_PRO1]

                    var1_SUP2_nodes  = self.nodes_dict[f"{var1}-SUP2"] 
                    var1_PRO1_nodes  = self.nodes_dict[f"{var1}-PRO1"]

                    D_var1_SUP2 = P1DesignMatrix(var1_SUP2_nodes).build(var1_values) * mask_SUP2.astype(float)[:, None]
                    D_var1_PRO1 = P1DesignMatrix(var1_PRO1_nodes).build(var1_values) * mask_PRO1.astype(float)[:, None]

                    D_lst.append(D_var1_SUP2)
                    D_lst.append(D_var1_PRO1)

                else:
                    raise ValueError("Not implemented yet")
            
            else:
                raise ValueError("Not implemented yet")
            
        D_new = np.concatenate(D_lst, axis=1)
        return D_new



    # def get_MultiVarNormal_for_sampling_residuals_using_CandelincFE_modes(self):
    #     # Sample from the same modes of the CANDELINC-FE, using residuals
    #     # Residuals
    #     V_s = self.U_s.T @ self.U_s
    #     V_t = self.U_t.T @ self.U_t
    #     V   = V_s * V_t
    #     M_a = np.einsum('sta,sr,tr->ar', self.X_observed, self.U_s, self.U_t, optimize=True)
    #     U_a_target = M_a @ np.linalg.inv(V)

    #     Residuals = (U_a_target - self.D @ self.B)
        
    #     # corr_matrix = np.corrcoef(P_res_acq_cp.T)
    #     # print(np.round(corr_matrix, decimals=2))
        
    #     mu  = np.mean(Residuals, axis=0)
    #     cov = np.cov(Residuals.T)

    #     mvn = MultivariateNormal(mean=mu, covariance_matrix=cov)

    #     return mvn, Residuals



    def get_MultiVarNormal_for_sampling_residuals_using_new_CP_modes(self, R_residuals):
        # Compute CP decomposition on the residuals to get new modes
        # Residuals are: original data - reconstructed data from the CANDELINC-FE

        # Recompute CP decomposition
        X_reduced_model = np.einsum('sr,tr,ar->sta', self.U_s, self.U_t, (self.D @ self.B), optimize=True)
        P_res           = self.X_observed - X_reduced_model

        normalized_cp_P_res, mse_cp_P_res = CP().get_normalized_cp_decomposition(data_tensor=P_res.copy(), rank=R_residuals)

        P_res_spatial_cp = normalized_cp_P_res["normalized_spatial_cp"]
        P_res_temp_cp    = normalized_cp_P_res["scaled_temp_cp"]
        P_res_acq_cp     = normalized_cp_P_res["normalized_acq_cp"]

        self.U_s_sampling = P_res_spatial_cp
        self.U_t_sampling = P_res_temp_cp
        
        # corr_matrix = np.corrcoef(P_res_acq_cp.T)
        # print(np.round(corr_matrix, decimals=2))
        
        mu  = np.mean(P_res_acq_cp, axis=0)
        cov = np.atleast_2d(np.cov(P_res_acq_cp.T))

        mvn = MultivariateNormal(mean=mu, covariance_matrix=cov, seed=self.seed)

        P_res_cp = (P_res_spatial_cp, P_res_temp_cp, P_res_acq_cp, mse_cp_P_res)

        return mvn, P_res_cp



    def generate_synthetic_field(self, D_new, coeffs = None):
        P_data_based = np.einsum('sr,tr,r->st', self.U_s, self.U_t, (D_new @ self.B).flatten(), optimize=True)
        # P_sample     = np.einsum('sr,tr,r->st', self.U_s, self.U_t, self.mvn.sample(), optimize=True)

        if coeffs is None: # Sample the coefficients from the multivariate normal distribution
            P_sample     = np.einsum('sr,tr,r->st', self.U_s_sampling, self.U_t_sampling, self.mvn.sample(), optimize=True)
        else:
            P_sample     = np.einsum('sr,tr,r->st', self.U_s_sampling, self.U_t_sampling, coeffs, optimize=True)

        P_synthetic  = P_data_based + P_sample
        return P_synthetic