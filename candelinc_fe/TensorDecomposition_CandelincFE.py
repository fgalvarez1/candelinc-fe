import numpy as np

from .DesignMatrix_P1 import P1DesignMatrix

class CandelincFE:
    def __init__(self, variables_names, variables_dict, K):
        self.variables_names = variables_names
        self.variables_dict = variables_dict
        self.K = K



    def generate_adaptive_nodes(self, var_values, min_acquisitions_per_node=3, max_nodes=10, min_nodes=2):
        """
        Generates an adaptive finite element mesh (nodes) based on data quantiles.
        
        Args:
            var_values (np.ndarray or list): The numerical data for the specific subgroup.
            min_acquisitions_per_node (int): The statistical evidence required per node.
            max_nodes (int): The absolute ceiling for complexity.
            min_nodes (int): The absolute floor for complexity.
            
        Returns:
            np.ndarray: A 1D array of strictly unique, sorted nodes.
        """
        # Ensure input is a float array
        var_values = np.asarray(var_values, dtype=float)
        
        N = len(var_values)
        assert N > 0, "The input data must contain at least one value."
            
        # Calculate number of nodes
        K = max(min_nodes, min(max_nodes, N // min_acquisitions_per_node))
        
        # Adaptive Spacing: Create evenly spaced percentiles
        q_vals = np.linspace(0.0, 1.0, K)
        
        # Map percentiles to actual clinical values
        nodes = np.quantile(var_values, q_vals)
        
        # Remove duplicates caused by tied data
        nodes = np.unique(nodes)
        
        assert len(nodes) > 1, "All acquisitions have the same value, cannot create a P1 mesh with a single node."
            
        return nodes



    def build_global_design_matrix(self):
        D_lst      = []
        nodes_dict = {}

        for var in self.variables_names:
            if var.count("-") == 2:
                var1, var2, var3 = var.split("-") # var2 and var3 are the categorical variables
                # Design matrix (A x K) for numerical variable
                var1_values = self.variables_dict[var1]
            
                # Get the full One-Hot status matrix (A x 2) for position, (A x 3) for group, etc.
                var2_values = self.variables_dict[var2]
                var3_values = self.variables_dict[var3]
                if var2 == "Position" and var3 == "Groupe":
                    # Position-Groupe
                    num_classes = len(set(var3_values))

                    mask_SUP2_healthy = (np.asarray(var2_values) == 0) & (np.asarray(var3_values) == 0)
                    mask_PRO1_healthy = (np.asarray(var2_values) == 1) & (np.asarray(var3_values) == 0)

                    var1_SUP2_healthy = np.asarray(var1_values)[mask_SUP2_healthy]
                    var1_PRO1_healthy = np.asarray(var1_values)[mask_PRO1_healthy]

                    var1_SUP2_healthy_nodes  = self.generate_adaptive_nodes(var1_SUP2_healthy, max_nodes=self.K)
                    var1_PRO1_healthy_nodes  = self.generate_adaptive_nodes(var1_PRO1_healthy, max_nodes=self.K)

                    nodes_dict[f"{var1}-SUP2-healthy"] = var1_SUP2_healthy_nodes
                    nodes_dict[f"{var1}-PRO1-healthy"] = var1_PRO1_healthy_nodes

                    D_var1_SUP2_healthy = P1DesignMatrix(var1_SUP2_healthy_nodes).build(var1_values) * mask_SUP2_healthy.astype(float)[:, None]
                    D_var1_PRO1_healthy = P1DesignMatrix(var1_PRO1_healthy_nodes).build(var1_values) * mask_PRO1_healthy.astype(float)[:, None]

                    D_lst.append(D_var1_SUP2_healthy)
                    D_lst.append(D_var1_PRO1_healthy)

                    if num_classes > 1:
                        mask_SUP2_asthma = (np.asarray(var2_values) == 0) & (np.asarray(var3_values) == 1)
                        mask_PRO1_asthma = (np.asarray(var2_values) == 1) & (np.asarray(var3_values) == 1)

                        var1_SUP2_asthma = np.asarray(var1_values)[mask_SUP2_asthma]
                        var1_PRO1_asthma = np.asarray(var1_values)[mask_PRO1_asthma]

                        var1_SUP2_asthma_nodes  = self.generate_adaptive_nodes(var1_SUP2_asthma, max_nodes=self.K)
                        var1_PRO1_asthma_nodes  = self.generate_adaptive_nodes(var1_PRO1_asthma, max_nodes=self.K)

                        nodes_dict[f"{var1}-SUP2-asthma"] = var1_SUP2_asthma_nodes
                        nodes_dict[f"{var1}-PRO1-asthma"] = var1_PRO1_asthma_nodes

                        D_var1_SUP2_asthma = P1DesignMatrix(var1_SUP2_asthma_nodes).build(var1_values) * mask_SUP2_asthma.astype(float)[:, None]
                        D_var1_PRO1_asthma = P1DesignMatrix(var1_PRO1_asthma_nodes).build(var1_values) * mask_PRO1_asthma.astype(float)[:, None]

                        D_lst.append(D_var1_SUP2_asthma)
                        D_lst.append(D_var1_PRO1_asthma)

                    if num_classes > 2:
                        mask_SUP2_copd = (np.asarray(var2_values) == 0) & (np.asarray(var3_values) == 2)
                        mask_PRO1_copd = (np.asarray(var2_values) == 1) & (np.asarray(var3_values) == 2)

                        var1_SUP2_copd = np.asarray(var1_values)[mask_SUP2_copd]
                        var1_PRO1_copd = np.asarray(var1_values)[mask_PRO1_copd]

                        var1_SUP2_copd_nodes  = self.generate_adaptive_nodes(var1_SUP2_copd, max_nodes=self.K)
                        var1_PRO1_copd_nodes  = self.generate_adaptive_nodes(var1_PRO1_copd, max_nodes=self.K)

                        nodes_dict[f"{var1}-SUP2-copd"] = var1_SUP2_copd_nodes
                        nodes_dict[f"{var1}-PRO1-copd"] = var1_PRO1_copd_nodes

                        D_var1_SUP2_copd = P1DesignMatrix(var1_SUP2_copd_nodes).build(var1_values) * mask_SUP2_copd.astype(float)[:, None]
                        D_var1_PRO1_copd = P1DesignMatrix(var1_PRO1_copd_nodes).build(var1_values) * mask_PRO1_copd.astype(float)[:, None]

                        D_lst.append(D_var1_SUP2_copd)
                        D_lst.append(D_var1_PRO1_copd)

                else:
                    raise ValueError(f"Combination of categorical variables still not implemented: {var2}-{var3}")

            elif var.count("-") == 1:
                var1, var2 = var.split("-") # var2 is the categorical variable
                
                if var1 in ["Position", "Groupe", "Sex"]: # var1 and var2 are categorical
                    var1_values  = self.variables_dict[var1]
                    num1_classes = len(set(var1_values))

                    var2_values  = self.variables_dict[var2]
                    num2_classes = len(set(var2_values))
                    
                    interaction_ids = (np.asarray(var1_values) * num2_classes) + np.asarray(var2_values)

                    D_var = np.eye(num1_classes * num2_classes)[interaction_ids]
                    D_lst.append(D_var)

                    nodes_dict[var] = [f"{i}-{j}" for i in range(num1_classes) for j in range(num2_classes)] # list(set(var_values))
                
                else:
                    # Design matrix (A x K) for numerical variable
                    var1_values = self.variables_dict[var1]
                    var2_values = self.variables_dict[var2]
                    if var2 == "Position":
                        mask_SUP2 = (np.asarray(var2_values) == 0)
                        mask_PRO1 = (np.asarray(var2_values) == 1)

                        var1_SUP2 = np.asarray(var1_values)[mask_SUP2]
                        var1_PRO1 = np.asarray(var1_values)[mask_PRO1]

                        var1_SUP2_nodes  = self.generate_adaptive_nodes(var1_SUP2, max_nodes=self.K)
                        var1_PRO1_nodes  = self.generate_adaptive_nodes(var1_PRO1, max_nodes=self.K)

                        nodes_dict[f"{var1}-SUP2"] = var1_SUP2_nodes
                        nodes_dict[f"{var1}-PRO1"] = var1_PRO1_nodes

                        D_var1_SUP2 = P1DesignMatrix(var1_SUP2_nodes).build(var1_values) * mask_SUP2.astype(float)[:, None]
                        D_var1_PRO1 = P1DesignMatrix(var1_PRO1_nodes).build(var1_values) * mask_PRO1.astype(float)[:, None]

                        D_lst.append(D_var1_SUP2)
                        D_lst.append(D_var1_PRO1)

                    elif var2 == "Groupe":
                        num_classes = len(set(var2_values))

                        mask_healthy                  = (np.asarray(var2_values) == 0)
                        var1_healthy                  = np.asarray(var1_values)[mask_healthy]
                        var1_healthy_nodes            = self.generate_adaptive_nodes(var1_healthy, max_nodes=self.K)
                        nodes_dict[f"{var1}-healthy"] = var1_healthy_nodes
                        D_var1_healthy                = P1DesignMatrix(var1_healthy_nodes).build(var1_values) * mask_healthy.astype(float)[:, None]
                        D_lst.append(D_var1_healthy)

                        if num_classes > 1:
                            mask_asthma                  = (np.asarray(var2_values) == 1)
                            var1_asthma                  = np.asarray(var1_values)[mask_asthma]
                            var1_asthma_nodes            = self.generate_adaptive_nodes(var1_asthma, max_nodes=self.K)
                            nodes_dict[f"{var1}-asthma"] = var1_asthma_nodes
                            D_var1_asthma                = P1DesignMatrix(var1_asthma_nodes).build(var1_values) * mask_asthma.astype(float)[:, None]
                            D_lst.append(D_var1_asthma)

                        if num_classes > 2:
                            mask_copd                  = (np.asarray(var2_values) == 2)
                            var1_copd                  = np.asarray(var1_values)[mask_copd]
                            var1_copd_nodes            = self.generate_adaptive_nodes(var1_copd, max_nodes=self.K)
                            nodes_dict[f"{var1}-copd"] = var1_copd_nodes
                            D_var1_copd                = P1DesignMatrix(var1_copd_nodes).build(var1_values) * mask_copd.astype(float)[:, None]
                            D_lst.append(D_var1_copd)

                        # NOTE For now only cases: healthy, healthy+asthma, healthy+asthma+copd. We can add more if needed.

                    elif var2 == "Sex":
                        mask_M = (np.asarray(var2_values) == 0)
                        mask_F = (np.asarray(var2_values) == 1)

                        var1_M = np.asarray(var1_values)[mask_M]
                        var1_F = np.asarray(var1_values)[mask_F]

                        var1_M_nodes  = self.generate_adaptive_nodes(var1_M, max_nodes=self.K)
                        var1_F_nodes  = self.generate_adaptive_nodes(var1_F, max_nodes=self.K)

                        nodes_dict[f"{var1}-M"] = var1_M_nodes
                        nodes_dict[f"{var1}-F"] = var1_F_nodes

                        D_var1_M = P1DesignMatrix(var1_M_nodes).build(var1_values) * mask_M.astype(float)[:, None]
                        D_var1_F = P1DesignMatrix(var1_F_nodes).build(var1_values) * mask_F.astype(float)[:, None]

                        D_lst.append(D_var1_M)
                        D_lst.append(D_var1_F)
                    
                    else:
                        raise ValueError(f"Categorical variable still not implemented: {var2}")

            elif var in ["Position", "Groupe", "Sex"]:
                var_values  = self.variables_dict[var]
                num_classes = len(set(var_values))
                D_var       = np.eye(num_classes)[np.asarray(var_values)]
                D_lst.append(D_var)

                nodes_dict[var] = list(set(var_values))
                assert self.variables_names == [var], "This case is intended for when we only want to include one categorical variable."
            
            else:
                var_values = self.variables_dict[var]
                var_nodes  = self.generate_adaptive_nodes(var_values, max_nodes=self.K)
                D_var      = P1DesignMatrix(var_nodes).build(var_values)
                D_lst.append(D_var)
                
                nodes_dict[var] = var_nodes
        
        D = np.concatenate(D_lst, axis=1)
        return D, nodes_dict



    def fit(self, X_observed, D, U_s_init, U_t_init, R, epochs=25):
        # ALS algorithm
        X_observed = np.asarray(X_observed, dtype=float)
        U_s        = np.asarray(U_s_init, dtype=float)
        U_t        = np.asarray(U_t_init, dtype=float)

        X_observed_power = np.mean(X_observed**2)
        
        # U_s = U_s_init.clone()
        # U_t = U_t_init.clone()

        B = np.random.default_rng().standard_normal((D.shape[1], R))

        delta_loss = float('inf')
        prev_loss  = float('inf')
        tol        = 1e-6
        epoch      = 0

        print("ALS Algorithm")
        while epoch < epochs and delta_loss > tol:
            
            # ==========================================
            # PHASE 1: EXACT UPDATE FOR B (Acquisition Mode)
            # ==========================================
            # 1. Compute Gram matrices
            V_s = U_s.T @ U_s
            V_t = U_t.T @ U_t
            V = V_s * V_t  # Hadamard product
            
            # 2. MTTKRP: Project X onto Space and Time
            M_a = np.einsum('sta,sr,tr->ar', X_observed, U_s, U_t, optimize=True)
            
            # 3. Unconstrained target: U_a_target = M_a @ V^{-1}
            U_a_target = M_a @ np.linalg.inv(V)
            
            # # 4. Constrained update for B
            # B = np.linalg.lstsq(D, U_a_target).solution

            # 4. A bit of Tikhonov regularization for numerical stability
            I = np.eye(D.shape[1])
            tau = 1e-6

            # Compute Ridge Regression explicitly
            D_T_D = D.T @ D
            D_T_U = D.T @ U_a_target

            # Solve the regularized system: (D^T D + tau*I) * B = D^T U_a_target
            B = np.linalg.solve(D_T_D + tau * I, D_T_U)
            
            # 5. Rebuild the current U_a for the next phases
            norms = np.linalg.norm(B, axis=0, keepdims=True)
            B     = B / norms
            U_a   = D @ B
            U_t   = U_t * norms
            
            # ==========================================
            # PHASE 2: EXACT UPDATE FOR U_s (Spatial Mode)
            # ==========================================
            V_t = U_t.T @ U_t
            V_a = U_a.T @ U_a
            V = V_t * V_a
            
            M_s = np.einsum('sta,tr,ar->sr', X_observed, U_t, U_a, optimize=True)
            U_s = M_s @ np.linalg.inv(V)
            
            # Normalize U_s to avoid scale explosion, push scale to U_t
            norms = np.linalg.norm(U_s, axis=0, keepdims=True)
            U_s = U_s / norms
            U_t = U_t * norms
            # B = B * norms  
            # U_a = D @ B  # Rebuild U_a with new scale
            
            # ==========================================
            # PHASE 3: EXACT UPDATE FOR U_t (Temporal Mode)
            # ==========================================
            V_s = U_s.T @ U_s
            V_a = U_a.T @ U_a
            V = V_s * V_a
            
            M_t = np.einsum('sta,sr,ar->tr', X_observed, U_s, U_a, optimize=True)
            U_t = M_t @ np.linalg.inv(V)
            
            # Normalize U_t, push scale to B
            # norms = np.linalg.norm(U_t, axis=0, keepdims=True)
            # U_t = U_t / norms
            # B = B * norms  
            # U_a = D @ B  # Rebuild U_a with new scale

            # ==========================================
            # EVALUATE LOSS
            # ==========================================
            # Reconstruct tensor to check convergence
            X_hat = np.einsum('sr,tr,ar->sta', U_s, U_t, U_a, optimize=True)
            loss  = np.mean((X_hat - X_observed)**2) / X_observed_power
            
            print(f"Epoch {epoch+1:02d} | Normalized MSE: {loss.item():.6f}")
            epoch += 1
            delta_loss = abs(prev_loss - loss.item())
            prev_loss  = loss.item()
        
        # Global R2 Score
        var_observed = np.var(X_observed)
        r2_score     = 1. - (np.mean((X_hat - X_observed)**2) / var_observed)
        print (f"Final R2 Score: {r2_score.item():.4f}")
        
        cfe_matrices = {"U_s" : U_s,
                        "U_t" : U_t,
                        "B"   : B}

        return cfe_matrices, loss.item()