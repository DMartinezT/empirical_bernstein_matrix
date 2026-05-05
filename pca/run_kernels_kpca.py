import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.sparse.linalg import eigs

# --- 1. KERNEL DEFINITIONS ---

def pairwise_sq_dists(X, Y=None):
    """Helper to compute pairwise squared Euclidean distances."""
    if Y is None: Y = X
    sq_dists = np.sum(X**2, axis=1).reshape(-1, 1) + np.sum(Y**2, axis=1) - 2 * np.dot(X, Y.T)
    return np.clip(sq_dists, 0, None) # Clip to 0 to avoid floating point negatives

def rbf_kernel(X, Y=None, gamma=0.1):
    """Gaussian RBF Kernel"""
    return np.exp(-gamma * pairwise_sq_dists(X, Y))

def laplace_kernel(X, Y=None, gamma=0.1):
    """Laplacian Kernel (using L2 distance)"""
    dists = np.sqrt(pairwise_sq_dists(X, Y))
    return np.exp(-gamma * dists)

def imq_kernel(X, Y=None, c=1.0):
    """Inverse Multi-Quadric (IMQ) Kernel"""
    return 1.0 / np.sqrt(pairwise_sq_dists(X, Y) + c**2)


# --- 2. SIMULATION FUNCTION ---

def simulate_kpca_kernels(n_values, d=10, alpha=0.05, trials=20):
    np.random.seed(42)
    
    # Define our dictionary of kernels to test
    kernels = {
        "RBF Kernel": rbf_kernel,
        "Laplace Kernel": laplace_kernel,
        "IMQ Kernel": imq_kernel
    }
    
    # Fixed data generating distribution: Standard Normal
    data_fn = lambda n, dim: np.random.normal(0, 1, size=(n, dim))
    
    results = []

    for kernel_name, kernel_fn in kernels.items():
        print(f"\n--- Processing Kernel: {kernel_name} ---")
        
        # 1. Oracle Computation (Offline via Monte Carlo)
        N_pop = 2000
        X_pop = data_fn(N_pop, d)
        K_pop = kernel_fn(X_pop)
        
        # Eigenvalues of Sigma are 1/N * eigenvalues of K
        eigenvalues = np.linalg.eigvalsh(K_pop) / N_pop
        eigenvalues = np.clip(eigenvalues, 0, None) 
        
        # True variance operator V = Sigma - Sigma^2
        gamma_j = eigenvalues * (1 - eigenvalues)
        norm_V = np.max(gamma_j)
        tr_V = np.sum(gamma_j)
        true_intrinsic_dim = tr_V / norm_V
        
        c_bound = 1.0
        B_bound = 1.0

        for n in n_values:
            assert n % 4 == 0
            r_n_list = []
            
            # Intrinsic Oracle Baseline
            log_term_oracle = np.log((2 / alpha) * true_intrinsic_dim)
            d_oracle = np.sqrt((2 * norm_V * log_term_oracle) / n) + ((c_bound / (3 * n)) * log_term_oracle)

            for _ in tqdm(range(trials), desc=f"n={n}"):
                X = data_fn(n, d)
                K = kernel_fn(X) 
                
                # --- EXACT DRAFT OEB VIA KERNEL TRICK ---
                delta = alpha
                d1, d2, d3 = ((n - 2) * delta) / n, delta / n, delta / n
                
                # Level 1 Traces
                Z_tr = np.zeros(n // 2)
                for i in range(n // 2):
                    Z_tr[i] = 1 - (K[2*i, 2*i+1] ** 2)
                
                tr_Sigma_n = (2 / n) * np.sum(Z_tr)
                varsigma_Z = np.std(Z_tr, ddof=1)
                
                # Level 1 Operator Norm
                M = np.zeros((n, n))
                for i in range(n // 2):
                    k_i = K[2*i, 2*i+1]
                    M[2*i, 2*i] = 1
                    M[2*i+1, 2*i+1] = 1
                    M[2*i, 2*i+1] = -k_i
                    M[2*i+1, 2*i] = -k_i
                M = M / n 
                
                # Fast Arnoldi iteration for top eigenvalue
                top_eval, _ = eigs(K @ M, k=1, which='LR', tol=1e-3)
                norm_Sigma_n = np.real(top_eval[0])
                
                tau_u = tr_Sigma_n + varsigma_Z * np.sqrt((2 * np.log(2 / d2)) / (n / 2)) + \
                        (7 * (2 * B_bound) * np.log(2 / d2)) / (3 * ((n / 2) - 1))
                
                # Level 2 Traces
                Z_prime_tr = np.zeros(n // 4)
                for i in range(n // 4):
                    idx_A1, idx_A2 = 4*i, 4*i+1
                    idx_B1, idx_B2 = 4*i+2, 4*i+3
                    
                    k_A = K[idx_A1, idx_A2]
                    k_B = K[idx_B1, idx_B2]
                    
                    M_A = 0.5 * np.array([[1, -k_A], [-k_A, 1]])
                    M_B = 0.5 * np.array([[1, -k_B], [-k_B, 1]])
                    
                    K_AB = K[idx_A1:idx_A2+1, idx_B1:idx_B2+1]
                    K_BA = K_AB.T
                    
                    tr_A2 = 0.5 * (1 - k_A**2)**2
                    tr_B2 = 0.5 * (1 - k_B**2)**2
                    tr_AB = np.trace(M_A @ K_AB @ M_B @ K_BA)
                    
                    Z_prime_tr[i] = 0.5 * (tr_A2 + tr_B2 - 2 * tr_AB)
                
                tr_Sigma_prime_n = (4 / n) * np.sum(Z_prime_tr)
                varsigma_Z_prime = np.std(Z_prime_tr, ddof=1)
                
                B_prime = 4 * B_bound * (c_bound ** 2)
                tau_prime_u = tr_Sigma_prime_n + varsigma_Z_prime * np.sqrt((2 * np.log(4 / d3)) / (n / 4)) + \
                              (7 * B_prime * np.log(4 / d3)) / (3 * ((n / 4) - 1))
                
                sigma_u_sq = norm_Sigma_n + np.sqrt((2 * tau_prime_u * np.log(4 / d3)) / (n / 2)) + \
                             (2 * (c_bound ** 2) * np.log(4 / d3)) / (3 * (n / 2))
                
                intrinsic_ratio = max(tau_u / sigma_u_sq, 1.0)
                log_term_Rn = np.log((2 / d1) * intrinsic_ratio)
                r_n = (np.sqrt(sigma_u_sq) * np.sqrt((2 / n) * log_term_Rn)) + ((c_bound / (3 * n)) * log_term_Rn)
                r_n_list.append(r_n)

            draft_ratios = np.array(r_n_list) / d_oracle
            results.append({
                "Kernel": kernel_name,
                "n": n,
                "Draft Mean": np.mean(draft_ratios),
                "Draft Lower": np.percentile(draft_ratios, 2.5),
                "Draft Upper": np.percentile(draft_ratios, 97.5)
            })

    return pd.DataFrame(results)

if __name__ == "__main__":
    n_values = [100, 300, 1000, 3000, 10000, 30000]
    df_results = simulate_kpca_kernels(n_values, trials=20)
    
    # Save the data to a file
    data_filename = "outputs/kpca_kernels_results.csv"
    df_results.to_csv(data_filename, index=False)
    print(f"\n[+] Simulation complete. Data saved to '{data_filename}'")