import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tqdm import tqdm
from scipy.sparse.linalg import eigs

def rbf_kernel(X, Y=None, gamma=0.1):
    """Gaussian RBF Kernel"""
    if Y is None: Y = X
    sq_dists = np.sum(X**2, axis=1).reshape(-1, 1) + np.sum(Y**2, axis=1) - 2 * np.dot(X, Y.T)
    return np.exp(-gamma * sq_dists)

def simulate_kpca_distributions(n_values, d=10, alpha=0.05, trials=20):

    np.random.seed(42)

    
    # Define the three input data generating distributions
    distributions = {
        "Gaussian $\\mathcal{N}(0, 1)$": lambda n, dim: np.random.normal(0, 1, size=(n, dim)),
        "Uniform $[-1, 1]$": lambda n, dim: np.random.uniform(-1, 1, size=(n, dim)),
        "Exponential $\\text{Exp}(1)$": lambda n, dim: np.random.exponential(1, size=(n, dim))
    }
    
    results = []

    for dist_name, data_fn in distributions.items():
        print(f"\n--- Processing Distribution: {dist_name} ---")
        
        # 1. Oracle Computation (Offline via Monte Carlo)
        N_pop = 2000
        X_pop = data_fn(N_pop, d)
        K_pop = rbf_kernel(X_pop)
        
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
                K = rbf_kernel(X) 
                
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
                "Distribution": dist_name,
                "n": n,
                "Draft Mean": np.mean(draft_ratios),
                "Draft Lower": np.percentile(draft_ratios, 2.5),
                "Draft Upper": np.percentile(draft_ratios, 97.5)
            })

    return pd.DataFrame(results)

def plot_infinite_kpca(df_results, filename="kpca_distributions_plot.png"):
    distributions = df_results["Distribution"].unique()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    
    def format_sci(n):
        exponent = int(np.log10(n))
        coeff = int(n / (10**exponent))
        return f"$10^{{{exponent}}}$" if coeff == 1 else f"${coeff} \\times 10^{{{exponent}}}$"

    for i, dist in enumerate(distributions):
        ax = axes[i]
        group = df_results[df_results["Distribution"] == dist]
        n_vals = group["n"].values
        
        ax.plot(n_vals, group["Draft Mean"], '^--', color="#2ca02c", linewidth=2.5, markersize=8, label="OEB (Ours)")
        ax.fill_between(n_vals, group["Draft Lower"], group["Draft Upper"], color="#2ca02c", alpha=0.25)
        ax.axhline(y=1.0, color="black", linestyle="--", linewidth=2, label="Intrinsic Oracle")
        
        ax.set_xscale("log")
        ax.set_title(f"Input: {dist}", fontweight="bold")
        ax.set_xlabel("Sample Size (n)")
        
        if i == 0:
            ax.set_ylabel("Ratio to Intrinsic Oracle")
            ax.legend(loc="upper right", framealpha=0.9)
            
        ax.set_xticks(n_vals)
        ax.set_xticklabels([format_sci(n) for n in n_vals])
        ax.minorticks_off()
        ax.grid(True, which="major", linestyle=":", alpha=0.7)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Saved exact KPCA distributions plot to {filename}")

# Run
n_values = [100, 300, 1000, 3000, 10000, 30000]
df = simulate_kpca_distributions(n_values, trials=20)
plot_infinite_kpca(df)