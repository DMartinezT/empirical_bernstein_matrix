import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

 

def psi_E(gamma):
    """Cumulant generating function for standard exponential."""
    return -np.log(1 - gamma) - gamma

def simulate_inequalities(n_values, d=100, alpha=0.05, trials=50, run_meb2=True):
    """
    Simulates and compares the confidence radii of:
      1. Oracle Matrix Bennett-Bernstein (D_tb)
      2. Hongjian's MEB 1 (D_meb1)
      3. Hongjian's MEB 2 (D_meb2) - Optional
      4. Draft's Intrinsic Dimension MEB (R_n)
    """
    # Define the 4 scenarios for a_i
    indices = np.arange(1, d + 1)
    scenarios = {
        "1. Completely Anisotropic": np.array([1.0] + [0.0] * (d - 1)),
        "2. Completely Isotropic": np.ones(d),
        "3. Polynomial Decay": indices ** (-2.0),
        "4. Exponential Decay": np.exp(-indices)
    }

    results = []

    for scenario_name, a in scenarios.items():
        # True variance parameters based on Unif[0, a_i]
        true_var = (a ** 2) / 12.0
        norm_V = np.max(true_var)
        

        # Realistic worst-case parameters for the Draft's inequality
        c_bound = 1.0
        B_bound = d * (c_bound ** 2)  # effectively just d

        for n in n_values:
            assert n % 4 == 0, "n must be a multiple of 4 for this splitting scheme."
            
            d_tb_list, d_meb1_list, d_meb2_list, r_n_list = [], [], [], []
            
            for _ in tqdm(range(trials)):
                # -----------------------------------------------------------
                # DATA GENERATION
                # Shape: (n, d)
                # -----------------------------------------------------------
                X = np.random.uniform(0, a, size=(n, d))
                
                # -----------------------------------------------------------
                # 1. ORACLE BENNETT-BERNSTEIN (Hongjian D_tb)
                # -----------------------------------------------------------
                term1_tb = (c_bound * np.log(d / alpha)) / (3 * n)
                term2_tb = np.sqrt((2 * np.log(d / alpha) * norm_V) / n)
                d_tb = term1_tb + term2_tb
                d_tb_list.append(d_tb)
                
                
                # -----------------------------------------------------------
                # 2. HONGJIAN'S MEB 1 (D_meb1)
                # -----------------------------------------------------------
                diffs = X[0:n:2] - X[1:n:2] 
                V_n_star_diag = np.sum(diffs ** 2, axis=0) / n
                norm_V_n_star = np.max(V_n_star_diag)
                
                log_term_meb1 = np.log((n * d) / ((n - 1) * alpha))
                log_term2_meb1 = np.log((2 * n * d) / alpha)
                
                term1_meb1 = log_term_meb1 / (3 * n)
                term2_meb1 = np.sqrt((2 * norm_V_n_star * log_term_meb1) / n)
                term3_meb1 = (np.sqrt(5/3) + 1) * np.sqrt(log_term_meb1 * log_term2_meb1) / n
                d_meb1 = term1_meb1 + term2_meb1 + term3_meb1
                d_meb1_list.append(d_meb1)
                
                # -----------------------------------------------------------
                # 3. HONGJIAN'S MEB 2 (D_meb2) - Supermartingale Method
                # -----------------------------------------------------------
                if run_meb2:
                    sum_X = np.zeros(d)
                    sum_X2 = np.zeros(d)
                    
                    v_bar_prev = max(0.25, 5 * np.log(d / alpha) / n)
                    X_bar_prev = np.zeros(d)
                    
                    sum_gamma = 0.0
                    sum_psi_E_diff = np.zeros(d)
                    
                    for i in range(1, n + 1):
                        x_i = X[i-1]
                        
                        # Calculate gamma_i
                        gamma_i = np.sqrt((2 * np.log(d / alpha)) / (n * v_bar_prev))
                        sum_gamma += gamma_i
                        
                        # Add psi_E term
                        diff_sq = (x_i - X_bar_prev) ** 2
                        sum_psi_E_diff += psi_E(gamma_i) * diff_sq
                        
                        # Fast O(1) update of running variance for next iteration
                        sum_X += x_i
                        sum_X2 += x_i ** 2
                        X_bar_k = sum_X / i
                        V_bar_k = (sum_X2 / i) - (X_bar_k ** 2)
                        
                        v_bar_prev = max(np.max(V_bar_k), 5 * np.log(d / alpha) / n)
                        X_bar_prev = X_bar_k
                        
                    d_meb2 = (np.log(d / alpha) + np.max(sum_psi_E_diff)) / sum_gamma
                    d_meb2_list.append(d_meb2)

                # -----------------------------------------------------------
                # 4. DRAFT'S INTRINSIC-DIMENSION MEB (R_n)
                # -----------------------------------------------------------
                delta = alpha
                d1 = ((n - 2) * delta) / n
                d2 = delta / n
                d3 = delta / n
                
                X_prime = 0.5 * (diffs ** 2) 
                Sigma_n_diag = (2 / n) * np.sum(X_prime, axis=0)
                norm_Sigma_n = np.max(Sigma_n_diag)
                tr_Sigma_n = np.sum(Sigma_n_diag)
                
                Z = np.sum(X_prime, axis=1)
                varsigma_Z = np.std(Z, ddof=1) if len(Z) > 1 else 0
                
                tau_u = tr_Sigma_n + varsigma_Z * np.sqrt((2 * np.log(2 / d2)) / (n / 2)) + \
                        (7 * (2 * B_bound) * np.log(2 / d2)) / (3 * ((n / 2) - 1))
                
                X_prime_diffs = X_prime[0:n//2:2] - X_prime[1:n//2:2]
                X_double_prime = 0.5 * (X_prime_diffs ** 2)
                Sigma_prime_n_diag = (4 / n) * np.sum(X_double_prime, axis=0)
                tr_Sigma_prime_n = np.sum(Sigma_prime_n_diag)
                
                Z_prime = np.sum(X_double_prime, axis=1)
                varsigma_Z_prime = np.std(Z_prime, ddof=1) if len(Z_prime) > 1 else 0
                
                B_prime = 4 * B_bound * (c_bound ** 2)
                tau_prime_u = tr_Sigma_prime_n + varsigma_Z_prime * np.sqrt((2 * np.log(4 / d3)) / (n / 4)) + \
                              (7 * B_prime * np.log(4 / d3)) / (3 * ((n / 4) - 1))
                
                sigma_u_sq = norm_Sigma_n + np.sqrt((2 * tau_prime_u * np.log(4 / d3)) / (n / 2)) + \
                             (2 * (c_bound ** 2) * np.log(4 / d3)) / (3 * (n / 2))
                
                intrinsic_ratio = max(tau_u / sigma_u_sq, 1.0)
                log_term_Rn = np.log((2 / d1) * intrinsic_ratio)
                
                term1_Rn = np.sqrt(sigma_u_sq) * np.sqrt((2 / n) * log_term_Rn)
                term2_Rn = (c_bound / (3 * n)) * log_term_Rn
                r_n = term1_Rn + term2_Rn
                r_n_list.append(r_n)

            # Averages
            avg_d_tb = np.mean(d_tb_list)
            avg_d_meb1 = np.mean(d_meb1_list)
            avg_r_n = np.mean(r_n_list)
            
            row_data = {
                "Scenario": scenario_name,
                "n": n,
                "Oracle (D_tb)": avg_d_tb,
                "MEB 1 / Oracle": avg_d_meb1 / avg_d_tb,
                "Draft / Oracle": avg_r_n / avg_d_tb
            }
            
            if run_meb2:
                avg_d_meb2 = np.mean(d_meb2_list)
                row_data["MEB 2 / Oracle"] = avg_d_meb2 / avg_d_tb
                
            results.append(row_data)

    df = pd.DataFrame(results)
    
    # Reorder columns for readability
    if run_meb2:
        df = df[['Scenario', 'n', 'Oracle (D_tb)', 'MEB 1 / Oracle', 'MEB 2 / Oracle', 'Draft / Oracle']]
    
    return df

def save_latex_tables(df_results, filename="neurips_simulation_table.tex"):
    """
    Converts the results DataFrame into a single, unified publication-ready 
    LaTeX table and saves it to a text file.
    """
    with open(filename, "w") as f:
        # Write a quick header for the file
        f.write("% =========================================================\n")
        f.write("% Unified LaTeX Table for Matrix Empirical Bernstein Simulations\n")
        f.write("% Requires: \\usepackage{booktabs}, \\usepackage{multirow}, \\usepackage{graphicx}\n")
        f.write("% =========================================================\n\n")

        # Clean up the scenario names (remove the "1. ", "2. " prefixes for the paper)
        df_cleaned = df_results.copy()
        df_cleaned["Scenario"] = df_cleaned["Scenario"].apply(lambda x: x.split(". ", 1)[-1])

        # Create a MultiIndex for a clean, grouped layout
        df_multi = df_cleaned.set_index(["Scenario", "n"])
        
        # Generate the raw tabular environment
        latex_tabular = df_multi.to_latex(
            float_format="%.3f",
            multirow=True,     # Groups the scenario names cleanly
            column_format="ll" + "c" * len(df_multi.columns) # Left align indices, center data
        )
        
        # Wrap the tabular data in a standard floating table environment.
        # Adding a resizebox ensures it fits neatly within the page margins.
        latex_wrapper = (
            "\\begin{table}[htbp]\n"
            "\\centering\n"
            "\\caption{Ratio of confidence radii across MEB inequalities under various spectral decays. Lower is better.}\n"
            "\\label{tab:meb_simulations_all}\n"
            "\\resizebox{\\textwidth}{!}{\n"
            f"{latex_tabular}"
            "}\n"
            "\\end{table}\n"
        )
        
        f.write(latex_wrapper)
            
    print(f"\n[+] Success! A single unified LaTeX table has been saved to '{filename}'")

def save_plots(df_results, filename="neurips_simulation_plot.png"):
    """
    Generates a 2x2 grid of plots showing the ratio of each empirical bound 
    to the Oracle bound across different sample sizes.
    """
    scenarios = df_results["Scenario"].unique()
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.flatten() 
    
    colors = {"MEB 1": "#1f77b4", "MEB 2": "#ff7f0e", "Draft": "#2ca02c"}
    markers = {"MEB 1": "o", "MEB 2": "s", "Draft": "^"}

    for i, scenario in enumerate(scenarios):
        ax = axes[i]
        group = df_results[df_results["Scenario"] == scenario]
        
        n_vals = group["n"]
        
        # Plot MEB 1
        ax.plot(n_vals, group["MEB 1 / Oracle"], 
                color=colors["MEB 1"], marker=markers["MEB 1"], 
                linewidth=2, label="MEB 1 (Hongjian)")
        
        # Plot MEB 2 if it exists
        if "MEB 2 / Oracle" in group.columns:
            ax.plot(n_vals, group["MEB 2 / Oracle"], 
                    color=colors["MEB 2"], marker=markers["MEB 2"], 
                    linewidth=2, label="MEB 2 (Hongjian)")
            
        # Plot Your Draft's Bound
        ax.plot(n_vals, group["Draft / Oracle"], 
                color=colors["Draft"], marker=markers["Draft"], 
                linewidth=2, label="Intrinsic MEB (Ours)")
        
        # Add the Oracle baseline
        ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1.5, label="Oracle (Baseline)")
        
        ax.set_xscale("log")
        clean_title = scenario.split(". ", 1)[-1] 
        ax.set_title(clean_title, fontweight="bold")
        ax.set_xlabel("Sample Size (n)")
        ax.set_ylabel("Ratio to Oracle")
        ax.grid(True, which="both", linestyle=":", alpha=0.6)
        
        if i == 0:
            ax.legend(loc="best", framealpha=0.9)

    plt.tight_layout()
    
    # Changed format to "png" here! (dpi=300 ensures it stays crisp for the paper)
    plt.savefig(filename, dpi=300, bbox_inches="tight") #format="png"
    plt.close()
    
    print(f"\n[+] Success! Plot saved as a high-res PNG to '{filename}'")

    
# Run the simulation 
n_values = [1000, 10000, 100000, 1000000, 10000000]
n_values = [10000, 100000, 1000000, 10000000]
# n_values = [100, 1000]
d = 3
trials = 5
alpha = 0.05
df_results = simulate_inequalities(n_values, d = d, alpha = alpha, trials = trials, run_meb2=False)

# Print results
for scenario, group in df_results.groupby("Scenario"):
    print(f"\n--- {scenario} ---")
    display_cols = [c for c in df_results.columns if c != "Scenario"]
    print(group[display_cols].to_string(index=False))

# Save the LaTeX tables to file
save_latex_tables(df_results, filename="neurips_simulation_tables.tex")
save_plots(df_results, filename="neurips_simulation_plot.png")