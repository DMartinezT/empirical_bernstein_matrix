import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

 

def psi_E(gamma):
    """Cumulant generating function for standard exponential."""
    return -np.log(1 - gamma) - gamma

def psi_E(gamma):
    return -np.log(1 - gamma) - gamma

def simulate_inequalities(n_values, d=100, alpha=0.05, trials=50, run_meb2=True):
    indices = np.arange(1, d + 1)
    
    # Removed Exponential Decay; kept 3 scenarios
    scenarios = {
        "1. Completely Isotropic": np.ones(d),
        "2. Completely Anisotropic": np.array([1.0] + [0.0] * (d - 1)),
        "3. Polynomial Decay": indices ** (-2.0)
    }

    results = []

    for scenario_name, a in scenarios.items():
        true_var = (a ** 2) / 12.0
        norm_V = np.max(true_var)
        tr_V = np.sum(true_var)
        true_intrinsic_dim = tr_V / norm_V
        
        c_bound = 1.0
        B_bound = d * (c_bound ** 2)

        for n in n_values:
            assert n % 4 == 0
            
            d_meb1_list, d_meb2_list, r_n_list = [], [], []
            
            # The Oracles are deterministic for a given n, so we only compute them once per n
            log_term_oracle = np.log((2 / alpha) * true_intrinsic_dim)
            d_oracle = np.sqrt((2 * norm_V * log_term_oracle) / n) + ((c_bound / (3 * n)) * log_term_oracle)

            log_2d_alpha = np.log((2 * d) / alpha)
            d_tb = ((c_bound * log_2d_alpha) / (3 * n)) + np.sqrt((2 * log_2d_alpha * norm_V) / n)
            
            for _ in tqdm(range(trials)):
                X = np.random.uniform(0, a, size=(n, d))
                
                # --- HONGJIAN'S MEB 1 ---
                diffs = X[0:n:2] - X[1:n:2] 
                V_n_star_diag = np.sum(diffs ** 2, axis=0) / n
                norm_V_n_star = np.max(V_n_star_diag)
                
                log_term_meb1 = np.log((n * 2 * d) / ((n - 1) * alpha))
                log_term2_meb1 = np.log((4 * n * d) / alpha) 
                
                d_meb1 = (log_term_meb1 / (3 * n)) + \
                         np.sqrt((2 * norm_V_n_star * log_term_meb1) / n) + \
                         (np.sqrt(5/3) + 1) * np.sqrt(log_term_meb1 * log_term2_meb1) / n
                d_meb1_list.append(d_meb1)
                
                # --- HONGJIAN'S MEB 2 ---
                if run_meb2:
                    sum_X = np.zeros(d)
                    sum_X2 = np.zeros(d)
                    v_bar_prev = max(0.25, 5 * log_2d_alpha / n)
                    X_bar_prev = np.zeros(d)
                    sum_gamma = 0.0
                    sum_psi_E_diff = np.zeros(d)
                    
                    for i in range(1, n + 1):
                        x_i = X[i-1]
                        gamma_i = np.sqrt((2 * log_2d_alpha) / (n * v_bar_prev))
                        sum_gamma += gamma_i
                        diff_sq = (x_i - X_bar_prev) ** 2
                        sum_psi_E_diff += psi_E(gamma_i) * diff_sq
                        
                        sum_X += x_i
                        sum_X2 += x_i ** 2
                        X_bar_k = sum_X / i
                        V_bar_k = (sum_X2 / i) - (X_bar_k ** 2)
                        v_bar_prev = max(np.max(V_bar_k), 5 * log_2d_alpha / n)
                        X_bar_prev = X_bar_k
                        
                    d_meb2 = (log_2d_alpha + np.max(sum_psi_E_diff)) / sum_gamma
                    d_meb2_list.append(d_meb2)

                # --- DRAFT'S INTRINSIC-DIMENSION MEB ---
                delta = alpha
                d1 = ((n - 2) * delta) / n
                d2, d3 = delta / n, delta / n
                
                X_prime = 0.5 * (diffs ** 2) 
                Sigma_n_diag = (2 / n) * np.sum(X_prime, axis=0)
                norm_Sigma_n, tr_Sigma_n = np.max(Sigma_n_diag), np.sum(Sigma_n_diag)
                
                Z = np.sum(X_prime, axis=1)
                varsigma_Z = np.std(Z, ddof=1) if len(Z) > 1 else 0
                
                tau_u = tr_Sigma_n + varsigma_Z * np.sqrt((2 * np.log(2 / d2)) / (n / 2)) + \
                        (7 * (2 * B_bound) * np.log(2 / d2)) / (3 * ((n / 2) - 1))
                
                X_prime_diffs = X_prime[0:n//2:2] - X_prime[1:n//2:2]
                X_double_prime = 0.5 * (X_prime_diffs ** 2)
                tr_Sigma_prime_n = np.sum((4 / n) * np.sum(X_double_prime, axis=0))
                
                Z_prime = np.sum(X_double_prime, axis=1)
                varsigma_Z_prime = np.std(Z_prime, ddof=1) if len(Z_prime) > 1 else 0
                
                B_prime = 4 * B_bound * (c_bound ** 2)
                tau_prime_u = tr_Sigma_prime_n + varsigma_Z_prime * np.sqrt((2 * np.log(4 / d3)) / (n / 4)) + \
                              (7 * B_prime * np.log(4 / d3)) / (3 * ((n / 4) - 1))
                
                sigma_u_sq = norm_Sigma_n + np.sqrt((2 * tau_prime_u * np.log(4 / d3)) / (n / 2)) + \
                             (2 * (c_bound ** 2) * np.log(4 / d3)) / (3 * (n / 2))
                
                intrinsic_ratio = max(tau_u / sigma_u_sq, 1.0)
                log_term_Rn = np.log((2 / d1) * intrinsic_ratio)
                r_n = (np.sqrt(sigma_u_sq) * np.sqrt((2 / n) * log_term_Rn)) + ((c_bound / (3 * n)) * log_term_Rn)
                r_n_list.append(r_n)

            # Store Means and 95% Confidence Intervals (2.5th and 97.5th percentiles)
            meb1_ratios = np.array(d_meb1_list) / d_oracle
            draft_ratios = np.array(r_n_list) / d_oracle
            
            row_data = {
                "Scenario": scenario_name,
                "n": n,
                "Ambient Oracle / Intrinsic": d_tb / d_oracle,
                "MEB 1 Mean": np.mean(meb1_ratios),
                "MEB 1 Lower": np.percentile(meb1_ratios, 2.5),
                "MEB 1 Upper": np.percentile(meb1_ratios, 97.5),
                "Draft Mean": np.mean(draft_ratios),
                "Draft Lower": np.percentile(draft_ratios, 2.5),
                "Draft Upper": np.percentile(draft_ratios, 97.5)
            }
            
            if run_meb2:
                meb2_ratios = np.array(d_meb2_list) / d_oracle
                row_data["MEB 2 Mean"] = np.mean(meb2_ratios)
                row_data["MEB 2 Lower"] = np.percentile(meb2_ratios, 2.5)
                row_data["MEB 2 Upper"] = np.percentile(meb2_ratios, 97.5)
                
            results.append(row_data)

    return pd.DataFrame(results)



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
    scenarios = df_results["Scenario"].unique()
    
    # 1x3 Grid setup. Width=15, Height=4 is perfect for a full-width NeurIPS figure.
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    
    colors = {"MEB 1": "#1f77b4", "MEB 2": "#ff7f0e", "Draft": "#2ca02c", "Ambient Oracle": "#d62728"}
    markers = {"MEB 1": "o", "MEB 2": "s", "Draft": "^"}

    for i, scenario in enumerate(scenarios):
        ax = axes[i]
        group = df_results[df_results["Scenario"] == scenario]
        n_vals = group["n"].values
        
        # --- Plot MEB 1 ---
        ax.plot(n_vals, group["MEB 1 Mean"], color=colors["MEB 1"], marker=markers["MEB 1"], linewidth=2, label="MEB 1")
        ax.fill_between(n_vals, group["MEB 1 Lower"], group["MEB 1 Upper"], color=colors["MEB 1"], alpha=0.15)
        
        # --- Plot MEB 2 ---
        if "MEB 2 Mean" in group.columns:
            ax.plot(n_vals, group["MEB 2 Mean"], color=colors["MEB 2"], marker=markers["MEB 2"], linewidth=2, label="MEB 2")
            ax.fill_between(n_vals, group["MEB 2 Lower"], group["MEB 2 Upper"], color=colors["MEB 2"], alpha=0.15)
            
        # --- Plot Draft Bound ---
        ax.plot(n_vals, group["Draft Mean"], color=colors["Draft"], marker=markers["Draft"], linewidth=2, label="OEB (Ours)")
        ax.fill_between(n_vals, group["Draft Lower"], group["Draft Upper"], color=colors["Draft"], alpha=0.25)
        
        # --- Plot Oracles ---
        ax.plot(n_vals, group["Ambient Oracle / Intrinsic"], color=colors["Ambient Oracle"], linestyle=":", linewidth=2.5, label="Ambient Oracle (Tropp)")
        ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1.5, label="Intrinsic Oracle (Ours)")
        
        # --- Formatting ---
        ax.set_xscale("log")
        clean_title = scenario.split(". ", 1)[-1] 
        ax.set_title(clean_title, fontweight="bold")
        ax.set_xlabel("Sample Size (n)")
        
        # Only add the Y-label to the first plot to keep things clean
        if i == 0:
            ax.set_ylabel("Ratio to Intrinsic Oracle")
            
        # Explicitly enforce X-ticks at the exact sample sizes considered
        ax.set_xticks(n_vals)
        ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter()) # Disables scientific notation (10^4)
        ax.set_xticklabels([f"{int(n):,}" for n in n_vals], rotation=45) # Formats nicely (e.g. 10,000)
        ax.minorticks_off() # Prevents log-scale minor ticks from cluttering the custom ticks
        
        ax.grid(True, which="major", linestyle=":", alpha=0.7)
        
        # Legend placement
        if i == 0:
            ax.legend(loc="best", framealpha=0.9, fontsize="small")

    plt.tight_layout()
    plt.savefig(filename, format="png", dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"\n[+] Success! 1x3 Plot saved as a high-res PNG to '{filename}'")


# Run the simulation 
n_values = [1000, 10000, 100000, 1000000, 10000000]
n_values = [10000, 100000, 1000000, 10000000]
n_values = [10000, 50000, 100000, 500000]
n_values = [10000, 30000, 100000, 300000, 1000000]
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