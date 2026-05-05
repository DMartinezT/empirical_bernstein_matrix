import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def save_plots(df_results, filename="neurips_simulation_plot.png"):
    scenarios = df_results["Scenario"].unique()
    
    # 1x3 Grid setup. 
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    
    colors = {"MEB 1": "#1f77b4", "MEB 2": "#ff7f0e", "Draft": "#2ca02c", "Ambient Oracle": "#d62728"}
    markers = {"MEB 1": "o", "MEB 2": "s", "Draft": "^"}

    # --- Helper to format numbers as LaTeX scientific notation ---
    def format_sci(n):
        exponent = int(np.log10(n))
        coeff = int(n / (10**exponent))
        if coeff == 1:
            return f"$10^{{{exponent}}}$"
        return f"${coeff} \\times 10^{{{exponent}}}$"

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
        ax.plot(n_vals, group["Ambient Oracle / Intrinsic"], color=colors["Ambient Oracle"], linestyle=":", linewidth=2.5, label="Ambient Oracle")
        ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1.5, label="Intrinsic Oracle")
        
        # --- Formatting ---
        ax.set_xscale("log")
        clean_title = scenario.split(". ", 1)[-1] 
        ax.set_title(clean_title, fontweight="bold")
        ax.set_xlabel("Sample Size (n)")
        
        if i == 0:
            ax.set_ylabel("Ratio to Intrinsic Oracle")
            
        # --- Updated X-Ticks Logic ---
        ax.set_xticks(n_vals)
        # We drop the ScalarFormatter and map our custom LaTeX function
        ax.set_xticklabels([format_sci(n) for n in n_vals], rotation=0) 
        ax.minorticks_off() 
        
        ax.grid(True, which="major", linestyle=":", alpha=0.7)
        
        if i == 0:
            ax.legend(loc="best", framealpha=0.9, fontsize="small")

    plt.tight_layout()
    plt.savefig(filename, format="png", dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"\n[+] Success! 1x3 Plot saved as a high-res PNG to '{filename}'")


if __name__ == "__main__":
    data_filename = "simulation_results.csv"
    plot_filename = "neurips_simulation_plot.png"
    
    try:
        # 1. LOAD THE DATA FROM THE FILE
        print(f"Loading data from '{data_filename}'...")
        df_loaded = pd.read_csv(data_filename)
        
        # 2. GENERATE AND SAVE THE PLOTS
        print("Generating plots...")
        save_plots(df_loaded, filename=plot_filename)
        
        # Optional: Generate the LaTeX tables too without re-running!
        # save_latex_tables(df_loaded, filename="neurips_simulation_tables.tex")
        
    except FileNotFoundError:
        print(f"Error: Could not find '{data_filename}'. Please run the simulation first.")