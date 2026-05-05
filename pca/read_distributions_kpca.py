import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


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





if __name__ == "__main__":
    data_filename = "outputs/kpca_distributions.csv"
    plot_filename = "outputs/kpca_distributions_plot.png"
    
    try:
        # 1. LOAD THE DATA FROM THE FILE
        print(f"Loading data from '{data_filename}'...")
        df_loaded = pd.read_csv(data_filename)
        
        # 2. GENERATE AND SAVE THE PLOTS
        print("Generating plots...")
        plot_infinite_kpca(df_loaded, filename=plot_filename)
        
    except FileNotFoundError:
        print(f"Error: Could not find '{data_filename}'. Please run the simulation first.")