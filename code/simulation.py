from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from data import simulate_dataset, f_t, g_x, h_x
from estimators import compute_acd, estimate_weighted_increments, get_rorr_estimates
from plot import plot_simulation


NUM_STRATA = 5
SEED = 1024
SAMPLE_SIZES = [10_000, 100_000, 1_000_000]


def run_simulation(sample_sizes):
    results_acd = []
    results_rorr = []
    results_aie = []

    true_acd = compute_acd(NUM_STRATA)

    for n in sample_sizes:
        data = simulate_dataset(n, NUM_STRATA, seed=SEED)

        # ACD Estimator
        acd_empirical = data.derivative.mean()
        acd_stderr = data.derivative.std() / np.sqrt(n)
        acd_ci = (acd_empirical - 1.96 * acd_stderr, acd_empirical + 1.96 * acd_stderr)
        results_acd.append([n, acd_empirical, f"({acd_ci[0]:.3f}, {acd_ci[1]:.3f})", true_acd])

        # RORR Estimator
        rorr_empirical, rorr_stderr, rorr_ci = get_rorr_estimates(data, g_x, h_x)
        rorr_target = (data.assign(plm_plim=lambda df: df.weight / (df.t_star + 1))).plm_plim.mean()
        results_rorr.append([n, rorr_empirical, f"({rorr_ci[0]:.3f}, {rorr_ci[1]:.3f})", rorr_target])

        # AIE Estimator
        aie_empirical, aie_stderr, aie_ci = estimate_weighted_increments(data, f_t, g_x, h_x)
        aie_target = data[data.t < data.t.max()].incremental.mean()
        results_aie.append([n, aie_empirical, f"({aie_ci[0]:.3f}, {aie_ci[1]:.3f})", aie_target])

    return results_acd, results_rorr, results_aie


def to_latex(df, estimator_name):
    latex = df.to_latex(index=False, float_format="%.3f")
    print(f"\nLaTeX Table for {estimator_name}:\n")
    print(latex)


if __name__ == "__main__":
    results_acd, results_rorr, results_aie = run_simulation(SAMPLE_SIZES)

    columns_acd = ["Sample Size", "Empirical ACD", "ACD CI", "ACD Target"]
    columns_rorr = ["Sample Size", "Empirical RORR", "RORR CI", "RORR Target"]
    columns_aie = ["Sample Size", "Empirical AIE", "AIE CI", "AIE Target"]

    df_acd = pd.DataFrame(results_acd, columns=columns_acd)
    df_rorr = pd.DataFrame(results_rorr, columns=columns_rorr)
    df_aie = pd.DataFrame(results_aie, columns=columns_aie)

    # NOTE: In the paper, we set the RORR and AIE targets to their empirical estimates with n=1m.
    to_latex(df_acd, "ACD Estimator")
    to_latex(df_rorr, "RORR Estimator")
    to_latex(df_aie, "AIE Estimator")

    plot_data = simulate_dataset(1_000_000, NUM_STRATA, seed=SEED)
    plot_simulation(plot_data).show()
    plt.savefig("figures/figure-1.png", dpi=300)
    # # Imports
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from matplotlib.backends.backend_pdf import PdfPages
# import os

# # Custom imports (assuming these modules exist based on previous requests)
# from data import simulate_dataset, f_t, g_x, h_x
# from estimators import (
#     compute_ocd,
#     compute_acdd,
#     estimate_weighted_increments,
#     get_rorr_estimates,
# )
# from plot import plot_simulation

# # --- Global constants for simulation
# NUM_STRATA = 5
# SEED = 1024
# SAMPLE_SIZES = [10000, 100000, 1000000]

# # --- Main simulation function
# def run_simulation(sample_sizes):
#     results_acd = []
#     results_rorr = []
#     results_aie = []

#     # Get the true value for ACD from the analytical formula
#     true_acd = compute_ocd(NUM_STRATA)

#     for n in sample_sizes:
#         data = simulate_dataset(n, NUM_STRATA, seed=SEED)

#         # ACD Estimator
#         acd_empirical, acd_std_err = compute_acdd(data)
#         acd_ci_lwr = acd_empirical - 1.96 * acd_std_err
#         acd_ci_upr = acd_empirical + 1.96 * acd_std_err
        
#         results_acd.append(
#             [n, acd_empirical, f"({acd_ci_lwr:.3f}, {acd_ci_upr:.3f})", true_acd]
#         )

#         # RORR Estimator
#         rorr_empirical, rorr_std_err, rorr_ci = get_rorr_estimates(
#             data, g_x, h_x
#         )
#         rorr_target = (
#             data.assign(plm=lambda df: df.weight / (df.t_star + 1)).plm.mean()
#         )
        
#         results_rorr.append(
#             [n, rorr_empirical, f"({rorr_ci[0]:.3f}, {rorr_ci[1]:.3f})", rorr_target]
#         )

#         # AIE Estimator
#         aie_results = estimate_weighted_increments(data, f_t, g_x, h_x)
#         aie_empirical = aie_results['estimate']
#         aie_ci = aie_results['ci']
#         aie_target = data[data.t < data.t.max()].incremental.mean()
        
#         results_aie.append(
#             [n, aie_empirical, f"({aie_ci[0]:.3f}, {aie_ci[1]:.3f})", aie_target]
#         )

#     return results_acd, results_rorr, results_aie


# # --- LaTeX formatting function
# def to_latex(df, estimator_name, float_format="%.3f"):
#     """
#     Prints a pandas DataFrame to a LaTeX table format.
#     """
#     latex_table = df.to_latex(index=False, float_format=float_format)
#     print(f"\\begin{{table}}[h!]\n\\centering\n\\caption{{{estimator_name} Results}}\n{latex_table}\\end{{table}}")


# # --- Main execution block
# if __name__ == "__main__":
#     results_acd, results_rorr, results_aie = run_simulation(SAMPLE_SIZES)

#     columns_acd = ["Sample Size", "Empirical ACD", "ACD CI", "ACD Target"]
#     columns_rorr = ["Sample Size", "Empirical RORR", "RORR CI", "RORR Target"]
#     columns_aie = ["Sample Size", "Empirical AIE", "AIE CI", "AIE Target"]

#     df_acd = pd.DataFrame(results_acd, columns=columns_acd)
#     df_rorr = pd.DataFrame(results_rorr, columns=columns_rorr)
#     df_aie = pd.DataFrame(results_aie, columns=columns_aie)

#     # Note: In the paper, we set the RORR and AIE targets to their empirical columns with n=...
#     # The code below is commented out, likely to be run separately for the final output.
#     # to_latex(df_acd, "ACD Estimator")
#     # to_latex(df_rorr, "RORR Estimator")
#     # to_latex(df_aie, "AIE Estimator")

#     # Plotting
#     plot_data = simulate_dataset(1_000_000, NUM_STRATA, seed=SEED)
    
#     # Assuming plot_simulation and savefig are defined in 'plot'
#     pdf = PdfPages('simulation_plots.pdf')
    
#     # The image shows `plot_simulation` call but no `pdf.savefig`
#     # and the code structure suggests a plot is being generated.
#     # The following is a logical conclusion based on standard plotting workflows.
#     fig = plot_simulation(plot_data)
#     # The next line is a logical guess to save the plot.
#     # pdf.savefig(fig, dpi=300)
#     # pdf.close()
    
#     # The code snippet `plt.savefig('kdd.pdf', dpi=300)` is also shown.
#     # We will include a placeholder for it since it directly matches the image.
#     # plt.savefig('kdd.pdf', dpi=300)