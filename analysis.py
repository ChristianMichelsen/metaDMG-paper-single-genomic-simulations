#%%

from importlib import reload

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

import utils

#%%

save_plots = False

#%%

d_damage_translate = utils.d_damage_translate

#%%

df = utils.load_results()


#%%


if save_plots:

    dfg = df.groupby(["sim_damage", "sim_N_reads"])

    with PdfPages("figures/individual_damage_bayesian.pdf") as pdf_bayesian, PdfPages(
        "figures/individual_damage_MAP.pdf"
    ) as pdf_MAP:

        for (sim_damage, sim_N_reads), group in tqdm(dfg):

            fig_bayesian = utils.plot_single_group(
                group,
                sim_damage,
                sim_N_reads,
                bayesian=True,
            )
            pdf_bayesian.savefig(fig_bayesian)
            plt.close()

            fig_MAP = utils.plot_single_group(
                group,
                sim_damage,
                sim_N_reads,
                bayesian=False,
            )
            pdf_MAP.savefig(fig_MAP)
            plt.close()


#%%


if save_plots:

    reload(utils)

    dfg = df.groupby(["sim_damage", "sim_N_reads"])

    with PdfPages(
        "figures/individual_fit_quality_bayesian.pdf"
    ) as pdf_bayesian, PdfPages("figures/individual_fit_quality_MAP.pdf") as pdf_MAP:

        for (sim_damage, sim_N_reads), group in tqdm(dfg):

            fig_bayesian = utils.plot_single_group_fit_quality(
                group,
                sim_damage,
                sim_N_reads,
                bayesian=True,
            )
            pdf_bayesian.savefig(fig_bayesian)
            plt.close()

            fig_MAP = utils.plot_single_group_fit_quality(
                group,
                sim_damage,
                sim_N_reads,
                bayesian=False,
            )
            pdf_MAP.savefig(fig_MAP)
            plt.close()

#%%


#%%


def mean_of_CI_halfrange(group):
    s_low = "Bayesian_D_max_confidence_interval_1_sigma_low"
    s_high = "Bayesian_D_max_confidence_interval_1_sigma_high"
    return np.mean((group[s_high] - group[s_low]) / 2)


def mean_of_CI_range_low(group):
    return group["Bayesian_D_max_confidence_interval_1_sigma_low"].mean()


def mean_of_CI_range_high(group):
    return group["Bayesian_D_max_confidence_interval_1_sigma_"].mean()


#%%


def get_df_aggregated(df):

    dfg = df.groupby(["sim_damage", "sim_N_reads"])

    out = []
    for (sim_damage, sim_N_reads), group in dfg:
        # break

        prefix = "Bayesian_D_max"
        prefix_MAP = "D_max"

        out.append(
            {
                "sim_damage": sim_damage,
                "damage": d_damage_translate[sim_damage],
                "sim_N_reads": sim_N_reads,
                # Bayesian
                f"{prefix}_mean_of_mean": group[f"{prefix}"].mean(),
                f"{prefix}_mean_of_median": group[f"{prefix}_median"].mean(),
                f"{prefix}_median_of_median": group[f"{prefix}_median"].median(),
                f"{prefix}_std_of_mean": group[f"{prefix}"].std(),
                f"{prefix}_mean_of_std": group[f"{prefix}_std"].mean(),
                f"{prefix}_mean_of_CI_halfrange": mean_of_CI_halfrange(group),
                f"{prefix}_median_of_CI_range_low": group[
                    f"{prefix}_confidence_interval_1_sigma_low"
                ].median(),
                f"{prefix}_median_of_CI_range_high": group[
                    f"{prefix}_confidence_interval_1_sigma_high"
                ].median(),
                "N_simulations": len(group),
                # MAP
                f"{prefix_MAP}_mean_of_mean": group[f"{prefix_MAP}"].mean(),
                f"{prefix_MAP}_std_of_mean": group[f"{prefix_MAP}"].std(),
                f"{prefix_MAP}_mean_of_std": group[f"{prefix_MAP}_std"].mean(),
                # Fit quality, Bayesian
                f"Bayesian_z_mean": group[f"Bayesian_z"].mean(),
                f"Bayesian_z_std": group[f"Bayesian_z"].std(),
                f"Bayesian_z_sdom": utils.sdom(group[f"Bayesian_z"]),
                # Fit quality, MAP
                f"lambda_LR_mean": group[f"lambda_LR"].mean(),
                f"lambda_LR_std": group[f"lambda_LR"].std(),
                f"lambda_LR_sdom": utils.sdom(group[f"lambda_LR"]),
            }
        )

    df_aggregated = pd.DataFrame(out)

    return df_aggregated


#%%

df_aggregated = get_df_aggregated(df)


# %%

if save_plots:

    reload(utils)

    dfg_agg = df_aggregated.groupby("sim_damage")

    with PdfPages("figures/combined_damage_bayesian.pdf") as pdf:
        for sim_damage, group_agg in dfg_agg:
            fig = utils.plot_single_group_agg(group_agg, sim_damage, bayesian=True)
            pdf.savefig(fig)
            plt.close()

    with PdfPages("figures/combined_damage_MAP.pdf") as pdf:
        for sim_damage, group_agg in dfg_agg:
            fig = utils.plot_single_group_agg(group_agg, sim_damage, bayesian=False)
            pdf.savefig(fig)
            plt.close()


# %%


if save_plots:

    reload(utils)

    dfg_agg = df_aggregated.groupby("sim_damage")

    with PdfPages("figures/combined_fit_quality_bayesian.pdf") as pdf:
        for sim_damage, group_agg in dfg_agg:
            # break
            fig = utils.plot_single_group_agg_fit_quality(
                group_agg,
                sim_damage,
                bayesian=True,
            )
            pdf.savefig(fig)
            plt.close()

    with PdfPages("figures/combined_fit_quality_MAP.pdf") as pdf:
        for sim_damage, group_agg in dfg_agg:
            fig = utils.plot_single_group_agg_fit_quality(
                group_agg,
                sim_damage,
                bayesian=False,
            )
            pdf.savefig(fig)
            plt.close()

# %%
