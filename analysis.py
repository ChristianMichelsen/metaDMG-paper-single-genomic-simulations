#%%

from importlib import reload
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

import utils

#%%

save_plots = False
save_plots = True

#%%

d_damage_translate = utils.d_damage_translate

#%%

# reload(utils)
df = utils.load_results()


#%%

path = "reads_damages.txt"

df_read_damages = utils.get_read_damages(path)
df_read_damages

#%%

if save_plots:

    reload(utils)

    for sim_species, dfg in df.groupby("sim_species"):

        filename = f"figures/individual_damage_{sim_species}_bayesian.pdf"

        with PdfPages(filename) as pdf_bayesian:

            for it in tqdm(dfg.groupby(["sim_damage", "sim_N_reads"])):

                (sim_damage, sim_N_reads), group_all_length = it

                fig_bayesian = utils.plot_single_group(
                    df_read_damages,
                    group_all_length,
                    sim_species,
                    sim_damage,
                    sim_N_reads,
                    bayesian=True,
                )
                pdf_bayesian.savefig(fig_bayesian)
                plt.close()


#%%


# if save_plots:

#     reload(utils)

#     for sim_species, dfg in df.groupby("sim_species"):
#         # break

#         with PdfPages(
#             f"figures/individual_fit_quality_{sim_species}_bayesian.pdf"
#         ) as pdf_bayesian:  # , PdfPages(
#             # f"figures/individual_fit_quality_{sim_species}_{sim_length}_MAP.pdf"
#             # ) as pdf_MAP:

#             for (sim_damage, sim_N_reads), group in tqdm(
#                 dfg.groupby(["sim_damage", "sim_N_reads"])
#             ):

#                 # break

#                 fig_bayesian = utils.plot_single_group_fit_quality(
#                     group,
#                     sim_damage,
#                     sim_N_reads,
#                     bayesian=True,
#                 )
#                 pdf_bayesian.savefig(fig_bayesian)
#                 plt.close()

#                 # fig_MAP = utils.plot_single_group_fit_quality(
#                 #     group,
#                 #     sim_damage,
#                 #     sim_N_reads,
#                 #     bayesian=False,
#                 # )
#                 # pdf_MAP.savefig(fig_MAP)
#                 # plt.close()


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


def get_df_aggregated(df, df_read_damages):

    dfg = df.groupby(["sim_species", "sim_length", "sim_damage", "sim_N_reads"])

    out = []
    for (sim_species, sim_length, sim_damage, sim_N_reads), group in dfg:
        # break

        prefix = "Bayesian_D_max"
        prefix_MAP = "D_max"

        s_damaged_reads = utils.get_damaged_reads(
            df_read_damages,
            sim_species,
            sim_damage,
            sim_N_reads,
            sim_length,
        )

        d = {
            "sim_damage": sim_damage,
            "damage": d_damage_translate[sim_damage],
            "sim_species": sim_species,
            "sim_length": sim_length,
            "sim_N_reads": sim_N_reads,
            "N_simulations": len(group),
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
            # damaged reads
            # f"reads_damaged": df_read_damages.,
        }

        if len(s_damaged_reads) > 1:
            d["reads_damaged"] = s_damaged_reads["mod1000"].median()
            d["reads_non_damaged"] = s_damaged_reads["mod0000"].median()
            d["reads_damaged_fraction"] = s_damaged_reads["frac_damaged"].mean()
        else:
            d["reads_damaged"] = np.nan
            d["reads_non_damaged"] = np.nan
            d["reads_damaged_fraction"] = np.nan
        out.append(d)

    df_aggregated = pd.DataFrame(out)

    return df_aggregated


#%%

df_aggregated = get_df_aggregated(df, df_read_damages)


# %%

if save_plots:

    reload(utils)

    for sim_species, dfg_agg in df_aggregated.groupby("sim_species"):
        # break

        filename = f"figures/combined_damage_{sim_species}_bayesian.pdf"
        with PdfPages(filename) as pdf:

            for it in tqdm(dfg_agg.groupby("sim_damage")):
                # break

                sim_damage, group_agg_all_lengths = it

                fig = utils.plot_single_group_agg(
                    df_read_damages,
                    group_agg_all_lengths,
                    sim_species,
                    sim_damage,
                    bayesian=True,
                )
                pdf.savefig(fig)
                plt.close()


# %%


# if save_plots:

#     reload(utils)

#     for (sim_species, sim_length), dfg_agg in tqdm(
#         df_aggregated.groupby(["sim_species", "sim_length"])
#     ):

#         with PdfPages(
#             f"figures/combined_fit_quality_{sim_species}_{sim_length}_bayesian.pdf"
#         ) as pdf:
#             for sim_damage, group_agg in dfg_agg.groupby("sim_damage"):
#                 fig = utils.plot_single_group_agg_fit_quality(
#                     group_agg,
#                     sim_damage,
#                     bayesian=True,
#                 )
#                 pdf.savefig(fig)
#                 plt.close()
