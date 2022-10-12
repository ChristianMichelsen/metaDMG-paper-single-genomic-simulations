#%%

from importlib import reload
from multiprocessing.spawn import get_executable
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

all_species = [
    "homo",
    "betula",
    "GC-low",
    "GC-mid",
]

#%%

reload(utils)

df = utils.load_multiple_species(all_species)


#%%

reload(utils)

df_damaged_reads = utils.load_multiple_damaged_reads(all_species)


#%%


reload(utils)

filename = f"figures/individual_damage_results_bayesian.pdf"
with PdfPages(filename) as pdf_bayesian:

    for _, group_all_species in tqdm(df.groupby(["sim_damage", "sim_N_reads"])):

        fig_bayesian = utils.plot_individual_damage_results(
            df,
            group_all_species,
            df_damaged_reads=df_damaged_reads,
            sim_length=60,
        )

        pdf_bayesian.savefig(fig_bayesian)
        plt.close()


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


def get_df_aggregated(df, df_damaged_reads):

    dfg = df.groupby(["sim_species", "sim_length", "sim_damage", "sim_N_reads"])

    out = []
    for (sim_species, sim_length, sim_damage, sim_N_reads), group in dfg:
        # break

        prefix = "Bayesian_D_max"
        prefix_MAP = "D_max"

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
            # f"reads_damaged": df_damaged_reads.,
        }

        if df_damaged_reads is not None:

            series_damaged_reads = utils.get_damaged_reads(
                df_damaged_reads,
                sim_species,
                sim_damage,
                sim_N_reads,
                sim_length,
            )
            series = series_damaged_reads

            if len(series) > 1:
                d["reads_damaged"] = series["mod1000"].median()
                d["reads_non_damaged"] = series["mod0000"].median()
                d["reads_damaged_fraction"] = series["frac_damaged"].mean()
            else:
                d["reads_damaged"] = np.nan
                d["reads_non_damaged"] = np.nan
                d["reads_damaged_fraction"] = np.nan
        out.append(d)

    df_aggregated = pd.DataFrame(out)

    return df_aggregated


#%%

df_aggregated = get_df_aggregated(df, df_damaged_reads)


# %%

reload(utils)

# for sim_species, dfg_agg in df_aggregated.groupby("sim_species"):
# break

filename = f"figures/combined_damage_results_bayesian.pdf"
with PdfPages(filename) as pdf:

    for sim_damage, group_agg_all_species in tqdm(df_aggregated.groupby("sim_damage")):

        fig = utils.plot_combined_damage_results(
            df,
            group_agg_all_species=group_agg_all_species,
            df_damaged_reads=df_damaged_reads,
            sim_length=60,
        )
        pdf.savefig(fig)
        plt.close()


# %%


reload(utils)
cut_types = ["prob_gt_1p_damage", "prob_not_zero_damage", "significance"]

filename = f"figures/contours_bayesian.pdf"
with PdfPages(filename) as pdf:

    for cut_type in cut_types:
        fig = utils.plot_contour_lines(df, cut_type)
        pdf.savefig(fig)
        plt.close()
