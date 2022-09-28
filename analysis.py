#%%

from importlib import reload

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

import utils

#%%

d_damage_translate = utils.d_damage_translate

#%%

df = utils.load_results()


#%%


dfg = df.groupby(["sim_damage", "sim_N_reads"])

with PdfPages("multipage_pdf_bayesian.pdf") as pdf_bayesian, PdfPages(
    "multipage_pdf_MAP.pdf"
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


def mean_of_CI_halfrange(group):
    s_low = "Bayesian_D_max_confidence_interval_1_sigma_low"
    s_high = "Bayesian_D_max_confidence_interval_1_sigma_high"
    return np.mean((group[s_high] - group[s_low]) / 2)


def mean_of_CI_range_low(group):
    return group["Bayesian_D_max_confidence_interval_1_sigma_low"].mean()


def mean_of_CI_range_high(group):
    return group["Bayesian_D_max_confidence_interval_1_sigma_"].mean()


#%%

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
            #
            f"{prefix_MAP}_mean_of_mean": group[f"{prefix_MAP}"].mean(),
            f"{prefix_MAP}_std_of_mean": group[f"{prefix_MAP}"].std(),
            f"{prefix_MAP}_mean_of_std": group[f"{prefix_MAP}_std"].mean(),
        }
    )

df_aggregated = pd.DataFrame(out)


# %%

reload(utils)


dfg_agg = df_aggregated.groupby("sim_damage")

with PdfPages("multipage_pdf_agg_bayesian.pdf") as pdf:
    for sim_damage, group_agg in dfg_agg:
        fig = utils.plot_single_group_agg(group_agg, sim_damage, bayesian=True)
        pdf.savefig(fig)
        plt.close()


with PdfPages("multipage_pdf_agg_MAP.pdf") as pdf:
    for sim_damage, group_agg in dfg_agg:
        fig = utils.plot_single_group_agg(group_agg, sim_damage, bayesian=False)
        pdf.savefig(fig)
        plt.close()


# %%
