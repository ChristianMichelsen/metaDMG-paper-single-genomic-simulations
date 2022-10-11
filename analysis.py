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

# species = "homo"
# species = "betula"
# species = "GC-low"
# species = "GC-mid"

all_species = [
    "homo",
    "betula",
    "GC-low",
    # "GC-mid",
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


x = x

#%%


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


#%%

# significance_cut = 3

df["log10_sim_N_reads"] = np.log10(df["sim_N_reads"])
df["log10_Bayesian_D_max_significance"] = np.log10(df["Bayesian_D_max_significance"])
df["log10_Bayesian_prob_zero_damage"] = np.log10(df["Bayesian_prob_zero_damage"])
df["log10_Bayesian_prob_lt_1p_damage"] = np.log10(df["Bayesian_prob_lt_1p_damage"])

#%%


xys = [
    ("Bayesian_D_max_significance", "Bayesian_D_max"),
    ("Bayesian_prob_lt_1p_damage", "Bayesian_D_max"),
    ("Bayesian_prob_zero_damage", "Bayesian_D_max"),
    ("Bayesian_prob_lt_1p_damage", "Bayesian_D_max_significance"),
    ("Bayesian_prob_zero_damage", "Bayesian_D_max_significance"),
    ("Bayesian_prob_lt_1p_damage", "Bayesian_prob_zero_damage"),
]


xys = [
    ("Bayesian_D_max_significance", "Bayesian_D_max"),
    ("log10_Bayesian_prob_lt_1p_damage", "Bayesian_D_max"),
    ("log10_Bayesian_prob_zero_damage", "Bayesian_D_max"),
    ("log10_Bayesian_prob_lt_1p_damage", "Bayesian_D_max_significance"),
    ("log10_Bayesian_prob_zero_damage", "Bayesian_D_max_significance"),
    ("log10_Bayesian_prob_lt_1p_damage", "log10_Bayesian_prob_zero_damage"),
]


for xy in tqdm(xys):

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x=xy[0],
        y=xy[1],
        hue="sim_damage_percent",
        palette="deep",
        size="sim_N_reads",
        legend=False,
        sizes=(2, 100),
        alpha=0.5,
        ax=ax,
    )

    x_str = xy[0].replace("Bayesian_", "")
    y_str = xy[1].replace("Bayesian_", "")

    ax.set(title=f"Bayesian, {x_str} vs {y_str}", xlabel=x_str, ylabel=y_str)

    fig.savefig(f"figures/comparison_{species}_{xy[0]}_vs_{xy[1]}.pdf")

    # plt.close("all")


#%%

columns = [
    "Bayesian_D_max_significance",
    "log10_Bayesian_prob_lt_1p_damage",
    "log10_Bayesian_prob_zero_damage",
    "Bayesian_D_max",
]

g = sns.PairGrid(
    df,
    vars=columns,
    hue="sim_damage_percent",
    palette="deep",
    diag_sharey=False,
    corner=True,
)

g.map_diag(
    sns.histplot,
    log_scale=(False, True),
    element="step",
    fill=False,
)
# g.map_diag(sns.kdeplot, log_scale=(False, True))
g.map_lower(
    sns.scatterplot,
    size=df["sim_N_reads"],
    sizes=(2, 100),
    alpha=0.5,
)

# g.add_legend()
g.add_legend(
    title="Legend:",
    adjust_subtitles=True,
)

# g.tight_layout()

g.figure.savefig(f"figures/comparison_{species}_pairgrid.pdf")


#%%


#%%


filename = f"figures/{species}_contour_significance_damage_N_reads.pdf"
with PdfPages(filename) as pdf:

    for significance_cut in [1, 2, 3, 4]:

        # significance_cut = 3
        reload(utils)
        df_fracs_significance = utils.get_df_frac_significance(
            df, significance_cut=significance_cut
        )

        # Reshape from long to wide
        df_wide_significance = pd.pivot(
            df_fracs_significance,
            index="sim_damage_percent",
            columns="sim_N_reads",
            values="frac",
        )

        df_wide_significance

        from scipy.ndimage import gaussian_filter

        reload(utils)

        data = df_wide_significance.values
        data = gaussian_filter(df_wide_significance.values, 0.5)

        levels = [0.1, 0.25, 0.5, 0.75]
        damage_positions = np.array([0.25, 0.22, 0.19, 0.16])

        fig, ax = plt.subplots()
        CS = ax.contour(
            df_wide_significance.columns,
            df_wide_significance.index,
            data,
            levels=levels,
            colors="k",
        )

        points = df_fracs_significance[
            ["sim_damage_percent", "log10_sim_N_reads"]
        ].values
        values = df_fracs_significance["frac"].values

        manual_locations = utils.get_CS_locations(
            points,
            values,
            damage_positions,
            levels,
        )

        ax.clabel(
            CS,
            inline=True,
            fontsize=10,
            manual=manual_locations,
        )

        ax.set(
            ylabel="Damage",
            xlabel="N reads",
            xscale="log",
            title=f"Significance cut = {significance_cut}",
        )

        pdf.savefig(fig)
        plt.close()

# %%

reload(utils)


columns = [
    "Bayesian_prob_zero_damage",
    "Bayesian_prob_lt_0.1p_damage",
    "Bayesian_prob_lt_1p_damage",
    "Bayesian_prob_lt_2p_damage",
    "Bayesian_prob_lt_5p_damage",
]

for column in tqdm(columns):

    filename = f"figures/{species}_contour_{column}_damage_N_reads.pdf"
    with PdfPages(filename) as pdf:

        # prob = 0.5

        for prob in [0.5, 0.1, 0.01, 0.001]:

            df_fracs_prob_zero = utils.get_df_frac_prob(
                df,
                prob=prob,
                column=column,
            )

            # Reshape from long to wide
            df_wide_prob_zero = pd.pivot(
                df_fracs_prob_zero,
                index="sim_damage_percent",
                columns="sim_N_reads",
                values="frac",
            )

            df_wide_prob_zero

            from scipy.ndimage import gaussian_filter

            reload(utils)

            data = df_wide_prob_zero.values
            # data = gaussian_filter(df_wide_prob_zero.values, 0.5)

            levels = [0.1, 0.25, 0.5, 0.75]
            damage_positions = np.array([0.25, 0.22, 0.19, 0.16])

            fig, ax = plt.subplots()
            CS = ax.contour(
                df_wide_prob_zero.columns,
                df_wide_prob_zero.index,
                data,
                levels=levels,
                colors="k",
            )

            points = df_fracs_prob_zero[
                ["sim_damage_percent", "log10_sim_N_reads"]
            ].values
            values = df_fracs_prob_zero["frac"].values

            manual_locations = utils.get_CS_locations(
                points,
                values,
                damage_positions,
                levels,
            )

            ax.clabel(
                CS,
                inline=True,
                fontsize=10,
                manual=manual_locations,
            )

            ax.set(
                ylabel="Damage",
                xlabel="N reads",
                xscale="log",
                title=f"column = {column}, prob = {prob}",
            )

            pdf.savefig(fig)
            plt.close()

# %%


def get_df_frac_damage(df):

    significance_cuts = np.linspace(0, 5, 100 + 1)

    out = []
    for (sim_damage, sim_N_reads), group in df.groupby(["sim_damage", "sim_N_reads"]):

        for significance_cut in significance_cuts:

            numerator = (group["Bayesian_D_max_significance"] > significance_cut).sum()
            denominator = len(group)
            frac = numerator / denominator
            out.append(
                {
                    "sim_damage": sim_damage,
                    "sim_damage_percent": d_damage_translate[sim_damage],
                    "sim_N_reads": sim_N_reads,
                    "log10_sim_N_reads": np.log10(sim_N_reads),
                    "significance_cut": significance_cut,
                    "frac": frac,
                }
            )

    df_fracs = pd.DataFrame(out)
    return df_fracs


df_fracs_significance = get_df_frac_damage(df)


filename = f"figures/{species}_contour_damage_{column}_N_reads.pdf"
with PdfPages(filename) as pdf:

    for sim_damage_percent, group in df_fracs_significance.groupby(
        "sim_damage_percent"
    ):
        # if sim_damage_percent == 0.1:
        # break

        # Reshape from long to wide
        df_wide_significance = pd.pivot(
            group,
            index="significance_cut",
            columns="sim_N_reads",
            values="frac",
        )

        df_wide_significance

        from scipy.ndimage import gaussian_filter

        reload(utils)

        data = df_wide_significance.values
        # data = gaussian_filter(df_wide_significance.values, 0.5)

        levels = [0.1, 0.25, 0.5, 0.75]

        fig, ax = plt.subplots()
        CS = ax.contour(
            df_wide_significance.columns,
            df_wide_significance.index,
            data,
            levels=levels,
            colors="k",
        )

        significance_positions = np.array([1, 2, 3, 4])
        points = group[["significance_cut", "log10_sim_N_reads"]].values
        values = group["frac"].values

        reload(utils)
        manual_locations = utils.get_CS_locations(
            points,
            values,
            y_axis_positions=significance_positions,
            levels=levels,
        )

        ax.clabel(
            CS,
            inline=True,
            fontsize=10,
            manual=manual_locations,
        )

        ax.set(
            ylabel="Significance cut",
            xlabel="N reads",
            xscale="log",
            title=f"sim_damage_percent = {sim_damage_percent}",
        )

        pdf.savefig(fig)
        plt.close()

# %%


def get_df_frac_prob(df, column):

    # prob_cuts = np.linspace(0, 0.5, 100 + 1)
    prob_cuts = np.logspace(-5, 0, 100 + 1)

    out = []
    for (sim_damage, sim_N_reads), group in df.groupby(["sim_damage", "sim_N_reads"]):

        for prob_cut in prob_cuts:

            numerator = (group[column] < prob_cut).sum()
            denominator = len(group)
            frac = numerator / denominator
            out.append(
                {
                    "sim_damage": sim_damage,
                    "sim_damage_percent": d_damage_translate[sim_damage],
                    "sim_N_reads": sim_N_reads,
                    "log10_sim_N_reads": np.log10(sim_N_reads),
                    "prob_cut": prob_cut,
                    "log_prob_cut": np.log10(prob_cut),
                    "frac": frac,
                }
            )

    df_fracs = pd.DataFrame(out)
    return df_fracs


for column in columns:
    # break

    df_fracs_prob = get_df_frac_prob(df, column)

    filename = f"figures/{species}_contour_damage_{column}_N_reads.pdf"
    with PdfPages(filename) as pdf:

        for sim_damage_percent, group in df_fracs_prob.groupby("sim_damage_percent"):
            # if sim_damage_percent == 0.1:
            #     break

            # Reshape from long to wide
            df_wide_prob = pd.pivot(
                group,
                index="prob_cut",
                columns="sim_N_reads",
                values="frac",
            )

            df_wide_prob

            from scipy.ndimage import gaussian_filter

            reload(utils)

            data = df_wide_prob.values
            # data = gaussian_filter(df_wide_prob.values, 0.5)

            levels = [0.1, 0.25, 0.5, 0.75]

            fig, ax = plt.subplots()
            CS = ax.contour(
                df_wide_prob.columns,
                df_wide_prob.index,
                data,
                levels=levels,
                colors="k",
            )

            # ax.set(xlim=(0, 500))

            prob_positions = np.array([0.6, 0.45, 0.3, 0.15])
            points = group[["prob_cut", "sim_N_reads"]].values
            values = group["frac"].values

            reload(utils)
            manual_locations = utils.get_CS_locations(
                points,
                values,
                y_axis_positions=prob_positions,
                levels=levels,
            )

            # manual_locations = [ (10.0, 0.6), (40., 0.45), (80., 0.3), (200., 0.15), ]

            ax.clabel(
                CS,
                inline=True,
                fontsize=10,
                manual=manual_locations,
            )

            ax.set(
                ylabel=column,
                xlabel="N reads",
                xscale="log",
                # yscale="log",
                title=f"sim_damage_percent = {sim_damage_percent}",
            )

            pdf.savefig(fig)
            plt.close()

# %%
