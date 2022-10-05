#%%
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

#%%


d_damage_translate = {
    0.0: 0.0,  # np.mean([0.00676 - 0.00526, 0.00413 - 0.00137]),
    0.014: 0.01,  # np.mean([0.01127 - 0.00526, 0.00841 - 0.00137]),
    0.047: 0.02,  # np.mean([0.02163 - 0.00526, 0.01881 - 0.00137]),
    0.138: 0.05,  # np.mean([0.05111 - 0.00524, 0.04824 - 0.00137]),
    0.303: 0.10,  # np.mean([0.10149 - 0.00523, 0.09855 - 0.00137]),
    0.466: 0.15,  # np.mean([0.15183 - 0.00518, 0.14900 - 0.00137]),
    0.96: 0.30,  # np.mean([0.30046 - 0.00518, 0.29910 - 0.00141]),
}


#%%


def sdom(x):
    return x.std() / np.sqrt(len(x))


#%%


sim_columns = ["sim_species", "sim_damage", "sim_N_reads", "sim_length", "sim_seed"]


def split_name(name):
    _, species, damage, N_reads, length, seed = name.split("-")
    return species, float(damage), int(N_reads), int(length), int(seed)


def split_name_pd(name):
    return pd.Series(split_name(name), index=sim_columns)


def load_results(use_columns_subset=True):

    path = Path("data/df.parquet")

    if path.exists():
        df = pd.read_parquet(path)
    else:

        results_dir = Path("data") / "results"
        out = []
        for path in tqdm(list(results_dir.glob("*.parquet"))):
            out.append(pd.read_parquet(path))
        df = pd.concat(out)
        df.to_parquet(path)

    columns = [
        "tax_id",
        "sample",
        "N_reads",
        "lambda_LR",
        "D_max",
        "mean_L",
        "mean_GC",
        "q",
        "A",
        "c",
        "phi",
        "rho_Ac",
        "valid",
        "asymmetry",
        "std_L",
        "std_GC",
        "D_max_std",
        "q_std",
        "phi_std",
        "A_std",
        "c_std",
        "N_x=1_forward",
        "N_x=1_reverse",
        "N_sum_total",
        "N_sum_forward",
        "N_sum_reverse",
        "N_min",
        "k_sum_total",
        "k_sum_forward",
        "k_sum_reverse",
        "Bayesian_z",
        "Bayesian_D_max",
        "Bayesian_D_max_std",
        "Bayesian_D_max_median",
        "Bayesian_D_max_confidence_interval_1_sigma_low",
        "Bayesian_D_max_confidence_interval_1_sigma_high",
        "Bayesian_q",
        "Bayesian_q_std",
        "Bayesian_q_median",
        "Bayesian_q_confidence_interval_1_sigma_low",
        "Bayesian_q_confidence_interval_1_sigma_high",
        "Bayesian_c",
        "Bayesian_c_std",
        "Bayesian_c_median",
        "Bayesian_c_confidence_interval_1_sigma_low",
        "Bayesian_c_confidence_interval_1_sigma_high",
        "Bayesian_phi",
        "Bayesian_phi_std",
        "Bayesian_phi_median",
        "Bayesian_phi_confidence_interval_1_sigma_low",
        "Bayesian_phi_confidence_interval_1_sigma_high",
        "Bayesian_rho_Ac",
        "f+1",
        "f-1",
    ]

    if use_columns_subset:
        df = df.loc[:, columns]

    df[sim_columns] = (
        df["sample"]
        .apply(split_name_pd)
        .astype(
            {
                "sim_N_reads": "int",
                "sim_length": "int",
                "sim_seed": "int",
            }
        )
    )

    df = df.sort_values(sim_columns).reset_index(drop=True)

    for col in sim_columns:
        df[col] = df[col].astype("category")

    return df


#%%


def get_read_damages(path):

    with open(path) as f:
        d = {}
        for line in f:
            line = line.strip()
            if line.startswith("HEADER"):
                continue

            if line.startswith("sim"):
                filename = Path(line)

                species, damage, N_reads, length, seed = split_name(filename.stem)

                d[str(filename)] = {
                    "mod0000": 0,
                    "mod1000": 0,
                    "sim_species": species,
                    "sim_damage": damage,
                    "sim_N_reads": N_reads,
                    "sim_length": length,
                    "sim_seed": seed,
                }
            else:
                counts, key, _ = line.split(" ")
                d[str(filename)][key] = int(counts)

    df_read_damages = pd.DataFrame(d).T.sort_values(sim_columns).reset_index(drop=True)

    df_read_damages["frac_damaged"] = df_read_damages["mod1000"] / (
        df_read_damages["mod1000"] + df_read_damages["mod0000"]
    )

    return df_read_damages


#%%


def from_low_high_to_errors(x_low, x_high):
    yerr = np.vstack([x_low, x_high])
    yerr_mean = yerr.mean(axis=0)
    yerr2 = yerr[1, :] - yerr_mean
    return yerr_mean, yerr2


#%%


def get_damaged_reads(
    df_read_damages,
    sim_species=None,
    sim_damage=None,
    sim_N_reads=None,
    sim_length=None,
):

    query = ""

    if sim_species is not None:
        query += f"and sim_species == '{sim_species}' "

    if sim_damage is not None:
        query += f"and sim_damage == {sim_damage} "

    if sim_N_reads is not None:
        query += f"and sim_N_reads == {sim_N_reads} "

    if sim_length is not None:
        query += f"and sim_length == {sim_length} "

    return df_read_damages.query(query[4:])


#%%


def plot_single_group(
    df_read_damages,
    group_all_length,
    sim_species,
    sim_damage,
    sim_N_reads,
    bayesian=True,
):

    y_limits = {
        0.0: (0, 0.08),
        0.014: (0, 0.15),
        0.047: (0, 0.15),
        0.138: (0, 0.15),
        0.303: (0, 0.20),
        0.466: (0, 0.25),
        0.96: (0, 0.60),
    }

    delta = 0.1
    damage = d_damage_translate[sim_damage]

    fig, axes = plt.subplots(figsize=(15, 12), nrows=3, sharex=True)

    for (sim_length, group), ax in zip(group_all_length.groupby("sim_length"), axes):
        # break

        s_damaged_reads = get_damaged_reads(
            df_read_damages,
            sim_species,
            sim_damage,
            sim_N_reads,
            sim_length,
        )
        if len(s_damaged_reads) > 0:
            mean_damaged_reads = s_damaged_reads["frac_damaged"].mean()
            s_mean_damaged_reads = f", {mean_damaged_reads:.1%} damaged reads"
        else:
            s_mean_damaged_reads = ""

        x = np.arange(len(group))

        if bayesian:

            y, sy = from_low_high_to_errors(
                group["Bayesian_D_max_confidence_interval_1_sigma_low"],
                group["Bayesian_D_max_confidence_interval_1_sigma_high"],
            )

            ax.errorbar(
                x,
                y,
                sy,
                fmt="None",
                capsize=4,
                capthick=1,
                label="68% C.I.",
            )

            ax.plot(
                x - delta,
                group["Bayesian_D_max"],
                "o",
                color="C2",
                label="Mean",
            )
            ax.plot(
                x + delta,
                group["Bayesian_D_max_median"],
                "s",
                color="C3",
                label="Median",
            )

        else:

            ax.errorbar(
                x,
                group["D_max"],
                group["D_max_std"],
                fmt=".",
                capsize=4,
                capthick=1,
                label="Mean ± SD",
            )

        ax.axhline(
            damage,
            color="k",
            linestyle="--",
            label='"Truth"',
        )

        ax.set(
            # xlabel="index",
            ylabel="Bayesian D_max" if bayesian else "D_max",
            title=f"Simulated lengths = {sim_length}{s_mean_damaged_reads}",
            xlim=(-0.9, len(group) - 0.1),
            ylim=y_limits[sim_damage],
        )

        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    ax.set(xlabel="index")

    fig.suptitle(
        f"Species: {sim_species}\n"
        f"{sim_N_reads} reads\n"
        f"sim_damage = {sim_damage}, "
        f"damage = {damage:.2%}",
    )
    ax.set(xlabel="Random seed")

    if bayesian:
        # ax.legend()
        handles, labels = ax.get_legend_handles_labels()
        order = [0, 1, 2, 3, 2]
        ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order])

    # fig.tight_layout()

    return fig


#%%


# def plot_single_group_fit_quality(group, sim_damage, sim_N_reads, bayesian=True):

#     damage = d_damage_translate[sim_damage]

#     column = "Bayesian_z" if bayesian else "lambda_LR"

#     fig, ax = plt.subplots(figsize=(10, 6))

#     sns.histplot(
#         data=group,
#         x=column,
#         stat="density",
#         kde=True,
#         line_kws={"alpha": 0.5, "linestyle": "--"},
#         fill=False,
#         ax=ax,
#         color="grey",
#         # element="step",
#     )
#     sns.rugplot(
#         data=group,
#         x=column,
#         ax=ax,
#         color="grey",
#     )

#     ax.set(
#         xlabel="Bayesian z" if bayesian else "MAP, Likehood Ratio, lambda_LR",
#         title=f"Fit quality, N_reads={sim_N_reads}, sim_damage={sim_damage}, damage={damage:.2%}",
#     )

#     if bayesian:

#         x_limits = {
#             0.0: (-3, 1.5),
#             0.014: (-5, 2.5),
#             0.047: (-3.5, 5.5),
#             0.138: (0, 9.0),
#             0.303: (1, 12),
#             0.466: (1.5, 14),
#             0.96: (3, 16),
#         }

#     else:

#         x_limits = {
#             0.0: (0, 5),
#             0.014: (0, 17),
#             0.047: (0, 50),
#             0.138: (4, 95),
#             0.303: (5, 130),
#             0.466: (8, 150),
#             0.96: (25, 200),
#         }

#     # ax.set(xlim=x_limits[sim_damage])

#     return fig


#%%


def plot_single_group_agg_bayesian(
    df_read_damages,
    group_agg_all_lengths,
    sim_species,
    sim_damage,
):

    damage = d_damage_translate[sim_damage]
    delta = 0.07

    fig, axes = plt.subplots(figsize=(10, 10), nrows=3, sharex=True)

    for (sim_length, group_agg), ax in zip(
        group_agg_all_lengths.groupby("sim_length"), axes
    ):

        ax.plot(
            group_agg["sim_N_reads"] * (1 - delta),
            group_agg["Bayesian_D_max_mean_of_mean"],
            "o",
            label="Mean of mean",
            color="C2",
        )

        ax.errorbar(
            group_agg["sim_N_reads"] * (1 - delta),
            group_agg["Bayesian_D_max_mean_of_mean"],
            group_agg["Bayesian_D_max_mean_of_std"],
            fmt="None",
            capsize=4,
            capthick=1,
            label="Mean of std (±1σ)",
            color="C2",
        )

        ax.plot(
            group_agg["sim_N_reads"] * (1 + delta),
            group_agg["Bayesian_D_max_median_of_median"],
            "s",
            label="Median of median",
            color="C3",
        )

        y, sy = from_low_high_to_errors(
            group_agg["Bayesian_D_max_median_of_CI_range_low"],
            group_agg["Bayesian_D_max_median_of_CI_range_high"],
        )

        ax.errorbar(
            group_agg["sim_N_reads"] * (1 + delta),
            y,
            sy,
            fmt="None",
            capsize=4,
            capthick=1,
            label="Median of CI (16%-84%)",
            color="C3",
        )

        ax.axhline(
            damage,
            color="k",
            linestyle="--",
            label='"Truth"',
        )
        ax.set_xscale("log")

        ax.set(
            # xlabel="N_reads",
            title=f"Simulated lengths = {sim_length}",
            ylabel="Bayesian_D_max",
            ylim=(0, 0.48),
        )

        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    for (sim_length, group_agg), ax in zip(
        group_agg_all_lengths.groupby("sim_length"), axes
    ):

        for sim_N_reads, group_agg_N_reads in group_agg.groupby("sim_N_reads"):
            # break

            s_damaged_reads = get_damaged_reads(
                df_read_damages,
                sim_species=sim_species,
                sim_damage=sim_damage,
                sim_length=sim_length,
                sim_N_reads=sim_N_reads,
            )
            if len(s_damaged_reads) > 0:
                mean_damaged_reads = s_damaged_reads["frac_damaged"].mean()
                s_mean_damaged_reads = f"{mean_damaged_reads:.1%}"
            else:
                s_mean_damaged_reads = ""

            y1 = (
                group_agg_N_reads["Bayesian_D_max_mean_of_mean"]
                + group_agg_N_reads["Bayesian_D_max_mean_of_std"]
            )
            y2 = group_agg_N_reads["Bayesian_D_max_median_of_CI_range_high"]
            y = max([y1.iloc[0], y2.iloc[0]])
            ax.text(
                sim_N_reads,
                y * 1.02,
                s_mean_damaged_reads,
                ha="center",
                va="bottom",
                fontsize=6,
            )

    ax.set(xlabel="N_reads")
    fig.suptitle(
        "Bayesian D-max\n"
        f"Species: {sim_species}\n"
        f"sim_damage = {sim_damage}, "
        f"damage = {damage:.2%}",
    )

    handles, labels = ax.get_legend_handles_labels()
    order = [0, 3, 1, 4, 2]
    ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order])

    return fig


# def plot_single_group_agg_MAP(group_agg_all_lengths, sim_species, sim_damage):

#     damage = d_damage_translate[sim_damage]

#     fig, ax = plt.subplots(figsize=(10, 6))

#     ax.errorbar(
#         group_agg["sim_N_reads"],
#         group_agg["D_max_mean_of_mean"],
#         group_agg["D_max_mean_of_std"],
#         fmt="o",
#         capsize=4,
#         capthick=1,
#         label="MAP estimate",
#         color="C2",
#     )

#     ax.axhline(
#         damage,
#         color="k",
#         linestyle="--",
#         label='"Truth"',
#     )
#     ax.set_xscale("log")
#     ax.set(
#         xlabel="N_reads",
#         ylabel="Bayesian_D_max",
#         title=f"MAP D-max, sim_damage={sim_damage}, damage={damage:.2%}",
#         ylim=(0, None),
#     )

#     ax.legend()

#     return fig


def plot_single_group_agg(
    df_read_damages,
    group_agg_all_lengths,
    sim_species,
    sim_damage,
    bayesian=True,
):
    if bayesian:
        return plot_single_group_agg_bayesian(
            df_read_damages,
            group_agg_all_lengths,
            sim_species,
            sim_damage,
        )
    # else:
    #     return plot_single_group_agg_MAP(group_agg_all_lengths, sim_species, sim_damage)


#%%


def plot_single_group_agg_fit_quality(group_agg, sim_damage, bayesian=True):

    damage = d_damage_translate[sim_damage]

    fig, ax = plt.subplots(figsize=(10, 6))

    col = "Bayesian_z" if bayesian else "lambda_LR"
    title = "Bayesian z" if bayesian else "MAP, Likehood Ratio"

    ax.errorbar(
        group_agg["sim_N_reads"],
        group_agg[f"{col}_mean"],
        group_agg[f"{col}_std"],
        fmt="o",
        capsize=4,
        capthick=1,
        color="C2",
    )

    ax.set(
        xlabel="N_reads",
        ylabel="Bayesian z" if bayesian else "MAP, lambda_LR",
        title=f"Fit quality, {title}, sim_damage={sim_damage}, damage={damage:.2%}",
        # ylim=(0, None),
    )

    ax.set_xscale("log")

    # ax.legend()

    return fig
