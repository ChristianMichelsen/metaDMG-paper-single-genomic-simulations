#%%
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

#%%


d_damage_translate = {
    0.0: np.mean([0.00676 - 0.00526, 0.00413 - 0.00137]),
    0.0001: np.mean([0.00680 - 0.00526, 0.00415 - 0.00137]),
    0.0010: np.mean([0.00712 - 0.00526, 0.00440 - 0.00137]),
    0.014: np.mean([0.01127 - 0.00526, 0.00841 - 0.00137]),
    0.047: np.mean([0.02163 - 0.00526, 0.01881 - 0.00137]),
    0.14: np.mean([0.05111 - 0.00524, 0.04824 - 0.00137]),
    0.30: np.mean([0.10149 - 0.00523, 0.09855 - 0.00137]),
    0.46: np.mean([0.15183 - 0.00518, 0.14900 - 0.00137]),
    0.615: np.mean([0.20136 - 0.00518, 0.19891 - 0.00140]),
    0.93: np.mean([0.30046 - 0.00518, 0.29910 - 0.00141]),
}


#%%


def split_name(name):
    _, damage, N_reads, seed = name.split("-")
    damage = float("0." + damage)
    return damage, int(N_reads), int(seed)


def split_name_pd(name):
    return pd.Series(split_name(name), index=["sim_damage", "sim_N_reads", "sim_seed"])


def load_results():

    df = pd.read_parquet("data/results")

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

    df = df.loc[:, columns]

    df[["sim_damage", "sim_N_reads", "sim_seed"]] = (
        df["sample"]
        .apply(split_name_pd)
        .astype({"sim_N_reads": "int", "sim_seed": "int"})
    )

    df = df.sort_values(["sim_damage", "sim_N_reads", "sim_seed"]).reset_index(
        drop=True
    )

    return df


#%%


def from_low_high_to_errors(x_low, x_high):
    yerr = np.vstack([x_low, x_high])
    yerr_mean = yerr.mean(axis=0)
    yerr2 = yerr[1, :] - yerr_mean
    return yerr_mean, yerr2


#%%


def plot_single_group(group, sim_damage, sim_N_reads, bayesian=True):

    y_limits = {
        0.0: (0, 0.08),
        0.0001: (0, 0.08),
        0.0010: (0, 0.08),
        0.014: (0, 0.10),
        0.047: (0, 0.12),
        0.14: (0, 0.15),
        0.30: (0, 0.20),
        0.46: (0, 0.25),
        0.615: (0.05, 0.35),
        0.93: (0.15, 0.40),
    }

    delta = 0.1
    damage = d_damage_translate[sim_damage]

    x = np.arange(len(group))

    fig, ax = plt.subplots(figsize=(15, 6))

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
        label="Truth",
    )
    ax.set(
        xlabel="index",
        ylabel="Bayesian D_max" if bayesian else "D_max",
        title=f"N_reads={sim_N_reads}, sim_damage={sim_damage}, damage={damage:.2%}",
    )

    ax.set(xlim=(-0.9, len(group) - 0.1), ylim=y_limits[sim_damage])

    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    if bayesian:
        # ax.legend()
        handles, labels = ax.get_legend_handles_labels()
        order = [0, 1, 2, 3, 2]
        ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order])

    return fig


#%%


def plot_single_group_agg_bayesian(group_agg, sim_damage):
    damage = d_damage_translate[sim_damage]
    delta = 0.07

    fig, ax = plt.subplots(figsize=(10, 6))

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
        xlabel="N_reads",
        ylabel="Bayesian_D_max",
        title=f"Bayesian D-max, sim_damage={sim_damage}, damage={damage:.2%}",
        ylim=(0, None),
    )

    handles, labels = ax.get_legend_handles_labels()
    order = [0, 3, 1, 4, 2]
    ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order])

    return fig


def plot_single_group_agg_MAP(group_agg, sim_damage):

    damage = d_damage_translate[sim_damage]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.errorbar(
        group_agg["sim_N_reads"],
        group_agg["D_max_mean_of_mean"],
        group_agg["D_max_mean_of_std"],
        fmt="o",
        capsize=4,
        capthick=1,
        label="MAP estimate",
        color="C2",
    )

    ax.axhline(
        damage,
        color="k",
        linestyle="--",
        label='"Truth"',
    )
    ax.set_xscale("log")
    ax.set(
        xlabel="N_reads",
        ylabel="Bayesian_D_max",
        title=f"MAP D-max, sim_damage={sim_damage}, damage={damage:.2%}",
        ylim=(0, None),
    )

    ax.legend()

    return fig


def plot_single_group_agg(group_agg, sim_damage, bayesian=True):
    if bayesian:
        return plot_single_group_agg_bayesian(group_agg, sim_damage)
    else:
        return plot_single_group_agg_MAP(group_agg, sim_damage)
