#%%
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from scipy.ndimage import gaussian_filter
from scipy.stats import norm as sp_norm
from tqdm import tqdm

#%%


d_damage_translate = {
    0.0: 0.0,  # np.mean([0.00676 - 0.00526, 0.00413 - 0.00137]),
    0.014: 0.01,  # np.mean([0.01127 - 0.00526, 0.00841 - 0.00137]),
    0.047: 0.02,  # np.mean([0.02163 - 0.00526, 0.01881 - 0.00137]),
    0.138: 0.05,  # np.mean([0.05111 - 0.00524, 0.04824 - 0.00137]),
    0.303: 0.10,  # np.mean([0.10149 - 0.00523, 0.09855 - 0.00137]),
    0.466: 0.15,  # np.mean([0.15183 - 0.00518, 0.14900 - 0.00137]),
    0.626: 0.20,
    0.96: 0.30,  # np.mean([0.30046 - 0.00518, 0.29910 - 0.00141]),
}


#%%


def sdom(x):
    return x.std() / np.sqrt(len(x))


#%%


sim_columns = ["sim_species", "sim_damage", "sim_N_reads", "sim_length", "sim_seed"]


def split_name(name):
    splitted = name.split("-")
    damage, N_reads, length, seed = splitted[-4:]
    species = "-".join(splitted[1:-4])
    return species, float(damage), int(N_reads), int(length), int(seed)


def split_name_pd(name):
    return pd.Series(split_name(name), index=sim_columns)


def load_df(data_dir, path_parquet):

    if isinstance(data_dir, str):
        data_dir = Path(data_dir)

    results_dir = data_dir / "results"

    all_paths = list(results_dir.glob("*.parquet"))

    if path_parquet.exists():
        df_previous = pd.read_parquet(path_parquet)

        previous_names = set(df_previous["sample"])

        paths_to_load = []
        for path in all_paths:
            name = path.stem.removesuffix(".results")
            if name not in previous_names:
                paths_to_load.append(path)
    else:
        df_previous = None
        paths_to_load = all_paths

    if len(paths_to_load) == 0:
        return df_previous

    out = []
    for path in tqdm(paths_to_load):
        out.append(pd.read_parquet(path))

    if df_previous is None:
        df = pd.concat(out, ignore_index=True)
    else:
        df = pd.concat([df_previous, pd.concat(out)], ignore_index=True)

    return df


ALL_SPECIES = [
    "homo",
    "betula",
    "GC-low",
    "GC-mid",
    "GC-high",
    "contig1k",
    "contig10k",
    "contig100k",
]


def get_data_dir(species):
    if species in ALL_SPECIES:
        return Path("data") / species
    raise AssertionError(f"Unknown species: {species}")


def get_damaged_reads_path(species):
    if species in ALL_SPECIES:
        return f"damaged_reads_{species}.txt"
    raise AssertionError(f"Unknown species: {species}")


def load_results(species=ALL_SPECIES, use_columns_subset=True):

    data_dir = get_data_dir(species)
    path_parquet = data_dir / "df.parquet"
    df = load_df(data_dir, path_parquet)
    df.to_parquet(path_parquet)

    df["Bayesian_significance"] = df["Bayesian_D_max"] / df["Bayesian_D_max_std"]

    columns = [
        "sample",
        "tax_id",
        "N_reads",
        "D_max",
        "D_max_std",
        "Bayesian_D_max",
        "Bayesian_D_max_std",
        "significance",
        "Bayesian_significance",
        "Bayesian_prob_lt_5p_damage",
        "Bayesian_prob_lt_2p_damage",
        "Bayesian_prob_lt_1p_damage",
        "Bayesian_prob_lt_0.1p_damage",
        "Bayesian_prob_zero_damage",
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
        "lambda_LR",
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
        "non_CT_GA_damage_frequency_mean",
        "non_CT_GA_damage_frequency_std",
        "Bayesian_D_max_median",
        "Bayesian_D_max_confidence_interval_1_sigma_low",
        "Bayesian_D_max_confidence_interval_1_sigma_high",
        "Bayesian_D_max_confidence_interval_2_sigma_low",
        "Bayesian_D_max_confidence_interval_2_sigma_high",
        "Bayesian_D_max_confidence_interval_3_sigma_low",
        "Bayesian_D_max_confidence_interval_3_sigma_high",
        "Bayesian_D_max_confidence_interval_95_low",
        "Bayesian_D_max_confidence_interval_95_high",
        "Bayesian_A",
        "Bayesian_A_std",
        "Bayesian_A_median",
        "Bayesian_A_confidence_interval_1_sigma_low",
        "Bayesian_A_confidence_interval_1_sigma_high",
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
        "Bayesian_z",
        "var_L",
        "var_GC",
        "f+1",
        "f+15",
        "f-1",
        "f-15",
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

    # for col in sim_columns:
    #     df[col] = df[col].astype("category")

    df["sim_species"] = df["sim_species"].astype("category")

    df["sim_damage_percent"] = df["sim_damage"].map(d_damage_translate)

    df["Bayesian_prob_not_zero_damage"] = 1 - df["Bayesian_prob_zero_damage"]
    df["Bayesian_prob_gt_1p_damage"] = 1 - df["Bayesian_prob_lt_1p_damage"]

    return df


#%%


def load_multiple_species(species=ALL_SPECIES):
    if not (isinstance(species, list) or isinstance(species, tuple)):
        species = [species]
    dfs = [load_results(specie) for specie in species]
    return pd.concat(dfs, axis=0, ignore_index=True)


#%%


def get_df_damaged_reads(path):

    if isinstance(path, str):
        path = Path(path)

    if not path.exists():
        return None

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

    df_damaged_reads = pd.DataFrame(d).T.sort_values(sim_columns).reset_index(drop=True)

    df_damaged_reads["frac_damaged"] = df_damaged_reads["mod1000"] / (
        df_damaged_reads["mod1000"] + df_damaged_reads["mod0000"]
    )

    return df_damaged_reads


def load_multiple_damaged_reads(species):
    if not (isinstance(species, list) or isinstance(species, tuple)):
        species = [species]

    dfs = []
    for specie in species:
        damaged_reads_path = get_damaged_reads_path(specie)
        dfs.append(get_df_damaged_reads(damaged_reads_path))

    return pd.concat(dfs, axis=0, ignore_index=True)


#%%


def mean_of_CI_halfrange(group):
    s_low = "Bayesian_D_max_confidence_interval_1_sigma_low"
    s_high = "Bayesian_D_max_confidence_interval_1_sigma_high"
    return np.mean((group[s_high] - group[s_low]) / 2)


def mean_of_CI_range_low(group):
    return group["Bayesian_D_max_confidence_interval_1_sigma_low"].mean()


def mean_of_CI_range_high(group):
    return group["Bayesian_D_max_confidence_interval_1_sigma_"].mean()


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
            f"Bayesian_z_sdom": sdom(group[f"Bayesian_z"]),
            # Fit quality, MAP
            f"lambda_LR_mean": group[f"lambda_LR"].mean(),
            f"lambda_LR_std": group[f"lambda_LR"].std(),
            f"lambda_LR_sdom": sdom(group[f"lambda_LR"]),
            # damaged reads
            # f"reads_damaged": df_damaged_reads.,
        }

        if df_damaged_reads is not None:

            series_damaged_reads = get_damaged_reads(
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


def _from_low_high_to_errors(x_low, x_high):
    yerr = np.vstack([x_low, x_high])
    yerr_mean = yerr.mean(axis=0)
    yerr2 = yerr[1, :] - yerr_mean
    return yerr_mean, yerr2


def from_low_high_to_errors_corrected(x_low, x_center, x_high):
    # xerr = np.vstack([x_low, x_high])

    x_low = np.array(x_low)
    x_center = np.array(x_center)
    x_high = np.array(x_high)

    xerr = np.zeros((2, len(x_center)))
    xerr[0, :] = x_center - x_low
    xerr[1, :] = x_high - x_center
    return x_center, xerr


# y, sy = from_low_high_to_errors_corrected([0, 0, 0], [0.5, 0.25, 0], [1, 1, 1])
# plt.errorbar([0, 1, 2], y, yerr=sy, fmt="o")


#%%


def get_damaged_reads(
    df_damaged_reads,
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

    return df_damaged_reads.query(query[4:])


#%%


y_limits_individual_damage = {
    0.0: (0, 0.10),
    0.014: (0, 0.15),
    0.047: (0, 0.15),
    0.138: (0, 0.15),
    0.303: (0, 0.20),
    0.466: (0, 0.25),
    0.626: (0, 0.35),
    0.96: (0, 0.60),
}


def plot_individual_damage_result(
    df_in,
    group_all_species,
    df_damaged_reads=None,
    sim_length=60,
    x_lim=None,
    figsize=(15, 15),
    method="Bayesian",
):

    sim_damage = group_all_species["sim_damage"].iloc[0]
    sim_damage_percent = d_damage_translate[sim_damage]
    sim_N_reads = group_all_species["sim_N_reads"].iloc[0]

    all_species = df_in["sim_species"].unique()
    N_species = len(all_species)

    delta = 0.1

    if method.lower() == "bayesian":
        prefix = "Bayesian_"
        ylabel = r"Damage, $D$"
    else:
        prefix = ""
        ylabel = r"Damage, $D$, (MAP)"

    fig, axes = plt.subplots(figsize=figsize, nrows=N_species, sharex=True)

    if N_species == 1:
        axes = [axes]

    for i, (sim_species, ax) in enumerate(zip(all_species, axes)):
        # break

        query = f"sim_species == '{sim_species}' and sim_length == {sim_length}"
        group = group_all_species.query(query)

        str_mean_damaged_reads = ""
        if df_damaged_reads is not None:
            series_damaged_reads = get_damaged_reads(
                df_damaged_reads,
                sim_species=sim_species,
                sim_damage=sim_damage,
                sim_N_reads=sim_N_reads,
                sim_length=sim_length,
            )
            if len(series_damaged_reads) > 0:
                mean_damaged_reads = series_damaged_reads["frac_damaged"].mean()
                str_mean_damaged_reads = (
                    f", {mean_damaged_reads*100:.1f}"
                    r"\% damaged reads (mean) in fasta file"
                )

        ax.set(
            ylabel=ylabel,
            title=f"Species = {sim_species}{str_mean_damaged_reads}",
            ylim=y_limits_individual_damage[sim_damage],
        )

        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

        ax.axhline(
            sim_damage_percent,
            color="k",
            linestyle="--",
            label=f"{sim_damage_percent*100:.0f}" r"\%",
        )

        if len(group) == 0:
            continue

        x = group["sim_seed"]

        ax.errorbar(
            x - delta,
            group[f"{prefix}D_max"],
            group[f"{prefix}D_max_std"],
            fmt="o",
            color="C0",
            label=r"Mean $\pm$ std.",
            capsize=0,
        )

        if method.lower() == "bayesian":

            y, sy = from_low_high_to_errors_corrected(
                group["Bayesian_D_max_confidence_interval_1_sigma_low"],
                group["Bayesian_D_max_median"],
                group["Bayesian_D_max_confidence_interval_1_sigma_high"],
            )

            mask = sy[1, :] >= 0
            not_mask = np.logical_not(mask)

            ax.errorbar(
                x[mask] + delta,
                y[mask],
                sy[:, mask],
                fmt="s",
                # capsize=4,
                # capthick=1,
                label=r"Median $\pm$ 68\% C.I.",
                color="C1",
            )

            y2, sy2 = _from_low_high_to_errors(
                group["Bayesian_D_max_confidence_interval_1_sigma_low"],
                group["Bayesian_D_max_confidence_interval_1_sigma_high"],
            )

            ax.plot(
                x[not_mask] + delta,
                group["Bayesian_D_max_median"].values[not_mask],
                "s",
                color="C1",
                # label="Median",
            )

            ax.errorbar(
                x[not_mask] + delta,
                y2[not_mask],
                sy2[not_mask],
                fmt="None",
                # capsize=4,
                # capthick=1,
                color="C1",
                # label="68% C.I.",
            )

    ax.set(
        xlabel="Iteration",
        xlim=(-0.9, 100 - 0.1) if x_lim is None else x_lim,
    )

    fig.suptitle(
        f"Individual damages: \n"
        f"{sim_N_reads} reads\n"
        f"Briggs damage = {sim_damage}\n"
        f"Damage percent = {sim_damage_percent*100:.0f}"
        r"\% "
        "\n" + ylabel,
    )
    fig.subplots_adjust(top=0.85)

    leg_kws = dict(loc="upper right", markerscale=0.75)
    if method.lower() == "bayesian":
        handles, labels = axes[0].get_legend_handles_labels()
        order = [1, 2, 0]
        axes[0].legend(
            [handles[idx] for idx in order],
            [labels[idx] for idx in order],
            **leg_kws,
        )
    else:
        axes[0].legend(**leg_kws)
    return fig


#%%


def plot_individual_damage_results(df, df_damaged_reads, suffix=""):

    filename = Path(f"figures/bayesian/bayesian_individual_damage_results{suffix}.pdf")
    filename.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(filename) as pdf:

        for _, group_all_species in tqdm(df.groupby(["sim_damage", "sim_N_reads"])):
            # break

            fig = plot_individual_damage_result(
                df_in=df,
                group_all_species=group_all_species,
                df_damaged_reads=df_damaged_reads,
                sim_length=60,
                method="Bayesian",
            )

            pdf.savefig(fig)
            plt.close()

    filename = Path(f"figures/MAP/MAP_individual_damage_results{suffix}.pdf")
    filename.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(filename) as pdf:

        for _, group_all_species in tqdm(df.groupby(["sim_damage", "sim_N_reads"])):

            fig = plot_individual_damage_result(
                df_in=df,
                group_all_species=group_all_species,
                df_damaged_reads=df_damaged_reads,
                sim_length=60,
                method="MAP",
            )

            pdf.savefig(fig)
            plt.close()


#%%


def plot_combined_damage_result(
    df,
    group_agg_all_species,
    df_damaged_reads=None,
    sim_length=60,
    method="Bayesian",
):

    sim_damage = group_agg_all_species["sim_damage"].iloc[0]
    sim_damage_percent = d_damage_translate[sim_damage]

    all_species = df["sim_species"].unique()
    N_species = len(all_species)

    delta = 0.07

    if method.lower() == "bayesian":
        prefix = "Bayesian_"
        ylabel = r"Damage, $D$"
    else:
        prefix = ""
        ylabel = r"Damage, $D$, (MAP)"

    fig, axes = plt.subplots(figsize=(15, 15), nrows=N_species, sharex=True)
    for i, (sim_species, ax) in enumerate(zip(all_species, axes)):
        # break

        query = f"sim_species == '{sim_species}' and sim_length == {sim_length}"
        group_agg = group_agg_all_species.query(query)

        ax.axhline(
            sim_damage_percent,
            color="k",
            linestyle="--",
            label=f"{sim_damage_percent*100:.0f}" + r"\%",
        )
        ax.set_xscale("log")

        ax.set(
            title=f"Species = {sim_species}",
            ylabel=ylabel,
            xlim=(0.8 * 10**1, 1.2 * 10**5),
            ylim=(0, 0.48),
        )

        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

        if len(group_agg) == 0:
            continue

        x = group_agg["sim_N_reads"]

        ax.errorbar(
            x * (1 - delta),
            group_agg[f"{prefix}D_max_mean_of_mean"],
            group_agg[f"{prefix}D_max_mean_of_std"],
            fmt="o",
            # capsize=4,
            # capthick=1,
            label="Mean of mean ± mean of std",
            color="C0",
        )

        if method.lower() == "bayesian":

            y, sy = from_low_high_to_errors_corrected(
                group_agg["Bayesian_D_max_median_of_CI_range_low"],
                group_agg["Bayesian_D_max_median_of_median"],
                group_agg["Bayesian_D_max_median_of_CI_range_high"],
            )

            mask = sy[1, :] >= 0
            not_mask = np.logical_not(mask)

            ax.errorbar(
                x.values[mask] * (1 + delta),
                y[mask],
                sy[:, mask],
                fmt="s",
                capsize=4,
                capthick=1,
                label=r"Median of median ± median of CI (68\%)",
                color="C1",
            )

            y2, sy2 = _from_low_high_to_errors(
                group_agg["Bayesian_D_max_median_of_CI_range_low"],
                group_agg["Bayesian_D_max_median_of_CI_range_high"],
            )

            ax.plot(
                x[not_mask] * (1 + delta),
                y[not_mask],
                "s",
                color="C1",
                # label="Median of median",
            )

            ax.errorbar(
                x[not_mask] * (1 + delta),
                y2[not_mask],
                sy2[not_mask],
                fmt="None",
                capsize=4,
                capthick=1,
                color="C1",
                # label="Median of CI (16%-84%)",
            )

        if df_damaged_reads is not None:

            for sim_N_reads, group_agg_N_reads in group_agg.groupby("sim_N_reads"):
                # break

                series_damaged_reads = get_damaged_reads(
                    df_damaged_reads,
                    sim_species=sim_species,
                    sim_damage=sim_damage,
                    sim_length=sim_length,
                    sim_N_reads=sim_N_reads,
                )

                str_mean_damaged_reads = ""
                if len(series_damaged_reads) > 0:
                    mean_damaged_reads = series_damaged_reads["frac_damaged"].mean()
                    str_mean_damaged_reads = f"{mean_damaged_reads*100:.1f}" + r"\%"

                top1 = (
                    group_agg_N_reads[f"{prefix}D_max_mean_of_mean"]
                    + group_agg_N_reads[f"{prefix}D_max_mean_of_std"]
                )
                if method.lower() == "bayesian":
                    top2 = group_agg_N_reads["Bayesian_D_max_median_of_CI_range_high"]
                    y = max([top1.iloc[0], top2.iloc[0]])
                else:
                    y = top1.iloc[0]

                ax.text(
                    sim_N_reads,
                    y * 1.02,
                    str_mean_damaged_reads,
                    ha="center",
                    va="bottom",
                    fontsize=6,
                )

    ax.set(xlabel="N_reads")
    fig.suptitle(
        f"{ylabel}\n"
        f"Briggs damage = {sim_damage}\n"
        f"Damage percent = {sim_damage_percent*100:.0f}" + r"\%",
    )

    if method.lower() == "bayesian":
        handles, labels = axes[0].get_legend_handles_labels()
        order = [1, 2, 0]
        axes[0].legend([handles[idx] for idx in order], [labels[idx] for idx in order])
    else:
        handles, labels = axes[0].get_legend_handles_labels()
        order = [1, 0]
        axes[0].legend([handles[idx] for idx in order], [labels[idx] for idx in order])

    return fig


#%%


def plot_combined_damage_results(
    df,
    df_aggregated,
    df_damaged_reads,
    suffix="",
):

    filename = Path(f"figures/bayesian/bayesian_combined_damage_results{suffix}.pdf")
    filename.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(filename) as pdf:

        for sim_damage, group_agg_all_species in tqdm(
            df_aggregated.groupby("sim_damage")
        ):
            # break

            fig = plot_combined_damage_result(
                df,
                group_agg_all_species=group_agg_all_species,
                df_damaged_reads=df_damaged_reads,
                sim_length=60,
                method="Bayesian",
            )
            pdf.savefig(fig)
            plt.close()

    filename = Path(f"figures/MAP/MAP_combined_damage_results{suffix}.pdf")
    filename.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(filename) as pdf:

        for sim_damage, group_agg_all_species in tqdm(
            df_aggregated.groupby("sim_damage")
        ):
            # break

            fig = plot_combined_damage_result(
                df,
                group_agg_all_species=group_agg_all_species,
                df_damaged_reads=df_damaged_reads,
                sim_length=60,
                method="MAP",
            )
            pdf.savefig(fig)
            plt.close()


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
        color="C0",
    )

    ax.set(
        xlabel="N_reads",
        ylabel="Bayesian z" if bayesian else "MAP, lambda_LR",
        title=f"Fit quality, {title}, sim_damage={sim_damage}, damage={damage*100:.2f}"
        + r"\%",
        # ylim=(0, None),
    )

    ax.set_xscale("log")

    # ax.legend()

    return fig


#%%


def get_df_frac(df_in, column, cut):

    out = []
    for (sim_species, sim_damage, sim_N_reads), group in df_in.groupby(
        ["sim_species", "sim_damage", "sim_N_reads"]
    ):

        numerator = (group[column] > cut).sum()
        denominator = len(group)
        frac = numerator / denominator

        out.append(
            {
                "sim_species": sim_species,
                "sim_damage": sim_damage,
                "sim_damage_percent": d_damage_translate[sim_damage],
                "sim_N_reads": sim_N_reads,
                "log10_sim_N_reads": np.log10(sim_N_reads),
                "frac": frac,
            }
        )

    df_fracs = pd.DataFrame(out)
    return df_fracs


#%%


def get_n_sigma_probability(n_sigma):
    return sp_norm.cdf(n_sigma) - sp_norm.cdf(-n_sigma)


CUTS = [2, 3, 4]


def get_contour_settings(cut_type, cuts=None, method="bayesian"):

    if method.lower() == "bayesian":
        column = "Bayesian_significance"
        title = "Bayesian Significance"
    else:
        column = "significance"
        title = "Significance (MAP)"

    if cut_type == "significance":
        contour_settings = {
            "column": column,
            "label_title": "Significance cut:",
            "figure_title": title,
            "cuts": CUTS if cuts is None else cuts,
            "cut_transform": lambda x: x,
            "label_template": lambda cut: f"{cut} " r"$\sigma$",
        }

    elif cut_type == "prob_not_zero_damage":
        contour_settings = {
            "column": "Bayesian_prob_not_zero_damage",
            "label_title": r"$P(D > 0\%)$ cut:",
            "figure_title": r"$P(D > 0\%)$",
            "cuts": CUTS if cuts is None else cuts,
            "cut_transform": get_n_sigma_probability,
            "label_template": lambda cut: f"{cut} " r"$\sigma$",
            # "cuts": [0.9, 0.95, 0.99, 0.999],
            # "label_template": lambda cut: f"{cut:.1%}",
        }

    elif cut_type == "prob_gt_1p_damage":
        contour_settings = {
            "column": "Bayesian_prob_gt_1p_damage",
            "label_title": r"$P(D > 1\%)$ cut:",
            "figure_title": r"$P(D > 1\%)$",
            "cuts": CUTS if cuts is None else cuts,
            "cut_transform": get_n_sigma_probability,
            "label_template": lambda cut: f"{cut} " r"$\sigma$",
            # "cuts": [0.9, 0.95, 0.99, 0.999],
            # "cuts": [get_n_sigma_probability(cut) for cut in [2, 3, 4]],
            # "label_template": lambda cut: f"{cut:.1%}",
        }

    else:
        raise ValueError(f"Unknown cut_type: {cut_type}")

    contour_settings["colors"] = ["C0", "C1", "C2", "C3", "C4"]
    contour_settings["levels"] = [0.5, 0.95]
    contour_settings["alphas"] = [0.5, 1.0]
    contour_settings["linestyles"] = ["dashed", "solid"]
    return contour_settings


#%%


# fig, ax = plt.subplots()

# contour_settings = get_contour_settings(
#     cut_type,
#     cuts=cuts,
#     method=method,
# )


def plot_contour_lines_on_ax(
    df,
    ax,
    contour_settings,
    sim_species,
    gaussian_noise=None,
    method="Bayesian",
):

    for cut, color in zip(contour_settings["cuts"], contour_settings["colors"]):
        # break

        cut_transformed = contour_settings["cut_transform"](cut)

        df_fracs = get_df_frac(
            df.query(f"sim_species == '{sim_species}'"),
            column=contour_settings["column"],
            cut=cut_transformed,
        )

        df_wide = pd.pivot(
            df_fracs,
            index="sim_damage_percent",
            columns="sim_N_reads",
            values="frac",
        )

        df_wide

        if gaussian_noise is None:
            data = df_wide.values

        else:
            data = gaussian_filter(df_wide.values, gaussian_noise)

        for level, alpha, linestyle in zip(
            contour_settings["levels"],
            contour_settings["alphas"],
            contour_settings["linestyles"],
        ):
            CS = ax.contour(
                df_wide.columns,
                df_wide.index,
                data,
                levels=[level],
                alpha=alpha,
                colors=color,
                linestyles=linestyle,
            )

        ax.plot(
            [np.nan, np.nan],
            [np.nan, np.nan],
            color=color,
            label=contour_settings["label_template"](cut),
        )

    ax.set(
        ylabel="Damage",
        xlabel="N reads",
        xscale="log",
        title=f"Species = {sim_species}",
    )
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    # if i == N_species - 1:
    # if i == 0:
    # if i >= 0:

    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1, 0.99),
        frameon=False,
        title=contour_settings["label_title"],
        alignment="right",
    )

    ax2 = ax.twinx()
    for level, alpha, linestyle in zip(
        contour_settings["levels"],
        contour_settings["alphas"],
        contour_settings["linestyles"],
    ):
        ax2.plot(
            np.nan,
            np.nan,
            ls=linestyle,
            label=f"{level*100:.0f}" + r"\%",
            c="black",
            alpha=alpha,
        )

    ax2.get_yaxis().set_visible(False)

    ax2.legend(
        loc="upper right",
        bbox_to_anchor=(1, 0.74),
        frameon=False,
        title="Sim. fraction:",
        alignment="right",
    )


# %%


def plot_contour_line(
    df,
    cut_type,
    gaussian_noise=None,
    cuts=None,
    method="Bayesian",
):

    contour_settings = get_contour_settings(
        cut_type,
        cuts=cuts,
        method=method,
    )

    all_species = df["sim_species"].unique()
    N_species = len(all_species)

    fig, axes = plt.subplots(figsize=(20, 5), ncols=N_species, sharey=True)
    for i, (sim_species, ax) in enumerate(zip(all_species, axes)):
        # break

        plot_contour_lines_on_ax(
            df,
            ax=ax,
            contour_settings=contour_settings,
            sim_species=sim_species,
            gaussian_noise=gaussian_noise,
            method=method,
        )

    fig.suptitle(contour_settings["figure_title"], fontsize=16)
    fig.subplots_adjust(top=0.85)
    return fig


#%%


def plot_contour_lines(df):

    filename = Path(f"figures/bayesian/bayesian_contours.pdf")
    filename.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(filename) as pdf:

        cut_types = [
            "significance",
            "prob_not_zero_damage",
            "prob_gt_1p_damage",
        ]

        for cut_type in cut_types:
            # break

            fig = plot_contour_line(
                df,
                cut_type,
                method="Bayesian",
                # gaussian_noise=0.5,
                # cuts=[1, 2, 3, 4],
            )

            pdf.savefig(fig)
            plt.close()

    filename = Path(f"figures/MAP/MAP_contours.pdf")
    filename.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(filename) as pdf:

        cut_types = [
            "significance",
            # "prob_not_zero_damage",
            # "prob_gt_1p_damage",
        ]

        for cut_type in cut_types:
            # break

            fig = plot_contour_line(
                df,
                cut_type,
                method="MAP",
                # gaussian_noise=0.5,
                # cuts=[1, 2, 3, 4],
            )

            pdf.savefig(fig)
            plt.close()


#%%


def plot_zero_damage_group(group_0, sim_N_reads, method="Bayesian"):

    if method.lower() == "bayesian":
        prefix = "Bayesian_"
        title_prefix = "Bayesian "
        ncols = 4
    else:
        prefix = ""
        title_prefix = ""
        ncols = 3

    fig, axes = plt.subplots(figsize=(20, 5), ncols=ncols)

    x0 = group_0[f"{prefix}D_max"].values
    axes[0].hist(
        x0,
        range=(0, 0.08),
        bins=100,
        histtype="step",
    )
    axes[0].set(
        xlabel=f"{title_prefix}D-max",
        ylabel="Counts",
        title=f"Max value: {np.nanmax(x0)*100:.3f}" + r"\%",
    )
    axes[0].xaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    x1 = group_0[f"{prefix}significance"].values
    axes[1].hist(
        x1,
        range=(0.2, 1.6),
        bins=100,
        histtype="step",
    )
    axes[1].set(
        xlabel=f"{title_prefix}significance",
        ylabel="Counts",
        title=f"Max value: {np.nanmax(x1):.3f}",
    )

    x2 = (group_0["Bayesian_D_max"] - group_0["Bayesian_D_max_std"]).values
    axes[2].hist(
        x2,
        range=(-0.15, 0.01),
        bins=100,
        histtype="step",
    )
    axes[2].set(
        xlabel=f"{title_prefix}(D_max - std)",
        ylabel="Counts",
        title=f"Max value: {np.nanmax(x2)*100:.3f}" + r"\%",
    )
    axes[2].xaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    if method.lower() == "bayesian":
        x3 = group_0["Bayesian_D_max_confidence_interval_1_sigma_low"].values
        axes[3].hist(
            x3,
            range=(0, 0.001),
            bins=100,
            histtype="step",
        )
        axes[3].set(
            xlabel="Bayesian D_max C.I. low",
            ylabel="Counts",
            title=f"Max value: {np.nanmax(x3)*100:.3f}" + r"\%",
        )
        axes[3].xaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    fig.suptitle(f"sim_N_reads = {sim_N_reads}, # = {len(group_0)}", fontsize=16)
    fig.subplots_adjust(top=0.85)

    return fig


#%%


def plot_zero_damage_groups(df_0):

    filename = Path(f"figures/bayesian/bayesian_zero_damage_plots.pdf")
    filename.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(filename) as pdf:
        for sim_N_reads, group_0 in tqdm(df_0.groupby("sim_N_reads")):
            fig = plot_zero_damage_group(
                group_0,
                sim_N_reads,
                method="Bayesian",
            )
            pdf.savefig(fig)
            plt.close()

    filename = Path(f"figures/MAP/MAP_zero_damage_plots.pdf")
    filename.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(filename) as pdf:

        for sim_N_reads, group_0 in tqdm(df_0.groupby("sim_N_reads")):

            # if sim_N_reads == 25000:
            #     break

            fig = plot_zero_damage_group(
                group_0,
                sim_N_reads,
                method="MAP",
            )
            pdf.savefig(fig)
            plt.close()


#%%


def plot_individual_damage_results_length(
    df_in,
    group_all_lengths,
    df_damaged_reads=None,
    sim_species="homo",
    x_lim=None,
    figsize=(15, 10),
    method="Bayesian",
):

    sim_damage = group_all_lengths["sim_damage"].iloc[0]
    sim_damage_percent = d_damage_translate[sim_damage]
    sim_N_reads = group_all_lengths["sim_N_reads"].iloc[0]

    all_lengths = df_in["sim_length"].unique()
    N_lengths = len(all_lengths)

    delta = 0.1

    if method.lower() == "bayesian":
        prefix = "Bayesian_"
        ylabel = "Bayesian D_max"

    else:
        prefix = ""
        ylabel = "D_max (MAP)"

    fig, axes = plt.subplots(figsize=figsize, nrows=N_lengths, sharex=True)

    if N_lengths == 1:
        axes = [axes]

    for i, (sim_length, ax) in enumerate(zip(all_lengths, axes)):
        # break

        query = f"sim_species == '{sim_species}' and sim_length == {sim_length}"
        group = group_all_lengths.query(query)

        str_mean_damaged_reads = ""
        if df_damaged_reads is not None:
            series_damaged_reads = get_damaged_reads(
                df_damaged_reads,
                sim_species=sim_species,
                sim_damage=sim_damage,
                sim_N_reads=sim_N_reads,
                sim_length=sim_length,
            )
            if len(series_damaged_reads) > 0:
                mean_damaged_reads = series_damaged_reads["frac_damaged"].mean()
                str_mean_damaged_reads = (
                    f", {mean_damaged_reads*100:.1f}"
                    r"\% damaged reads (mean) in fasta file"
                )

        ax.set(
            ylabel=ylabel,
            title=f"Mean Fragment Length = {sim_length}{str_mean_damaged_reads}",
            ylim=y_limits_individual_damage[sim_damage],
        )

        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

        ax.axhline(
            sim_damage_percent,
            color="k",
            linestyle="--",
            label=f"{sim_damage_percent*100:.0f}" r"\%",
        )

        if len(group) == 0:
            continue

        x = group["sim_seed"]

        ax.errorbar(
            x - delta,
            group["Bayesian_D_max"],
            group["Bayesian_D_max_std"],
            fmt="o",
            color="C0",
            label="Mean ± std",
        )

        if method.lower() == "bayesian":

            y, sy = from_low_high_to_errors_corrected(
                group["Bayesian_D_max_confidence_interval_1_sigma_low"],
                group["Bayesian_D_max_median"],
                group["Bayesian_D_max_confidence_interval_1_sigma_high"],
            )

            mask = sy[1, :] >= 0
            not_mask = np.logical_not(mask)

            ax.errorbar(
                x[mask] + delta,
                y[mask],
                sy[:, mask],
                fmt="s",
                capsize=4,
                capthick=1,
                label="Median ± 68% C.I.",
                color="C1",
            )

            y2, sy2 = _from_low_high_to_errors(
                group["Bayesian_D_max_confidence_interval_1_sigma_low"],
                group["Bayesian_D_max_confidence_interval_1_sigma_high"],
            )

            ax.plot(
                x[not_mask] + delta,
                group["Bayesian_D_max_median"].values[not_mask],
                "s",
                color="C1",
                # label="Median",
            )

            ax.errorbar(
                x[not_mask] + delta,
                y2[not_mask],
                sy2[not_mask],
                fmt="None",
                capsize=4,
                capthick=1,
                color="C1",
                # label="68% C.I.",
            )

    ax.set(
        xlabel="Iteration",
        xlim=(-0.9, 100 - 0.1) if x_lim is None else x_lim,
    )

    fig.suptitle(
        f"{ylabel} \n"
        f"Individual damages: \n"
        f"{sim_N_reads} reads\n"
        f"Briggs damage = {sim_damage}\n"
        f"Damage percent = {sim_damage_percent*100:.0f}"
        r"\%",
    )
    fig.subplots_adjust(top=0.8)

    if method.lower() == "bayesian":

        try:
            handles, labels = axes[0].get_legend_handles_labels()
            order = [1, 2, 0]
            axes[0].legend(
                [handles[idx] for idx in order], [labels[idx] for idx in order]
            )
        except IndexError:
            s = f"IndexError for sim_damage = {sim_damage}, sim_N_reads = {sim_N_reads}"
            print(s)

    else:
        handles, labels = axes[0].get_legend_handles_labels()
        order = [1, 0]
        axes[0].legend([handles[idx] for idx in order], [labels[idx] for idx in order])

    return fig


#%%


def plot_individual_damage_results_lengths(df_homo_99, df_damaged_reads):

    filename = Path(f"figures/bayesian/bayesian_individual_damage_results_lengths.pdf")
    filename.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(filename) as pdf:

        for (sim_damage, sim_N_reads), group_all_lengths in tqdm(
            df_homo_99.groupby(["sim_damage", "sim_N_reads"])
        ):

            fig = plot_individual_damage_results_length(
                df_in=df_homo_99,
                group_all_lengths=group_all_lengths,
                df_damaged_reads=df_damaged_reads,
                sim_species="homo",
                method="Bayesian",
            )

            pdf.savefig(fig)
            plt.close()

    filename = Path(f"figures/MAP/MAP_individual_damage_results_lengths.pdf")
    filename.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(filename) as pdf:

        for (sim_damage, sim_N_reads), group_all_lengths in tqdm(
            df_homo_99.groupby(["sim_damage", "sim_N_reads"])
        ):

            fig = plot_individual_damage_results_length(
                df_in=df_homo_99,
                group_all_lengths=group_all_lengths,
                df_damaged_reads=df_damaged_reads,
                sim_species="homo",
                method="MAP",
            )

            pdf.savefig(fig)
            plt.close()


#%%


def plot_combined_damage_results_length(
    df_in,
    group_agg_all_lengths,
    df_damaged_reads=None,
    sim_species="homo",
    method="Bayesian",
):

    sim_damage = group_agg_all_lengths["sim_damage"].iloc[0]
    sim_damage_percent = d_damage_translate[sim_damage]

    all_lengths = df_in["sim_length"].unique()
    N_length = len(all_lengths)

    delta = 0.07

    if method.lower() == "bayesian":
        prefix = "Bayesian_"
        ylabel = "Bayesian D_max"

    else:
        prefix = ""
        ylabel = "D_max (MAP)"

    fig, axes = plt.subplots(figsize=(15, 12), nrows=N_length, sharex=True)
    for i, (sim_length, ax) in enumerate(zip(all_lengths, axes)):
        # break

        query = f"sim_species == '{sim_species}' and sim_length == {sim_length}"
        group_agg = group_agg_all_lengths.query(query)

        ax.axhline(
            sim_damage_percent,
            color="k",
            linestyle="--",
            label=f"{sim_damage_percent*100:.0f}" r"\%",
        )
        ax.set_xscale("log")

        ax.set(
            title=f"Mean Fragment Length = {sim_length}",
            ylabel=ylabel,
            ylim=(0, 0.48),
        )

        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

        if len(group_agg) == 0:
            continue

        x = group_agg["sim_N_reads"]

        ax.errorbar(
            x * (1 - delta),
            group_agg[f"{prefix}D_max_mean_of_mean"],
            group_agg[f"{prefix}D_max_mean_of_std"],
            fmt="o",
            # capsize=4,
            # capthick=1,
            label="Mean of mean ± mean of std",
            color="C0",
        )

        if method.lower() == "bayesian":

            y, sy = from_low_high_to_errors_corrected(
                group_agg["Bayesian_D_max_median_of_CI_range_low"],
                group_agg["Bayesian_D_max_median_of_median"],
                group_agg["Bayesian_D_max_median_of_CI_range_high"],
            )

            mask = sy[1, :] >= 0
            not_mask = np.logical_not(mask)

            ax.errorbar(
                x.values[mask] * (1 + delta),
                y[mask],
                sy[:, mask],
                fmt="s",
                capsize=4,
                capthick=1,
                label="Median of median ± median of CI (16%-84%)",
                color="C1",
            )

            y2, sy2 = _from_low_high_to_errors(
                group_agg["Bayesian_D_max_median_of_CI_range_low"],
                group_agg["Bayesian_D_max_median_of_CI_range_high"],
            )

            ax.plot(
                x[not_mask] * (1 + delta),
                y[not_mask],
                "s",
                color="C1",
                # label="Median of median",
            )

            ax.errorbar(
                x[not_mask] * (1 + delta),
                y2[not_mask],
                sy2[not_mask],
                fmt="None",
                capsize=4,
                capthick=1,
                color="C1",
                # label="Median of CI (16%-84%)",
            )

        if df_damaged_reads is not None:

            for sim_N_reads, group_agg_N_reads in group_agg.groupby("sim_N_reads"):
                # break

                series_damaged_reads = get_damaged_reads(
                    df_damaged_reads,
                    sim_species=sim_species,
                    sim_damage=sim_damage,
                    sim_length=sim_length,
                    sim_N_reads=sim_N_reads,
                )

                str_mean_damaged_reads = ""
                if len(series_damaged_reads) > 0:
                    mean_damaged_reads = series_damaged_reads["frac_damaged"].mean()
                    str_mean_damaged_reads = f"{mean_damaged_reads*100:.1f}" r"\%"

                top1 = (
                    group_agg_N_reads[f"{prefix}D_max_mean_of_mean"]
                    + group_agg_N_reads[f"{prefix}D_max_mean_of_std"]
                )
                if method.lower() == "bayesian":
                    top2 = group_agg_N_reads["Bayesian_D_max_median_of_CI_range_high"]
                    y = max([top1.iloc[0], top2.iloc[0]])
                else:
                    y = top1.iloc[0]

                ax.text(
                    sim_N_reads,
                    y * 1.02,
                    str_mean_damaged_reads,
                    ha="center",
                    va="bottom",
                    fontsize=6,
                )

    ax.set(xlabel="N_reads")
    fig.suptitle(
        f"{ylabel}\n"
        f"Briggs damage = {sim_damage}\n"
        f"Damage percent = {sim_damage_percent*100:.0f}"
        r"\%",
    )

    if method.lower() == "bayesian":

        try:
            handles, labels = axes[0].get_legend_handles_labels()
            order = [1, 2, 0]
            axes[0].legend(
                [handles[idx] for idx in order], [labels[idx] for idx in order]
            )

        except IndexError:
            print(f"IndexError for sim_damage = {sim_damage}")
    else:
        handles, labels = axes[0].get_legend_handles_labels()
        order = [1, 0]
        axes[0].legend([handles[idx] for idx in order], [labels[idx] for idx in order])

    return fig


#%%


def plot_combined_damage_results_lengths(
    df_homo_99,
    df_aggregated_lengths,
    df_damaged_reads,
):

    filename = Path(f"figures/bayesian/bayesian_combined_damage_results_lengths.pdf")
    filename.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(filename) as pdf:

        for sim_damage, group_agg_all_lengths in tqdm(
            df_aggregated_lengths.groupby("sim_damage")
        ):

            fig = plot_combined_damage_results_length(
                df_homo_99,
                group_agg_all_lengths=group_agg_all_lengths,
                df_damaged_reads=df_damaged_reads,
                sim_species="homo",
                method="Bayesian",
            )
            pdf.savefig(fig)

            plt.close()

    filename = Path(f"figures/MAP/MAP_combined_damage_results_lengths.pdf")
    filename.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(filename) as pdf:

        for sim_damage, group_agg_all_lengths in tqdm(
            df_aggregated_lengths.groupby("sim_damage")
        ):

            fig = plot_combined_damage_results_length(
                df_homo_99,
                group_agg_all_lengths=group_agg_all_lengths,
                df_damaged_reads=df_damaged_reads,
                sim_species="homo",
                method="MAP",
            )
            pdf.savefig(fig)

            plt.close()


# %%


# def plot_individual_damage_results_contig(
#     df_in,
#     group_all_contigs,
#     df_damaged_reads=None,
#     x_lim=None,
#     figsize=(15, 10),
#     method="Bayesian",
#     sim_length=60,
# ):

#     sim_damage = group_all_contigs["sim_damage"].iloc[0]
#     sim_damage_percent = d_damage_translate[sim_damage]
#     sim_N_reads = group_all_contigs["sim_N_reads"].iloc[0]

#     all_contigs = df_in["sim_species"].unique()
#     N_contigs = len(all_contigs)

#     delta = 0.1

#     if method.lower() == "bayesian":
#         prefix = "Bayesian_"
#         ylabel = "Bayesian D_max"

#     else:
#         prefix = ""
#         ylabel = "D_max (MAP)"

#     fig, axes = plt.subplots(figsize=figsize, nrows=N_contigs, sharex=True)

#     if N_contigs == 1:
#         axes = [axes]

#     for i, (sim_species, ax) in enumerate(zip(all_contigs, axes)):
#         # break

#         query = f"sim_species == '{sim_species}'"
#         group = group_all_contigs.query(query)

#         str_mean_damaged_reads = ""
#         if df_damaged_reads is not None:
#             series_damaged_reads = get_damaged_reads(
#                 df_damaged_reads,
#                 sim_species=sim_species,
#                 sim_damage=sim_damage,
#                 sim_N_reads=sim_N_reads,
#                 sim_length=sim_length,
#             )
#             if len(series_damaged_reads) > 0:
#                 mean_damaged_reads = series_damaged_reads["frac_damaged"].mean()
#                 str_mean_damaged_reads = (
#                     f", {mean_damaged_reads:.1%} damaged reads (mean) in fasta file"
#                 )

#         ax.set(
#             ylabel=ylabel,
#             title=f"Species = {sim_species}{str_mean_damaged_reads}",
#             ylim=y_limits_individual_damage[sim_damage],
#         )

#         ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

#         ax.axhline(
#             sim_damage_percent,
#             color="k",
#             linestyle="--",
#             label=f"{sim_damage_percent:.0%}",
#         )

#         if len(group) == 0:
#             continue

#         x = group["sim_seed"]

#         ax.errorbar(
#             x - delta,
#             group["Bayesian_D_max"],
#             group["Bayesian_D_max_std"],
#             fmt="o",
#             color="C0",
#             label="Mean ± std",
#         )

#         if method.lower() == "bayesian":

#             y, sy = from_low_high_to_errors_corrected(
#                 group["Bayesian_D_max_confidence_interval_1_sigma_low"],
#                 group["Bayesian_D_max_median"],
#                 group["Bayesian_D_max_confidence_interval_1_sigma_high"],
#             )

#             mask = sy[1, :] >= 0
#             not_mask = np.logical_not(mask)

#             ax.errorbar(
#                 x[mask] + delta,
#                 y[mask],
#                 sy[:, mask],
#                 fmt="s",
#                 capsize=4,
#                 capthick=1,
#                 label="Median ± 68% C.I.",
#                 color="C1",
#             )

#             y2, sy2 = _from_low_high_to_errors(
#                 group["Bayesian_D_max_confidence_interval_1_sigma_low"],
#                 group["Bayesian_D_max_confidence_interval_1_sigma_high"],
#             )

#             ax.plot(
#                 x[not_mask] + delta,
#                 group["Bayesian_D_max_median"].values[not_mask],
#                 "s",
#                 color="C1",
#                 # label="Median",
#             )

#             ax.errorbar(
#                 x[not_mask] + delta,
#                 y2[not_mask],
#                 sy2[not_mask],
#                 fmt="None",
#                 capsize=4,
#                 capthick=1,
#                 color="C1",
#                 # label="68% C.I.",
#             )

#     ax.set(
#         xlabel="Iteration",
#         xlim=(-0.9, 100 - 0.1) if x_lim is None else x_lim,
#     )

#     fig.suptitle(
#         f"{ylabel} \n"
#         f"Individual damages: \n"
#         f"{sim_N_reads} reads\n"
#         f"Briggs damage = {sim_damage}\n"
#         f"Damage percent = {sim_damage_percent:.0%}",
#     )
#     fig.subplots_adjust(top=0.82)

#     if method.lower() == "bayesian":

#         try:
#             handles, labels = axes[0].get_legend_handles_labels()
#             order = [1, 2, 0]
#             axes[0].legend(
#                 [handles[idx] for idx in order], [labels[idx] for idx in order]
#             )
#         except IndexError:
#             s = f"IndexError for sim_damage = {sim_damage}, sim_N_reads = {sim_N_reads}"
#             print(s)

#     else:
#         handles, labels = axes[0].get_legend_handles_labels()
#         order = [1, 0]
#         axes[0].legend([handles[idx] for idx in order], [labels[idx] for idx in order])

#     return fig


# def plot_individual_damage_results_contigs(df_contigs, df_damaged_reads):

#     filename = Path(f"figures/bayesian/bayesian_individual_damage_results_contigs.pdf")
#     filename.parent.mkdir(parents=True, exist_ok=True)

#     with PdfPages(filename) as pdf:

#         for (sim_damage, sim_N_reads), group_all_contigs in tqdm(
#             df_contigs.groupby(["sim_damage", "sim_N_reads"])
#         ):

#             fig = plot_individual_damage_results_contig(
#                 df_in=df_contigs,
#                 group_all_contigs=group_all_contigs,
#                 df_damaged_reads=df_damaged_reads,
#                 method="Bayesian",
#             )

#             pdf.savefig(fig)
#             plt.close()

#     filename = Path(f"figures/MAP/MAP_individual_damage_results_contigs.pdf")
#     filename.parent.mkdir(parents=True, exist_ok=True)

#     with PdfPages(filename) as pdf:

#         for (sim_damage, sim_N_reads), group_all_contigs in tqdm(
#             df_contigs.groupby(["sim_damage", "sim_N_reads"])
#         ):

#             fig = plot_individual_damage_results_contig(
#                 df_in=df_contigs,
#                 group_all_contigs=group_all_contigs,
#                 df_damaged_reads=df_damaged_reads,
#                 method="MAP",
#             )

#             pdf.savefig(fig)
#             plt.close()


#%%

# # np.set_printoptions(suppress=True)
# from scipy.interpolate import griddata


# def get_CS_locations(points, values, y_axis_positions, levels):

#     # N_reads = np.linspace(points[:, 1].min(), points[:, 1].max(), 1000)
#     # N_reads = np.linspace(0, 500, 10 + 1)
#     N_reads = np.logspace(
#         np.log10(points[:, 1].min()), np.log10(points[:, 1].max()), 1000
#     )

#     grid_x, grid_y = np.meshgrid(
#         y_axis_positions,
#         N_reads,
#         # np.log10(N_reads),
#     )

#     grid_z1 = griddata(points, values, (grid_x, grid_y), method="cubic")

#     manual_locations = []
#     for i, level in enumerate(levels):
#         # break
#         N_read_position = N_reads[np.abs(grid_z1[:, i] - level).argmin()]
#         manual_locations.append((N_read_position, y_axis_positions[i]))
#     manual_locations

#     return manual_locations


#%%

# # significance_cut = 3

# df["log10_sim_N_reads"] = np.log10(df["sim_N_reads"])
# df["log10_Bayesian_D_max_significance"] = np.log10(df["Bayesian_significance"])
# df["log10_Bayesian_prob_zero_damage"] = np.log10(df["Bayesian_prob_zero_damage"])
# df["log10_Bayesian_prob_lt_1p_damage"] = np.log10(df["Bayesian_prob_lt_1p_damage"])

# #%%


# xys = [
#     ("Bayesian_significance", "Bayesian_D_max"),
#     ("Bayesian_prob_lt_1p_damage", "Bayesian_D_max"),
#     ("Bayesian_prob_zero_damage", "Bayesian_D_max"),
#     ("Bayesian_prob_lt_1p_damage", "Bayesian_significance"),
#     ("Bayesian_prob_zero_damage", "Bayesian_significance"),
#     ("Bayesian_prob_lt_1p_damage", "Bayesian_prob_zero_damage"),
# ]


# xys = [
#     ("Bayesian_significance", "Bayesian_D_max"),
#     ("log10_Bayesian_prob_lt_1p_damage", "Bayesian_D_max"),
#     ("log10_Bayesian_prob_zero_damage", "Bayesian_D_max"),
#     ("log10_Bayesian_prob_lt_1p_damage", "Bayesian_significance"),
#     ("log10_Bayesian_prob_zero_damage", "Bayesian_significance"),
#     ("log10_Bayesian_prob_lt_1p_damage", "log10_Bayesian_prob_zero_damage"),
# ]


# for xy in tqdm(xys):

#     fig, ax = plt.subplots(figsize=(10, 6))
#     sns.scatterplot(
#         data=df,
#         x=xy[0],
#         y=xy[1],
#         hue="sim_damage_percent",
#         palette="deep",
#         size="sim_N_reads",
#         legend=False,
#         sizes=(2, 100),
#         alpha=0.5,
#         ax=ax,
#     )

#     x_str = xy[0].replace("Bayesian_", "")
#     y_str = xy[1].replace("Bayesian_", "")

#     ax.set(title=f"Bayesian, {x_str} vs {y_str}", xlabel=x_str, ylabel=y_str)

#     fig.savefig(f"figures/comparison_{species}_{xy[0]}_vs_{xy[1]}.pdf")

#     # plt.close("all")


# #%%

# columns = [
#     "Bayesian_significance",
#     "log10_Bayesian_prob_lt_1p_damage",
#     "log10_Bayesian_prob_zero_damage",
#     "Bayesian_D_max",
# ]

# g = sns.PairGrid(
#     df,
#     vars=columns,
#     hue="sim_damage_percent",
#     palette="deep",
#     diag_sharey=False,
#     corner=True,
# )

# g.map_diag(
#     sns.histplot,
#     log_scale=(False, True),
#     element="step",
#     fill=False,
# )
# # g.map_diag(sns.kdeplot, log_scale=(False, True))
# g.map_lower(
#     sns.scatterplot,
#     size=df["sim_N_reads"],
#     sizes=(2, 100),
#     alpha=0.5,
# )

# # g.add_legend()
# g.add_legend(
#     title="Legend:",
#     adjust_subtitles=True,
# )

# # g.tight_layout()

# g.figure.savefig(f"figures/comparison_{species}_pairgrid.pdf")
