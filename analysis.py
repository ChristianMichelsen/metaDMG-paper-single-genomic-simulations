#%%

from importlib import reload
from multiprocessing.spawn import get_executable
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

import utils

#%%

plt.style.use("plotstyle.mplstyle")

#%%

# save_plots = False
# save_plots = True


make_plots = False
# make_plots = True


#%%

# D_DAMAGE = utils.D_DAMAGE


#%%

all_species = [
    "homo",
    "betula",
    "GC-low",
    "GC-mid",
    "GC-high",
    "contig1k",
    "contig10k",
    "contig100k",
]

#%%

# reload(utils)
df_all = utils.load_multiple_species(all_species)

#%%

df = df_all.query(
    f"sim_length == 60 and sim_seed < 100 and sim_species in {all_species[:-3]}"
)

#%%


# reload(utils)
# df_damaged_reads = utils.load_multiple_damaged_reads(all_species)
df_damaged_reads = None

#%%

# reload(utils)
df_known_damage = utils.get_df_known_damage()


#%%

# x = x

#%%

reload(utils)
if make_plots:
    print("Plotting individual damage")
    utils.plot_individual_damage_results(
        df=df,
        df_known_damage=df_known_damage,
        df_damaged_reads=df_damaged_reads,
    )


#%%

reload(utils)
df_aggregated = utils.get_df_aggregated(
    df_in=df,
    df_known_damage=df_known_damage,
    df_damaged_reads=df_damaged_reads,
)


# %%

# reload(utils)
if make_plots:
    print("Plotting combined damage")
    utils.plot_combined_damage_results(
        df=df,
        df_aggregated=df_aggregated,
        df_known_damage=df_known_damage,
        df_damaged_reads=df_damaged_reads,
    )


#%%


df_aggregated_homo = df_aggregated.query("sim_species == 'homo'")

reload(utils)
if make_plots:
    fig = utils.plot_combined_MAEs(
        df_aggregated_homo=df_aggregated_homo,
        df_known_damage=df_known_damage,
        method="Bayesian",
        # ylim=(0, 2),
    )


# %%

reload(utils)
if make_plots:
    print("Plotting contour lines")
    utils.plot_contour_lines(df)


#%%


# reload(utils)
if make_plots:

    reload(utils)
    fig, ax = plt.subplots(figsize=(3.5, 3))

    utils.plot_contour_lines_on_ax(
        df,
        ax=ax,
        sim_species="homo",
        method="Bayesian",
        title="",
        frac_legend_pos=(1, 0.60),
        fill_contour=True,
        fill_cut_value=4,
        # fill_cut_value=2,
        fill_level_value=0.95,
        # fill_level_value=0.5,
    )

    ax.set(xlim=(20, 1.2 * 10**5))

    # fig.tight_layout()

    fig.savefig("figures/contour_homo.pdf", bbox_inches="tight")


#%%

# reload(utils)

df_zero_damage = df_all.query(
    "sim_species == 'homo' and sim_damage == 0 and sim_length == 60"
)

if make_plots:
    utils.plot_zero_damage_groups(df_zero_damage)


#%%

df_homo_99 = df_all.query("sim_species == 'homo' and sim_seed < 100")

reload(utils)
if make_plots:

    print("Plotting individual damage lengths")
    utils.plot_individual_damage_results_lengths(
        df_homo_99=df_homo_99,
        df_known_damage=df_known_damage,
        df_damaged_reads=df_damaged_reads,
    )


#%%

reload(utils)

df_aggregated_lengths = utils.get_df_aggregated(
    df_in=df_homo_99,
    df_known_damage=df_known_damage,
    df_damaged_reads=df_damaged_reads,
)

if make_plots:

    print("Plotting combined damage lengths")
    utils.plot_combined_damage_results_lengths(
        df_homo_99=df_homo_99,
        df_aggregated_lengths=df_aggregated_lengths,
        df_known_damage=df_known_damage,
        df_damaged_reads=df_damaged_reads,
    )


#%%


df_contigs = df_all.query(f"sim_species in {all_species[-3:]}")

reload(utils)
if make_plots:

    print("Plotting individual damage: contigs")
    utils.plot_individual_damage_results(
        df=df_contigs,
        df_known_damage=df_known_damage,
        df_damaged_reads=df_damaged_reads,
        suffix="_contigs",
    )

# %%

reload(utils)

df_aggregated_contigs = utils.get_df_aggregated(
    df_in=df_contigs,
    df_known_damage=df_known_damage,
    df_damaged_reads=df_damaged_reads,
)

# reload(utils)
if make_plots:
    print("Plotting combined damage: contigs")
    utils.plot_combined_damage_results(
        df=df_contigs,
        df_aggregated=df_aggregated_contigs,
        df_known_damage=df_known_damage,
        df_damaged_reads=df_damaged_reads,
        suffix="_contigs",
    )


#%%

sim_species = "homo"
sim_damage = 0.31
sim_N_reads = 100
sim_length = 60
min_seed = 60
max_seed = 80

group = df.query(
    f"sim_species == '{sim_species}'"
    f" and sim_damage == {sim_damage}"
    f" and sim_length == {sim_length}"
    f" and sim_N_reads == {sim_N_reads}"
    f" and {min_seed} <= sim_seed < {max_seed}"
)


group_agg = df_aggregated.query(
    f"sim_species == '{sim_species}'"
    f" and sim_damage == {sim_damage}"
    f" and sim_length == {sim_length}"
    # f" and sim_N_reads == {sim_N_reads}"
    # f" and sim_seed < {max_seed}"
)


#%%

import arviz as az
from scipy.stats import beta as sp_beta
from scipy.stats import betabinom as sp_betabinom
from scipy.stats import norm as sp_norm

posterior = az.from_netcdf("sim-homo-0.31-100-60-69.nc").posterior


A = posterior["A"].values[0]
phi = posterior["phi"].values[0]
N = 16
mu = np.mean(A)
stds = np.sqrt(A * (1 - A) * (phi + N) / ((phi + 1) * N))
std = np.mean(stds)


#%%

fig, ax = plt.subplots()
xmin, xmax = 0, 0.18
Nbins = 50
ax.hist(A, Nbins, range=(xmin, xmax), density=True, histtype="step", label=r"$\bar{D}$")
ax.hist(
    stds, Nbins, range=(xmin, xmax), density=True, histtype="step", label=r"$\sigma_D$"
)
ax.set(xlim=(xmin, xmax))
ax.legend()

#%%
if make_plots:

    fig, (ax1, ax2) = plt.subplots(figsize=(10, 3.5), ncols=2)

    ymin, ymax = -0.0001, 0.25

    reload(utils)
    utils.plot_individual_damage_result(
        df_in=df,
        group_all_keys=group,
        df_known_damage=df_known_damage,
        df_damaged_reads=df_damaged_reads,
        method="Bayesian",
        splitby="species",
        keys=["homo"],
        xlim=(min_seed - 0.5, max_seed - 0.1),
        ylim=(ymin, ymax),
        # fig_title=f"Simulation, {sim_N_reads} reads",
        fig_title=f"",
        ax_titles=False,
        ax_in=ax1,
        # loc="upper left",
        bbox_to_anchor=None,
        ncols=1,
        markerscale=0.7,
    )

    reload(utils)
    utils.plot_combined_damage_result(
        df_in=df,
        group_agg_all_keys=group_agg,
        df_known_damage=df_known_damage,
        df_damaged_reads=None,
        method="Bayesian",
        splitby="species",
        keys=["homo"],
        fig_title=f"Simulation",
        # xlim=(0.7 * 10**2, 1.3 * 10**5),
        xlim=(0.7 * 10, 1.3 * 10**5),
        ylim=(ymin, ymax),
        ax_titles=False,
        delta=0.1,
        ax_in=ax2,
        loc="upper right",
        markerscale=0.7,
    )

    # ax1.annotate("A)", (0.02, 0.9), xycoords="axes fraction", fontsize=14)
    ax1.set_title(
        r"\textbf{A}) Homo, $\delta_\mathrm{ss}$ = "
        f"{sim_damage:.2f}, L = {sim_length}, {sim_N_reads} reads",
        pad=15,
    )
    ax2.set_title(
        r"\textbf{B}) Homo, $\delta_\mathrm{ss}$ = "
        f"{sim_damage:.2f}, L = {sim_length}",
        pad=15,
    )

    fig.tight_layout()

    fig.savefig("figures/ngsngs_overview.pdf", bbox_inches="tight")


#%%

group_zero_damage = df_zero_damage.query("sim_N_reads == 1000")

reload(utils)

g = utils.plot_zero_damage_group(
    group_zero_damage,
    method="Bayesian",
    title="",
    xlim=(0.1, 0.9),
    ylim=(0.0, 0.008),
)

g.savefig("figures/zero_damage_1000_reads.pdf")
# %%


#%%


def parse_pydamage_name(name):
    _, species, damage, N_reads, L, seed = name.split("-")
    return species, float(damage), int(N_reads), int(L), int(seed.split(".")[0])


def load_pydamage_results():

    if Path("pydamage/pydamage.parquet").exists():
        df = pd.read_parquet("pydamage/pydamage.parquet")

    else:

        paths = Path("pydamage") / "homo"

        dfs = []
        for path in tqdm(list(paths.glob("*.csv"))):
            df = pd.read_csv(path)

            sim_data = parse_pydamage_name(path.stem)
            for i, col in enumerate(utils.simulation_columns):
                df[col] = sim_data[i]

            dfs.append(df)

        df = (
            pd.concat(dfs)
            .drop(columns=["reference"])
            .sort_values(["sim_damage", "sim_N_reads", "sim_seed"])
            .reset_index(drop=True)
        )
        df["sim_damage_percent_approx"] = df["sim_damage"].map(utils.D_DAMAGE_APPROX)
        df["D_max"] = df["damage_model_pmax"]
        df["D_max_std"] = df["damage_model_pmax_stdev"]
        df["significance"] = df["D_max"] / df["D_max_std"]

        df.to_parquet("pydamage/pydamage.parquet")

    return df


df_pydamage = load_pydamage_results().query("sim_length == 60")


#%%

x = x

# %%


sim_species = "homo"
sim_length = 60

df_pydamage_100 = df_pydamage.query("sim_seed < 100").copy()
df_metaDMG_100 = df_all.query(
    f"sim_species == '{sim_species}' & sim_length == {sim_length} & sim_seed < 100"
).copy()


df_pydamage_100["method"] = "pydamage"
df_metaDMG_100["method"] = "metaDMG"

cols = utils.simulation_columns + [
    "sim_damage_percent_approx",
    "D_max",
    "D_max_std",
    "significance",
    "method",
]
df_combined = pd.concat([df_pydamage_100[cols], df_metaDMG_100[cols]])

#%%


d_xlim = {
    0.0: (0, 10),
    0.01: (0, 10),
    0.02: (0, 10),
    0.05: (0, 10),
    0.1: (0, 20),
    0.15: (0, 30),
    0.20: (0, 50),
    0.30: (0, 100),
}
d_ylim = {
    0.0: (0, 0.03),
    0.01: (0, 0.12),
    0.02: (0, 0.12),
    0.05: (0, 0.2),
    0.1: (0.0, 0.3),
    0.15: (0.0, 0.4),
    0.20: (0.0, 0.4),
    0.30: (0.0, 0.5),
}


def plot_pydamage_comparison(sim_damage, sim_N_reads, group):

    sim_damage_percent_approx = utils.D_DAMAGE_APPROX[sim_damage]

    xlim = d_xlim[sim_damage_percent_approx]
    ylim = d_ylim[sim_damage_percent_approx]

    known_damage = utils.get_known_damage(
        df_known_damage=df_known_damage,
        sim_damage=sim_damage,
        sim_species=sim_species,
        sim_length=sim_length,
    )

    d_markers = {"pydamage": "v", "metaDMG": "o"}

    fig, ax = plt.subplots()

    for method in ["pydamage", "metaDMG"]:
        data = group.query(f"method == '{method}'")

        ax.scatter(
            data[f"significance"],
            data[f"D_max"],
            s=10,
            marker=d_markers[method],
            label=f"{method}",
            clip_on=True,
        )

    ax.axhline(
        known_damage,
        color="k",
        linestyle="--",
        label=r"$D_\mathrm{known} = " f"{known_damage*100:.1f}" r"\%$",
    )

    ax.set(
        xlabel="Significance (MAP)",
        ylabel="Damage (MAP)",
        xlim=xlim,
        ylim=ylim,
    )

    title = f"{sim_N_reads} reads\n" f"Briggs damage = {sim_damage}\n"
    ax.set_title(title, pad=30, fontsize=12, loc="left")

    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    ax.spines.right.set_visible(True)
    ax.spines.top.set_visible(True)

    leg_kws = dict(
        markerscale=1.5,
        bbox_to_anchor=(1, 1.1),
        loc="upper right",
        ncols=3,
    )
    handles, labels = ax.get_legend_handles_labels()
    order = [1, 0, 2]
    ax.legend(
        [handles[idx] for idx in order],
        [labels[idx] for idx in order],
        **leg_kws,
    )

    return fig


#%%


filename = Path(f"figures/pydamage_comparison.pdf")
filename.parent.mkdir(parents=True, exist_ok=True)

with PdfPages(filename) as pdf:

    it = tqdm(
        df_combined.query("sim_damage > 0").groupby(["sim_damage", "sim_N_reads"])
    )

    for (sim_damage, sim_N_reads), group in it:

        fig = plot_pydamage_comparison(sim_damage, sim_N_reads, group)

        pdf.savefig(fig, bbox_inches="tight")
        plt.close()


#%%

df_pydamage_zero_damage = df_pydamage.query(
    "sim_species == 'homo' and sim_damage == 0 and sim_length == 60"
)

df_metaDMG_zero_damage = df_all.query(
    "sim_species == 'homo' and sim_damage == 0 and sim_length == 60"
)


#%%

for sim_N_reads in df_metaDMG_zero_damage.sim_N_reads.unique():
    break


group_pydamage_zero_damage = df_pydamage_zero_damage.query(
    f"sim_N_reads == {sim_N_reads}"
)

group_metaDMG_zero_damage = df_metaDMG_zero_damage.query(
    f"sim_N_reads == {sim_N_reads}"
)

reload(utils)

fig = utils.plot_zero_damage_group(
    group_metaDMG_zero_damage,
    method="MAP",
    title="metaDMG. \n"
    rf"sim_N_reads = {sim_N_reads}, \# = {len(group_metaDMG_zero_damage)}",
)

fig = utils.plot_zero_damage_group(
    group_pydamage_zero_damage,
    method="MAP",
    title="pydamage. \n"
    rf"sim_N_reads = {sim_N_reads}, \# = {len(group_pydamage_zero_damage)}",
)

# %%


def f_forward(xs, cuts):
    out = []

    for x in xs:
        y = x
        for i_cut in range(len(cuts) - 1):
            if cuts[i_cut] <= x < cuts[i_cut + 1]:
                y = i_cut + (x - cuts[i_cut]) / (cuts[i_cut + 1] - cuts[i_cut])
                break

        out.append(y)
    return np.array(out)


def f_reverse(xs, cuts):

    cuts_x_forward = f_forward(cuts[:-1], cuts)

    out = []
    for x in xs:
        y = x
        for i_cut in range(len(cuts_x_forward) - 1):
            if cuts_x_forward[i_cut] <= x < cuts_x_forward[i_cut + 1]:
                y = (x - i_cut) * (cuts[i_cut + 1] - cuts[i_cut]) + cuts[i_cut]
                break
        out.append(y)
    return out


cuts_significance = np.array([0, 1, 2, 3, 4, 5, 10, 100, 1000, np.inf])

functions_significance = [
    lambda xs: f_forward(xs, cuts_significance),
    lambda xs: f_reverse(xs, cuts_significance),
]


cuts_damage = np.array([0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.10, 0.20, 0.5, 1])

functions_damage = [
    lambda xs: f_forward(xs, cuts_damage),
    lambda xs: f_reverse(xs, cuts_damage),
]


#%%


def plot_pydamage_comparison_zero_damage(sim_N_reads, group_zero_damage):

    fig, axes = plt.subplots(ncols=2, sharey=False, figsize=(8, 3))
    ax1, ax2 = axes

    for method, ax in zip(["metaDMG", "pydamage"], axes):

        data = group_zero_damage.query(f"method == '{method}'")
        mask = (data[f"significance"] > 2) & (data[f"D_max"] > 0.01)

        ax.scatter(data[f"significance"][mask], data[f"D_max"][mask], s=10, color="C1")
        ax.scatter(
            data[f"significance"][~mask], data[f"D_max"][~mask], s=10, color="C2"
        )

        title = f"{method}: {sim_N_reads} reads"
        ax.set_title(title, pad=10, fontsize=12, loc="left")

        ax.set_xscale("function", functions=functions_significance)
        ax.set_xticks(cuts_significance[:-1])

        ax.set_yscale("function", functions=functions_damage)
        ax.set_yticks(cuts_damage[:-1])
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

        # ax.grid()

    xlim = (0, 1000)
    ylim = (0, 0.5)
    ax1.set(
        xlabel="Significance (MAP)",
        ylabel="Damage (MAP)",
        xlim=xlim,
        ylim=ylim,
    )
    ax2.set(
        xlabel="Significance (MAP)",
        xlim=xlim,
        ylim=ylim,
    )

    for ax in axes:
        kwargs = dict(
            color="C1",
            linestyle="--",
            alpha=0.3,
            linewidth=1,
        )
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.plot([2, xlim[1]], [0.01, 0.01], **kwargs)
        ax.plot([2, 2], [0.01, ylim[1]], **kwargs)
        ax.set(xlim=xlim)

        ax.fill_between(
            [2, xlim[1]],
            [0.01, 0.01],
            [ylim[1], ylim[1]],
            color="C1",
            alpha=0.1,
        )

        ax.fill_between(
            [0, 2, 2, xlim[1]],
            [ylim[1], ylim[1], 0.01, 0.01],
            color="C2",
            alpha=0.1,
        )

        d = 0.5  # proportion of vertical to horizontal extent of the slanted line
        kwargs = dict(
            markersize=5,
            linestyle="none",
            color="k",
            mec="k",
            mew=1,
            clip_on=False,
        )
        ax.plot(
            [0, 0],
            [0.060, 0.055],
            marker=[(-1, d), (1, -d)],
            **kwargs,
        )

        ax.plot(
            [5, 5.5],
            [0.0, 0.0],
            marker=[(-0.5, d), (0.5, -d)],
            **kwargs,
        )

    return fig


fig = plot_pydamage_comparison_zero_damage(sim_N_reads, group_zero_damage)


# %%

filename = Path(f"figures/pydamage_comparison_zero_damage.pdf")
filename.parent.mkdir(parents=True, exist_ok=True)

with PdfPages(filename) as pdf:

    it = tqdm(df_combined.query("sim_damage == 0").groupby("sim_N_reads"))

    for sim_N_reads, group_zero_damage in it:
        fig = plot_pydamage_comparison_zero_damage(sim_N_reads, group_zero_damage)

        pdf.savefig(fig, bbox_inches="tight")
        plt.close()

# %%
