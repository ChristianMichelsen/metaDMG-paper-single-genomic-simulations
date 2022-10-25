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
make_plots = True


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
df_true_damage = utils.get_df_true_damage()


#%%

x = x

#%%

reload(utils)
if make_plots:
    utils.plot_individual_damage_results(
        df=df,
        df_true_damage=df_true_damage,
        df_damaged_reads=df_damaged_reads,
    )


#%%

reload(utils)
df_aggregated = utils.get_df_aggregated(
    df_in=df,
    df_true_damage=df_true_damage,
    df_damaged_reads=df_damaged_reads,
)


# %%

# reload(utils)
if make_plots:
    utils.plot_combined_damage_results(
        df=df,
        df_aggregated=df_aggregated,
        df_true_damage=df_true_damage,
        df_damaged_reads=df_damaged_reads,
    )


#%%


df_aggregated_homo = df_aggregated.query("sim_species == 'homo'")

reload(utils)
if make_plots:
    fig = utils.plot_combined_MAEs(
        df_aggregated_homo=df_aggregated_homo,
        df_true_damage=df_true_damage,
        method="Bayesian",
        # ylim=(0, 2),
    )


# %%

reload(utils)
if make_plots:
    utils.plot_contour_lines(df)


#%%


# reload(utils)
if make_plots:

    fig, ax = plt.subplots(figsize=(3.5, 3))

    utils.plot_contour_lines_on_ax(
        df,
        ax=ax,
        sim_species="homo",
        method="Bayesian",
        title="",
        frac_legend_pos=(1, 0.60),
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

    utils.plot_individual_damage_results_lengths(
        df_homo_99=df_homo_99,
        df_true_damage=df_true_damage,
        df_damaged_reads=df_damaged_reads,
    )


#%%

reload(utils)

df_aggregated_lengths = utils.get_df_aggregated(
    df_in=df_homo_99,
    df_true_damage=df_true_damage,
    df_damaged_reads=df_damaged_reads,
)

if make_plots:

    utils.plot_combined_damage_results_lengths(
        df_homo_99=df_homo_99,
        df_aggregated_lengths=df_aggregated_lengths,
        df_true_damage=df_true_damage,
        df_damaged_reads=df_damaged_reads,
    )


#%%


df_contigs = df_all.query(f"sim_species in {all_species[-3:]}")

reload(utils)
if make_plots:

    utils.plot_individual_damage_results(
        df=df_contigs,
        df_true_damage=df_true_damage,
        df_damaged_reads=df_damaged_reads,
        suffix="_contigs",
    )

# %%

reload(utils)

df_aggregated_contigs = utils.get_df_aggregated(
    df_in=df_contigs,
    df_true_damage=df_true_damage,
    df_damaged_reads=df_damaged_reads,
)

# reload(utils)
if make_plots:
    utils.plot_combined_damage_results(
        df=df_contigs,
        df_aggregated=df_aggregated_contigs,
        df_true_damage=df_true_damage,
        df_damaged_reads=df_damaged_reads,
        suffix="_contigs",
    )


#%%

sim_species = "homo"
sim_damage = 0.065
sim_N_reads = 100
sim_length = 60
max_seed = 10

group = df.query(
    f"sim_species == '{sim_species}'"
    f" and sim_damage == {sim_damage}"
    f" and sim_length == {sim_length}"
    f" and sim_N_reads == {sim_N_reads}"
    f" and sim_seed < {max_seed}"
)


group_agg = df_aggregated.query(
    f"sim_species == '{sim_species}'"
    f" and sim_damage == {sim_damage}"
    f" and sim_length == {sim_length}"
    # f" and sim_N_reads == {sim_N_reads}"
    # f" and sim_seed < {max_seed}"
)


#%%


if make_plots:

    fig, (ax1, ax2) = plt.subplots(figsize=(10, 3.3), ncols=2)

    reload(utils)
    utils.plot_individual_damage_result(
        df_in=df,
        group_all_keys=group,
        df_true_damage=df_true_damage,
        df_damaged_reads=df_damaged_reads,
        method="Bayesian",
        splitby="species",
        keys=["homo"],
        figsize=(6, 4),
        xlim=(-0.5, max_seed - 0.1),
        ylim=(-0.0001, 0.09),
        # fig_title=f"Simulation, {sim_N_reads} reads",
        fig_title=f"",
        ax_titles=False,
        ax_in=ax1,
        loc="upper left",
    )

    reload(utils)
    utils.plot_combined_damage_result(
        df_in=df,
        group_agg_all_keys=group_agg,
        df_true_damage=df_true_damage,
        df_damaged_reads=None,
        method="Bayesian",
        splitby="species",
        keys=["homo"],
        figsize=(6, 4),
        fig_title=f"Simulation",
        # xlim=(0.7 * 10**1, 1.3 * 10**5),
        xlim=(0.7 * 10**2, 1.3 * 10**5),
        # ylim=(-0.0001, 0.18),
        ylim=(-0.0001, 0.09),
        ax_titles=False,
        delta=0.1,
        ax_in=ax2,
        loc="upper left",
    )

    # ax1.annotate("A)", (0.02, 0.9), xycoords="axes fraction", fontsize=14)
    ax1.set(
        title=r"\textbf{A}) Homo, $\delta_\mathrm{ss}$ = "
        f"{sim_damage:.3f}, L = {sim_length}, {sim_N_reads} reads"
    )
    ax2.set(
        title=r"\textbf{B}) Homo, $\delta_\mathrm{ss}$ = "
        f"{sim_damage:.3f}, L = {sim_length}"
    )

    fig.savefig("figures/ngsngs_overview.pdf", bbox_inches="tight")


#%%

group_zero_damage = df_zero_damage.query("sim_N_reads == 100")

reload(utils)
g = utils.plot_zero_damage_group(
    group_zero_damage,
    method="Bayesian",
    title="",
    xlim=(0.25, 0.8),
    ylim=(0.004, 0.0185),
)

g.savefig("figures/zero_damage_100_reads.pdf")
# %%
