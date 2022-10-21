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

d_damage_translate = utils.d_damage_translate


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

reload(utils)

df_all = utils.load_multiple_species(all_species)

#%%

df = df_all.query(
    f"sim_length == 60 and sim_seed < 100 and sim_species in {all_species[:-3]}"
)

#%%

reload(utils)

df_damaged_reads = utils.load_multiple_damaged_reads(all_species)


#%%

# x = x

#%%

reload(utils)
if make_plots:
    utils.plot_individual_damage_results(df, df_damaged_reads)


#%%

df_aggregated = utils.get_df_aggregated(df, df_damaged_reads)


# %%

reload(utils)
if make_plots:
    utils.plot_combined_damage_results(
        df,
        df_aggregated,
        df_damaged_reads,
    )


# %%

reload(utils)
if make_plots:
    utils.plot_contour_lines(df)

#%%

reload(utils)

df_0 = df_all.query("sim_species == 'homo' and sim_damage == 0 and sim_length == 60")

if make_plots:
    utils.plot_zero_damage_groups(df_0)


#%%

df_homo_99 = df_all.query("sim_species == 'homo' and sim_seed < 100")

reload(utils)
if make_plots:

    utils.plot_individual_damage_results_lengths(
        df_homo_99,
        df_damaged_reads,
    )


#%%

reload(utils)

df_aggregated_lengths = utils.get_df_aggregated(df_homo_99, df_damaged_reads)


if make_plots:

    utils.plot_combined_damage_results_lengths(
        df_homo_99,
        df_aggregated_lengths,
        df_damaged_reads,
    )


#%%


df_contigs = df_all.query(f"sim_species in {all_species[-3:]}")

reload(utils)
if make_plots:

    utils.plot_individual_damage_results(
        df_contigs,
        df_damaged_reads,
        suffix="_contigs",
    )

# %%


df_aggregated_contigs = utils.get_df_aggregated(df_contigs, df_damaged_reads)

reload(utils)
if make_plots:
    utils.plot_combined_damage_results(
        df_contigs,
        df_aggregated_contigs,
        df_damaged_reads,
        suffix="_contigs",
    )


#%%


sim_species = "homo"
sim_damage = 0.047
sim_damage_percent = d_damage_translate[sim_damage]
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

#%%


#%%

# group_agg = df_aggregated.query()


group_agg = df_aggregated.query(
    f"sim_species == '{sim_species}'"
    f" and sim_damage == {sim_damage}"
    f" and sim_length == {sim_length}"
    # f" and sim_N_reads == {sim_N_reads}"
    # f" and sim_seed < {max_seed}"
)


#%%


fig, (ax1, ax2) = plt.subplots(figsize=(14, 4), ncols=2)

reload(utils)
utils.plot_individual_damage_result(
    df_in=df,
    group_all_keys=group,
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
)


reload(utils)
utils.plot_combined_damage_result(
    df_in=df,
    group_agg_all_keys=group_agg,
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

fig.savefig("figures/simulation_overview.pdf")

# %%
