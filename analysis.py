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

x = x

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

# %%


#%%

reload(utils)

df_0 = df_all.query("sim_species == 'homo' and sim_damage == 0 and sim_length == 60")

if make_plots:
    utils.plot_zero_damage_groups(df_0)


#%%

# sns.scatterplot(data=df_0, x="Bayesian_significance", y="Bayesian_D_max")


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


delta = 0.1

prefix = "Bayesian_"

fig, ax = plt.subplots()

ax.axhline(
    sim_damage_percent,
    color="k",
    linestyle="--",
    label=f"{sim_damage_percent:.0%}",
)

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

y, sy = utils.from_low_high_to_errors_corrected(
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
    color="C1",
    label=r"Median $\pm$ 68\% C.I.",
)

y2, sy2 = utils._from_low_high_to_errors(
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
    color="C1",
    # label="68% C.I.",
)


ax.set(
    title=f"Simulation",
    xlabel="Iteration",
    ylabel=r"Damage, $D$",
    xlim=(-0.9, max_seed - 0.1),
    ylim=(-0.0001, 0.1),
)

ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

handles, labels = ax.get_legend_handles_labels()
order = [1, 2, 0]
ax.legend(
    [handles[idx] for idx in order],
    [labels[idx] for idx in order],
    loc="upper right",
    markerscale=0.7,
    # frameon=True,
)

fig.savefig("fig-test.pdf")


# %%
