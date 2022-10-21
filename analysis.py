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

x = x

#%%

reload(utils)

df_damaged_reads = utils.load_multiple_damaged_reads(all_species)

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
