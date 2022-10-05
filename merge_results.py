#%%

from importlib import reload
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import utils

#%%

datapath = Path("data") / "results"

# %%

for path in datapath.glob("*.parquet"):
    break
# %%

df = utils.load_results(use_columns_subset=False)


#%%

# reload(utils)
for p, dfg in tqdm(df.groupby(utils.sim_columns[:-1])):

    sim_species, sim_damage, sim_N_reads, sim_length = p

    name = f"{sim_species}-{sim_damage}-{sim_N_reads}-{sim_length}"
    path = Path("data") / f"results-{sim_species}" / f"{name}.parquet"

    dfg["tax_id"] = dfg["sample"]
    dfg["sample"] = name

    path.parent.mkdir(exist_ok=True, parents=True)
    dfg.to_parquet(path)

# %%
