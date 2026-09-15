import xarray as xr
import numpy as np
import math
import os

from implicit_filter import IconFilter, transform_velocity_to_nodes

dataset_path = "/scratch/b/b382615/filters/implicit_filter_ICON_test.nc"
filter_cache = "icon_cash.npz"

scales = 2 * math.pi / np.logspace(1, 3, 30)  # Size of 10 to 1000 km

if __name__ == "__main__":
    rank = int(os.environ.get("SLURM_PROCID", 1))  # Task unique number
    num_tasks = int(os.environ.get("SLURM_NTASKS", 1))  # Total number of tasks

    # Make sure the filter uses the GPU. This has to happen before
    # load_from_file: JAX fixes its platform when the first array is created,
    # and loading the cache creates the filter's arrays, so a set_backend after
    # it would be inert.
    IconFilter().set_backend("gpu")

    icon_filter = IconFilter.load_from_file(filter_cache)

    ds = xr.open_dataset(dataset_path)
    time = ds.time.values
    tiemsteps = np.arange(len(time))  # Number of total timesteps to be computed

    # Select timesteps for this task
    tiemsteps = tiemsteps[(tiemsteps % num_tasks) == rank]

    for t in tiemsteps:  # Loop over snapshots for th
        ux = ds["u"].values[t, :]
        vy = ds["v"].values[t, :]
        uxn, vyn = transform_velocity_to_nodes(ux, vy, icon_filter)
        spectra = icon_filter.compute_spectra_velocity(1, scales, uxn, vyn)

        np.savez(f"spectra_icon_{time[t]}", spectra=spectra, scales=scales)
