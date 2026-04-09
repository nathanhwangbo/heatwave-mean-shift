# analyzing the output of 0_era_meanshift.py

from _ssl import HOSTFLAG_ALWAYS_CHECK_SUBJECT
from changing_heat_extremes import flags
from changing_heat_extremes import plot_helpers as phelpers
# from changing_heat_extremes import analysis_helpers as ahelpers

import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import hvplot.xarray  # noqa: F401
import holoviews as hv
from pathlib import Path

hvplot.extension(phelpers.backend_hv)

fig_dir = Path("figures")
data_dir = Path("processed_data")


fig_kwargs = dict(
    xlabel="",
    ylabel="",
    fig_inches=(phelpers.width_default, phelpers.height_wide),
    xaxis=None,
    yaxis=None,
    **phelpers.global_kwargs,
)

layout_kwargs = dict(sublabel_format="", tight=True, tight_padding=(1, 17))


hw_obs = xr.open_dataset(
    data_dir
    / f"hw_metrics_{flags.ref_years[0]}_{flags.new_years[1]}_anom{flags.label}.nc"
).sel(
    percentile=flags.percentile_threshold,
    definition="-".join(map(str, flags.hw_def[0])),
)
hw_synth = xr.open_dataset(
    data_dir
    / f"hw_metrics_{flags.ref_years[0]}_{flags.new_years[1]}_synth_anom{flags.label}.nc"
).sel(
    percentile=flags.percentile_threshold,
    definition="-".join(map(str, flags.hw_def[0])),
)


##########################################################

# instead of looking at trends over time, look at simple change between the two periods
# mean(1986-2021) - mean(1950-1985)

###########################################################


def get_delta_fig(mean_diff_ds, cmap_hwf_hook, cmap_hwd_hook, cmap_sumheat_hook):
    """
    Creates figures looking at changes in difference in mean heatwave characteristics over two time period
    mean_diff_ds is an xarray dataset, the output of hdp.metric.compute_group_metrics, manipulated to represent the difference in metrics across two time periods
    label_source is a string (intended either "synthetic" or "observed") describing the source of the more recent time period used to calculate of mean_diff_ds
    label_summer is a string (intended either "jja" or "doy") describing how summer was defined in mean_diff_ds
    """
    hwf_delta = mean_diff_ds["t2m_x.t2m_x_threshold.HWF"]
    horizontal_cbar_hwf = phelpers.horizontal_cbar_hv(
        clabel=r"$\Delta$ Frequency (days)"
    )

    deltamap_hwf = hwf_delta.hvplot.quadmesh(
        projection=ccrs.PlateCarree(),
        coastline=True,
        # title=f"delta in heatwave frequency ({label_summer})\nmean({label_source} {flags.new_years[0]}:{flags.new_years[1]}) - mean(obs {flags.ref_years[0]}:{flags.ref_years[1]})",
        title="",
    ).opts(
        hv.opts.QuadMesh(
            colorbar=False,
            ylim=(-59, None),
            hooks=[cmap_hwf_hook, horizontal_cbar_hwf],
            **fig_kwargs,
        )
    )

    # delta in hwd
    hwd_delta = mean_diff_ds["t2m_x.t2m_x_threshold.HWD"]
    horizontal_cbar_hwd = phelpers.horizontal_cbar_hv(
        clabel=r"$\Delta$ Duration (days)"
    )
    deltamap_hwd = hwd_delta.hvplot.quadmesh(
        projection=ccrs.PlateCarree(),
        coastline=True,
        title="",
    ).opts(
        hv.opts.QuadMesh(
            colorbar=False,
            ylim=(-59, None),
            hooks=[cmap_hwd_hook, horizontal_cbar_hwd],
            **fig_kwargs,
        )
    )

    # delta in heatsum
    heatsum_delta = mean_diff_ds["t2m_x.t2m_x_threshold.sumHeat"]
    horizontal_cbar_heatsum = phelpers.horizontal_cbar_hv(
        clabel=r"$\Delta$ Cumulative Heat (°C-days)"
    )
    deltamap_heatsum = heatsum_delta.hvplot.quadmesh(
        projection=ccrs.PlateCarree(), coastline=True, title=""
    ).opts(
        hv.opts.QuadMesh(
            colorbar=False,
            ylim=(-59, None),
            hooks=[cmap_sumheat_hook, horizontal_cbar_heatsum],
            **fig_kwargs,
        )
    )

    # combine
    figlist_delta = [deltamap_hwf, deltamap_hwd, deltamap_heatsum]

    return figlist_delta


cmap_hwf_hook = phelpers.cbar_helper_hv(0, 15, cmap=phelpers.cmap_red)
cmap_hwd_hook = phelpers.cbar_helper_hv(-2, 6, cmap=phelpers.cmap_rdbu, cmap_center=0)
cmap_heatsum_hook = phelpers.cbar_helper_hv(0, 25, cmap=phelpers.cmap_red)

# ERA observed --------------------------------------
hw_old_obs = hw_obs.sel(time=slice(str(flags.ref_years[0]), str(flags.ref_years[1])))
hw_new_obs = hw_obs.sel(time=slice(str(flags.new_years[0]), str(flags.new_years[1])))
mean_diff_obs = hw_new_obs.mean(dim="time") - hw_old_obs.mean(dim="time")
figlist_delta_obs = get_delta_fig(
    mean_diff_obs, cmap_hwf_hook, cmap_hwd_hook, cmap_heatsum_hook
)

# make sure order matches get_delta_fig!
var_list = ["HWF", "HWD", "sumHeat"]  # , "MAX"]

# add in some text
fig_delta_obs = hv.Layout(
    [
        (
            figlist_delta_obs[i]
            * hv.Text(
                -180 + 220,
                -60 + 11,
                f"Global Mean={str(mean_diff_obs[f't2m_x.t2m_x_threshold.{var_list[i]}'].mean().values.round(2))}",
                fontsize=phelpers.label_size - 2,
            ).opts(ylim=(-59, 80), xlim=(-180, 180))
        )
        for i in range(len(var_list))
    ]
).opts(**layout_kwargs)

# hvplot.save(fig_delta_obs, f"fig_delta_obs_anom_{flags.label}_ref{flags.ref_years[0]}_{flags.ref_years[1]}.html")


# ERA synthetic second half -------------------------------------
hw_old_synth = hw_synth.sel(
    time=slice(str(flags.ref_years[0]), str(flags.ref_years[1]))
)
hw_new_synth = hw_synth.sel(
    time=slice(str(flags.new_years[0]), str(flags.new_years[1]))
)
mean_diff_synth = hw_new_synth.mean(dim="time") - hw_old_synth.mean(dim="time")
fig_delta_synth_init = get_delta_fig(
    mean_diff_synth, cmap_hwf_hook, cmap_hwd_hook, cmap_heatsum_hook
)


### also calculate the correlations for each, and add as labels ---
cor_obs_synth = xr.combine_by_coords(
    [
        xr.corr(
            mean_diff_obs[var],
            mean_diff_synth[var],
        )
        for var in mean_diff_obs.data_vars
    ]
)


fig_delta_synth = hv.Layout(
    [
        (
            fig_delta_synth_init[i]
            * hv.Text(
                -180 + 35,
                -60 + 11,
                f"r={str(cor_obs_synth[f't2m_x.t2m_x_threshold.{var_list[i]}'].values.round(2))}",
                fontsize=phelpers.label_size - 2,
            )
            * hv.Text(
                -180 + 220,
                -60 + 11,
                f"Global Mean={str(mean_diff_synth[f't2m_x.t2m_x_threshold.{var_list[i]}'].mean().values.round(2))}",
                fontsize=phelpers.label_size - 2,
            )
        ).opts(ylim=(-59, 80), xlim=(-180, 180))
        for i in range(len(var_list))
    ]
).opts(**layout_kwargs)
# hvplot.save(fig_delta_synth, f"fig_delta_synth_anom_{flags.label}_ref{flags.ref_years[0]}_{flags.ref_years[1]}.html")

# difference between observed and synthetic -------------------

cmap_hwf_diff_hook = phelpers.cbar_helper_hv(-8, 8, cmap=phelpers.cmap_rdbu)
cmap_hwd_diff_hook = phelpers.cbar_helper_hv(-4, 4, cmap=phelpers.cmap_rdbu)
cmap_heatsum_diff_hook = phelpers.cbar_helper_hv(-12, 12, cmap=phelpers.cmap_rdbu)


obs_minus_synth = mean_diff_obs - mean_diff_synth
fig_obs_minus_synth_init = get_delta_fig(
    obs_minus_synth, cmap_hwf_diff_hook, cmap_hwd_diff_hook, cmap_heatsum_diff_hook
)

# update the labels of each of these plots
fig_obs_minus_synth_init[0].opts(
    hv.opts.QuadMesh(
        hooks=[
            cmap_hwf_diff_hook,
            phelpers.horizontal_cbar_hv(clabel="Obs - Synth (days)"),
        ]
    )
)
fig_obs_minus_synth_init[1].opts(
    hv.opts.QuadMesh(
        hooks=[
            cmap_hwd_diff_hook,
            phelpers.horizontal_cbar_hv(clabel="Obs - Synth (days)"),
        ]
    )
)

fig_obs_minus_synth_init[2].opts(
    hv.opts.QuadMesh(
        hooks=[
            cmap_heatsum_diff_hook,
            phelpers.horizontal_cbar_hv(clabel=r"Obs - Synth (°C-days)"),
        ]
    )
)


# add in mean absolute error over the map
mean_abs_diff = abs(obs_minus_synth).mean()

fig_obs_minus_synth = hv.Layout(
    [
        (
            fig_obs_minus_synth_init[i]
            * hv.Text(
                -180 + 52,
                -60 + 11,
                f"MAE={str(mean_abs_diff[f't2m_x.t2m_x_threshold.{var_list[i]}'].values.round(2))}",
                fontsize=phelpers.label_size - 2,
            )
        ).opts(ylim=(-59, 80), xlim=(-180, 180))
        for i in range(len(var_list))
    ]
).opts(**layout_kwargs)


# stitch all together into a single figure -------------------------
fig1 = fig_delta_obs[0] + fig_delta_synth[0] + fig_obs_minus_synth[0]
for i in np.arange(1, len(fig_delta_obs)).tolist():
    fig1 += fig_delta_obs[i] + fig_delta_synth[i] + fig_obs_minus_synth[i]
fig1 = fig1.cols(3).opts(**layout_kwargs)

# iterate over the subplots and add the label to the title. --
# weird ordering bc I want to go vertical instead of horizontal
# letter_ordering = ["a", "d", "g", "b", "e", "h", "c", "f", "i"]

# fig1_updated = phelpers.add_subplot_labels(fig1, labels=letter_ordering)

# fig1_updated = (fig1.cols(3)).opts(
#     shared_axes=False,
#     # sublabel_format="({alpha})",
#     tight=True,
#     tight_padding=(1, 20),
#     # sublabel_position=(-0.05, 0.1),
#     # sublabel_size=phelpers.label_size,
# )
# # hvplot.save(fig1_updated, fig_dir / f"fig_meanshift_{flags.label}_ref{flags.ref_years[0]}_{flags.ref_years[1]}.png")

# add subplot labels in matplotlib code
fig = hv.render(fig1, backend="matplotlib")

# collect only map axes (exclude colorbars)
map_axes = [ax for ax in fig.axes if ax.__class__.__name__ == "GeoAxes"]

# weird ordering bc I want to go vertical instead of horizontal
# labels = ["(a)", "(d)", "(g)", "(b)", "(e)", "(h)", "(c)", "(f)", "(i)"]
labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)", "(i)"]

for ax, lab in zip(map_axes, labels, strict=False):
    ax.text(
        0.015,
        0.18,
        lab,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=phelpers.label_size,
        fontweight="normal",
    )

# ! uncomment to save figure. this is the main output of this script!
# fig.savefig(
#     fig_dir
#     / f"fig_meanshift_{flags.label}_ref{flags.ref_years[0]}_{flags.ref_years[1]}.png",
#     dpi=200,
#     bbox_inches="tight",
# )

##########################################3
# supplemental analyses mentioned in the paper
###########################################3

# correlation between hwf and sumheat changes in the obs
xr.corr(
    mean_diff_obs["t2m_x.t2m_x_threshold.HWF"],
    mean_diff_obs["t2m_x.t2m_x_threshold.sumHeat"],
)


xr.corr(
    mean_diff_obs["t2m_x.t2m_x_threshold.HWF"],
    mean_diff_obs["t2m_x.t2m_x_threshold.HWD"],
)

# repeated for synthetic
xr.corr(
    mean_diff_synth["t2m_x.t2m_x_threshold.HWF"],
    mean_diff_synth["t2m_x.t2m_x_threshold.HWD"],
)

xr.corr(
    mean_diff_synth["t2m_x.t2m_x_threshold.HWF"],
    mean_diff_synth["t2m_x.t2m_x_threshold.sumHeat"],
)

#


#################################33
# mimicking fig 3 of martinez-villalobos et al 2025
# which look at the relationship between mean shift and heat wave duration

# specifically:
# x-axis is mean shift / standard deviation
# y-axis is 99th quantile of heatwave duration (over all heatwaves).
# - but they define heatwaves differently than us (i.e. no 3 day minimum). So maybe we should shrink the quantile
##################################

## obs -----------------------
import pandas as pd
import hvplot.pandas  # ensure pandas backend is available for scatter

# y-axis: 90th quantile of heatwave duration (over all heatwaves) -> ratio (new / old)
hwd_q90_old = hw_old_obs["t2m_x.t2m_x_threshold.HWD"].quantile(0.90, dim="time")
hwd_q90_new = hw_new_obs["t2m_x.t2m_x_threshold.HWD"].quantile(0.90, dim="time")
hwd_q90_ratio = hwd_q90_new / hwd_q90_old

# (hwd_q90_new - hwd_q90_old).hvplot(clim = (-7, 7), cmap = phelpers.cmap_rdbu, colorbar=True)


# x-axis: temperature mean shift / standard deviation
combined_ds = xr.open_dataset(
    data_dir / f"moments_ds_{flags.label}.nc"
)  # grab mean shift and variance
normalized_mean_shift = combined_ds["t2m_x_mean_diff"] / np.sqrt(
    combined_ds["t2m_x_var"]
)


# Merge the two DataArrays into a single Dataset for plotting
scatter_ds = xr.Dataset(
    {"normalized_mean_shift": normalized_mean_shift, "hwd_q90_ratio": hwd_q90_ratio}
)

# Flatten and drop NaNs to improve plotting performance
scatter_df = scatter_ds.to_dataframe().dropna().reset_index()

# Create 10 equal-width bins
bins = pd.cut(scatter_df["normalized_mean_shift"], bins=10)

# Use the midpoint of each bin as the x-axis value to keep it numeric and sorted
scatter_df["x_bin_mid"] = bins.apply(lambda x: x.mid).astype(float)

# some messing around to try and keep a numeric x axis
# unique_mids = sorted(scatter_df["x_bin_mid"].unique())
# mid_to_label = {mid: f"{i:02d}: {mid:.2f}" for i, mid in enumerate(unique_mids)}
# scatter_df["x_bin_sorted_label"] = scatter_df["x_bin_mid"].map(mid_to_label)
# scatter_df.hvplot.box(
#     y="hwd_q90_ratio",
#     by="x_bin_sorted_label",
#     height=450,
#     width=900,
#     rot=45,
#     xlabel="Mean Shift / Standard Deviation (Sorted Bins)",
#     ylabel="HWD 90th Quantile Ratio (New/Old)",
#     title="Distribution of Heatwave Duration Response by Mean Shift Bins",
#     **phelpers.global_kwargs,
# ).opts(xticks=unique_mids)

import matplotlib.pyplot as plt

# group the ratio by the numeric bin midpoint
grouped_data = [
    group["hwd_q90_ratio"].values for name, group in scatter_df.groupby("x_bin_mid")
]
positions = sorted(scatter_df["x_bin_mid"].unique().round(2))
fig, ax = plt.subplots(figsize=(10, 6))

# 3. Create boxplots
ax.boxplot(
    grouped_data,
    positions=positions,
    widths=0.15,  # Controls width in numeric x-axis units
    patch_artist=True,
    medianprops={"color": "black"},
    boxprops={"facecolor": "lightgray", "alpha": 0.7},
)
ax.set_xlabel("Mean Shift / Standard Deviation (Bin Midpoint)")
ax.set_ylabel("HWD 90th Quantile Ratio (New/Old)")
ax.set_title("")
ax.grid(axis="y", linestyle="--", alpha=0.7)

# fig.savefig(
#     fig_dir / "supplemental" / "boxplot_numeric_x.png", dpi=200, bbox_inches="tight"
# )
