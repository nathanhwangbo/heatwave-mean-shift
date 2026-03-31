"""
generate heatmap figures, with deciles bins of
- climatology moments on the x axis
- change in heatwave metrics on the y axis

inputs:
- moments_ds.nc from 2a_get_moments

outputs:
- fig_qbins.png
"""

from changing_heat_extremes import flags
from changing_heat_extremes import plot_helpers as phelpers
import xarray as xr
import polars as pl
import polars.selectors as cs
import string
import numpy as np
import holoviews as hv
import hvplot.xarray  # noqa: F401
import hvplot.polars  # noqa: F401
from pathlib import Path

hvplot.extension(phelpers.backend_hv)

data_dir = Path("processed_data")

fig_kwargs = dict(
    fig_inches=(phelpers.width_default, phelpers.height_wide),
    **phelpers.global_kwargs,
)

layout_kwargs = dict(tight=True, tight_padding=0)

###########################################3
# read in data from 2a_get_moments
###########################################

combined_ds = xr.open_dataset(data_dir / f"moments_ds_{flags.label}.nc")
combined_df = pl.from_pandas(combined_ds.to_dataframe(), include_index=True).drop_nulls()


# combined_ds.plot.scatter(
#     x="t2m_x_mean_diff", y="t2m_x_skew", hue="t2m_x.t2m_x_threshold.HWF", s=10
# )


##########################################
# if the above has too many points
# let's combine points into decile bins
#########################################

# exclude 0 beacuse pl.qcut is left-closed by default
decile_list = np.linspace(0.1, 1, 10)
vars_to_bin = ["t2m_x_mean_diff", "t2m_x_var", "t2m_x_skew"]

q_df = combined_df.clone()
for var in vars_to_bin:
    qbins = (
        combined_df[var]
        .qcut(quantiles=decile_list, include_breaks=True)
        .to_frame()
        .unnest(var)
        # .select("breakpoint")
        .rename({"category": f"{var}_cat", "breakpoint": f"{var}_q"})
    )
    q_df = q_df.hstack(qbins)


### variance ----------------------------------------

hw_cols = [name for name in q_df.columns if "threshold" in name]
qvar_df = (
    q_df.group_by(["t2m_x_mean_diff_q", "t2m_x_var_q"])
    .agg(pl.col(hw_cols).median(), pl.len())
    .with_columns(cs.numeric().round(1))
)

# make the ordering match what hv.quadmesh is expecting
qvar_ds = (
    (qvar_df.to_pandas().set_index(["t2m_x_mean_diff_q", "t2m_x_var_q"]).to_xarray())
    .sortby(["t2m_x_mean_diff_q", "t2m_x_var_q"])
    .transpose("t2m_x_var_q", "t2m_x_mean_diff_q")
)

# get the boundaries (i.e. including the min)
mean_diff_qs = np.concatenate(([q_df["t2m_x_mean_diff"].min()], qvar_ds["t2m_x_mean_diff_q"].values)).round(1)
var_qs = np.concatenate(([q_df["t2m_x_var"].min()], qvar_ds["t2m_x_var_q"].values)).round(1)

# plot ticks to cut off the big min and max, and linearly between
tx_lim = (
    mean_diff_qs[1] - 0.3,
    mean_diff_qs[-2] + 0.3,
)  # cut off the huge first and last bin
tx_xticks = np.linspace(tx_lim[0], tx_lim[1], 5)

# var_lim = (var_qs[1] - 0.3, var_qs[-2] + 0.3)
var_yticks = np.linspace(var_qs[0], var_qs[-1], 5)


# # mark which boxes have fewer than 100 gridcells
# small_n_var = qvar_df.filter(pl.col("len") < 100).select(
#     ["t2m_x_mean_diff_q", "t2m_x_var_q"]
# )

# # map the box edges to the center
# mean_diff_midpoints = mean_diff_qs[:-1] + np.diff(mean_diff_qs) / 2
# var_midpoints = var_qs[:-1] + np.diff(var_qs) / 2
# x_coords_sorted = np.sort(qvar_ds["t2m_x_mean_diff_q"].values)
# y_coords_sorted = np.sort(qvar_ds["t2m_x_var_q"].values)
# x_map = dict(zip(x_coords_sorted, mean_diff_midpoints))
# y_map = dict(zip(y_coords_sorted, var_midpoints))
# small_n_var = small_n_var.with_columns([
#     pl.col("t2m_x_mean_diff_q").replace(x_map).alias("x_center"),
#     pl.col("t2m_x_var_q").replace(y_map).alias("y_center")
# ])

# markers_var = hv.Points(small_n_var, kdims=["x_center", "y_center"]).opts(
#         color="black",
#         marker="x",
#         s=10,
#     )

# make the plot
# hwf_q95 =qvar_ds["t2m_x.t2m_x_threshold.HWF"].quantile(0.95).values
cbar_hwf = phelpers.cbar_helper_hv(1, 20, num_bins=11, cmap="YlOrRd")
fig_var_hwf = hv.QuadMesh((mean_diff_qs, var_qs, qvar_ds["t2m_x.t2m_x_threshold.HWF"])).opts(
    edgecolors="white",
    # xticks=tx_xticks,
    xticks=0,
    yticks=var_yticks,
    colorbar=True,
    xlim=tx_lim,  # cut off the huge first and last bin (-0.8 -> 3.8)
    # title="(a) Change in HWF",
    xlabel="",
    ylabel="Variance (°C²)",
    clabel=r"$\Delta$ HWF (days)",
    hooks=[cbar_hwf, phelpers.ticks_bound_hv("x")],
    cbar_extend="both",
    **fig_kwargs,
)

# duration
hwd_q95 = qvar_ds["t2m_x.t2m_x_threshold.HWD"].quantile(0.95).values
cbar_hwd = phelpers.cbar_helper_hv(0, 7, cmap="YlOrRd")
fig_var_hwd = hv.QuadMesh((mean_diff_qs, var_qs, qvar_ds["t2m_x.t2m_x_threshold.HWD"])).opts(
    edgecolors="white",
    # xticks=tx_xticks,
    xticks=0,
    yticks=0,
    colorbar=True,
    xlim=tx_lim,  # cut off the huge first and last bin (-0.8 -> 3.8)
    # title="(b) $\Delta$ HWD",
    xlabel="",
    ylabel="",
    clabel=r"$\Delta$ HWD (days)",
    hooks=[cbar_hwd, phelpers.ticks_bound_hv("x")],
    cbar_extend="both",
    **fig_kwargs,
)

# sumheat
hwf_q95 = qvar_ds["t2m_x.t2m_x_threshold.sumHeat"].quantile(0.95).values
cbar_sumheat = phelpers.cbar_helper_hv(1, 35, cmap="YlOrRd")
fig_var_sumheat = hv.QuadMesh((mean_diff_qs, var_qs, qvar_ds["t2m_x.t2m_x_threshold.sumHeat"])).opts(
    edgecolors="white",
    # xticks=tx_xticks,
    xticks=0,
    yticks=0,
    colorbar=True,
    xlim=tx_lim,  # cut off the huge first and last bin (-0.8 -> 3.8)
    # title=r"(b) $\Delta$ sumHeat",
    xlabel="",
    ylabel="",
    clabel=r"$\Delta$ sumHeat (°C-days)",
    hooks=[cbar_sumheat, phelpers.ticks_bound_hv("x")],
    cbar_extend="both",
    **fig_kwargs,
)


figlist_var_qbins = [fig_var_hwf, fig_var_hwd, fig_var_sumheat]
fig_layout_var_qbins = hv.Layout(figlist_var_qbins).cols(3).opts(**layout_kwargs)
# hvplot.save(fig_layout_var_qbins, "fig_qbins_var.svg")

### skewness ----------------------------------------
qskew_df = (
    q_df.group_by(["t2m_x_mean_diff_q", "t2m_x_skew_q"])
    .agg(pl.col(hw_cols).median(), pl.len())
    .with_columns(cs.numeric().round(2))
)

# make the ordering match what hv.quadmesh is expecting
qskew_ds = (
    (qskew_df.to_pandas().set_index(["t2m_x_mean_diff_q", "t2m_x_skew_q"]).to_xarray())
    .sortby(["t2m_x_mean_diff_q", "t2m_x_skew_q"])
    .transpose("t2m_x_skew_q", "t2m_x_mean_diff_q")
)

# get the boundaries (i.e. including the min)
mean_diff_qs = np.concatenate(([q_df["t2m_x_mean_diff"].min()], qskew_ds["t2m_x_mean_diff_q"].values)).round(1)
skew_qs = np.concatenate(([q_df["t2m_x_skew"].min()], qskew_ds["t2m_x_skew_q"].values)).round(2)

# plot ticks excluding the super wide first and last bin, and linearly between
skew_lim = (skew_qs[1] - 0.3, skew_qs[-2] + 0.3)
skew_yticks = np.linspace(skew_lim[0], skew_lim[1], 5).round(2)


fig_skew_hwf = hv.QuadMesh((mean_diff_qs, skew_qs, qskew_ds["t2m_x.t2m_x_threshold.HWF"])).opts(
    edgecolors="white",
    xticks=tx_xticks,
    yticks=skew_yticks,
    colorbar=True,
    xlim=tx_lim,  # cut off the huge first and last bin (-0.8 -> 3.8)
    ylim=skew_lim,  # cut off the huge first and last bin (-1.57 -> 1.35)
    # title="(d) $\Delta$ HWF",
    xlabel=r"$\Delta$ Tx (°C)",
    ylabel="Skewnes",
    clabel=r"$\Delta$ HWF (days)",
    hooks=[cbar_hwf, phelpers.ticks_bound_hv("both")],
    cbar_extend="both",
    **fig_kwargs,
)

# duration
fig_skew_hwd = hv.QuadMesh((mean_diff_qs, skew_qs, qskew_ds["t2m_x.t2m_x_threshold.HWD"])).opts(
    edgecolors="white",
    xticks=tx_xticks,
    yticks=0,
    colorbar=True,
    xlim=tx_lim,  # cut off the huge first and last bin (-0.8 -> 3.8)
    ylim=skew_lim,  # cut off the huge first and last bin (-1.57 -> 1.35)
    # title="(e) $\Delta$ HWD",
    xlabel=r"$\Delta$ Tx (°C)",
    ylabel="",
    clabel=r"$\Delta$ HWD (days)",
    hooks=[cbar_hwd, phelpers.ticks_bound_hv("x")],
    cbar_extend="both",
    **fig_kwargs,
)

# sumheat
fig_skew_sumheat = hv.QuadMesh((mean_diff_qs, skew_qs, qskew_ds["t2m_x.t2m_x_threshold.sumHeat"])).opts(
    edgecolors="white",
    xticks=tx_xticks,
    yticks=0,
    colorbar=True,
    xlim=tx_lim,  # cut off the huge first and last bin (-0.8 -> 3.8)
    ylim=skew_lim,  # cut off the huge first and last bin (-1.57 -> 1.35)
    xlabel=r"$\Delta$ Tx (°C)",
    ylabel="",
    clabel=r"$\Delta$ sumHeat (°C-days)",
    hooks=[cbar_sumheat, phelpers.ticks_bound_hv("x")],
    cbar_extend="both",
    **fig_kwargs,
)


figlist_skew_qbins = [fig_skew_hwf, fig_skew_hwd, fig_skew_sumheat]
fig_layout_skew_qbins = hv.Layout(figlist_skew_qbins).cols(3).opts(**layout_kwargs)

##########################3
# fig_qbins ----
###########################3

fig_qbins_final = (
    hv.Layout(figlist_var_qbins + figlist_skew_qbins).cols(3).opts(sublabel_format="", **layout_kwargs)
    # .opts(
    #     sublabel_format="({alpha})", sublabel_position=(-0.075, 0.75), sublabel_size=phelpers.tick_size, **layout_kwargs
    # )
)
# hvplot.save(fig_qbins_final, phelpers.fig_dir / f"fig_qbins_{flags.label}.png")

# Render to matplotlib, then add manual subplot labels
fig = hv.render(fig_qbins_final, backend="matplotlib")

num_panels = len(fig_qbins_final)

# vibe coded warning
# separate the colorbar axes from the main axes (by size, width*height)
main_axes = sorted(
    fig.axes,
    key=lambda ax: ax.get_position().width * ax.get_position().height,
    reverse=True,
)[0:num_panels]

# make sure the axes are in the right order
# Order: top-left -> top-right, then bottom-left -> bottom-right
main_axes = sorted(
    main_axes,
    key=lambda ax: (-ax.get_position().y0, ax.get_position().x0),
)

labels = [f"({i})" for i in string.ascii_lowercase[0:num_panels]]
for ax, lab in zip(main_axes, labels, strict=False):
    ax.text(
        0.02,
        0.98,
        lab,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=phelpers.label_size,
        fontweight="normal",
    )

# ! save to file
# fig.savefig(phelpers.fig_dir / f"fig_qbins_{flags.label}.png", dpi=200, bbox_inches="tight")


###################################
# supplemental analyses used in the paper
####################################

# Figure of gridcell counts -----------------
cbar_count = phelpers.cbar_helper_hv(0, 200, cmap="YlOrRd")

fig_var_count = hv.QuadMesh((mean_diff_qs, var_qs, qvar_ds["len"])).opts(
    edgecolors="white",
    xticks=mean_diff_qs[1::2],
    yticks=var_qs[::2],
    colorbar=False,
    xlim=(
        mean_diff_qs[1] - 0.3,
        mean_diff_qs[-2] + 0.3,
    ),  # cut off the huge first and last bin (-0.8 -> 3.8)
    title="Variance",
    xlabel="Change in Tx (°C)",
    ylabel="Variance (°C²)",
    hooks=[cbar_count],
    cbar_extend="both",
    **fig_kwargs,
)

fig_skew_count = hv.QuadMesh((mean_diff_qs, skew_qs, qskew_ds["len"])).opts(
    edgecolors="white",
    xticks=mean_diff_qs[1::2],
    yticks=skew_qs[1::2],
    colorbar=True,
    xlim=(
        mean_diff_qs[1] - 0.3,
        mean_diff_qs[-2] + 0.3,
    ),  # cut off the huge first and last bin (-0.8 -> 3.8)
    ylim=(
        skew_qs[1] - 0.3,
        skew_qs[-2] + 0.3,
    ),  # cut off the huge first and last bin (-1.57 -> 1.35)
    title="Skewness",
    xlabel="Change in Tx (°C)",
    ylabel="Climatological Skew",
    clabel="Number of Gridcells",
    hooks=[cbar_count],
    cbar_extend="both",
    **fig_kwargs,
)

fig_counts = (fig_var_count + fig_skew_count).opts(**layout_kwargs)

# hvplot.save(fig_counts, phelpers.fig_dir / "supplemental" / "fig_qbin_counts.png")


#################################
# misc results cited in the paper
#################################

# pull out the ~1.1-1.2c bin
bin_eg = q_df["t2m_x_mean_diff_cat"].unique()[3]
bin_eg_df = q_df.filter(pl.col("t2m_x_mean_diff_cat") == bin_eg)
var_bin_eg_df = bin_eg_df.group_by("t2m_x_var_q").agg(pl.col(hw_cols).median()).sort("t2m_x_var_q")
skew_bin_eg_df = bin_eg_df.group_by("t2m_x_skew_q").agg(pl.col(hw_cols).median()).sort("t2m_x_skew_q")

#################################
## comparing the "gradients" to get a sense of movement across mean shift / variance
#################################
metrics_to_plot = {
    "t2m_x.t2m_x_threshold.HWF": "HWF",
    "t2m_x.t2m_x_threshold.HWD": "HWD",
    "t2m_x.t2m_x_threshold.sumHeat": "sumHeat",
}

figlist_grad_ratio = []
da_list_grad_ratio = []

# Create the hook, then tap directly into its stored colormap to set the bad values
cbar_ratio = phelpers.cbar_helper_hv(-1, 1, cmap="PuOr", cmap_center=0)
cbar_ratio.keywords["cbar_kwargs"]["cmap"].set_bad("white")

for i, (var_name, short_name) in enumerate(metrics_to_plot.items()):
    # Calculate finite differences per decile
    dz_dx_decile = qvar_ds[var_name].diff("t2m_x_mean_diff_q")
    dz_dy_decile = qvar_ds[var_name].diff("t2m_x_var_q")

    # Align to a 9x9 grid to center the differences
    dz_dx_9x9 = dz_dx_decile.rolling(t2m_x_var_q=2).mean().dropna("t2m_x_var_q")
    dz_dy_9x9 = dz_dy_decile.rolling(t2m_x_mean_diff_q=2).mean().dropna("t2m_x_mean_diff_q")

    # Ratio of the absolute per-decile gradients, converted to log10
    raw_ratio = np.abs(dz_dx_9x9 / dz_dy_9x9)
    grad_ratio = np.log10(raw_ratio)
    grad_ratio = grad_ratio.where(np.isfinite(grad_ratio))
    da_list_grad_ratio.append(grad_ratio)

    # Reindex back to the 10x10 grid of the original dataset
    grad_ratio_10x10 = grad_ratio.reindex_like(qvar_ds)
    grad_ratio_10x10.name = f"grad_ratio_{short_name}"

    is_first = i == 0

    # Slice off the first and last x-ticks for the 2nd and 3rd panels
    current_xticks = tx_xticks if is_first else tx_xticks[1:-1]

    # Only apply the bounding formatter hook to the first panel
    current_hooks = [cbar_ratio, phelpers.ticks_bound_hv("both")] if is_first else [cbar_ratio]

    fig_grad = hv.QuadMesh((mean_diff_qs, var_qs, grad_ratio_10x10)).opts(
        colorbar=False,
        edgecolors="white",
        # xlabel=r"$\Delta$ Tx (°C)",
        xlabel="",
        ylabel="Variance (°C²)" if is_first else "",
        xlim=tx_lim,
        # xticks=current_xticks,
        xticks=0,
        yticks=var_yticks if is_first else 0,
        hooks=current_hooks,
        **fig_kwargs,
    )
    figlist_grad_ratio.append(fig_grad)

fig_layout_grads = hv.Layout(figlist_grad_ratio).cols(3).opts(sublabel_format="", **layout_kwargs)

ds_grad_ratio = xr.merge(da_list_grad_ratio)
# proportion where gradient across mean deciles is larger than across variance deciles
np.mean(ds_grad_ratio > 0).values

#################################
## repeat gradient calculation for skewness instead of variance
#################################

figlist_grad_ratio_skew = []
da_list_grad_ratio_skew = []

# Create the hook, then tap directly into its stored colormap to set the bad values
cbar_ratio_skew = phelpers.cbar_helper_hv(-1, 1, cmap="PuOr", cmap_center=0)
cbar_ratio_skew.keywords["cbar_kwargs"]["cmap"].set_bad("white")

for i, (var_name, short_name) in enumerate(metrics_to_plot.items()):
    # Calculate finite differences per decile using qskew_ds
    dz_dx_decile = qskew_ds[var_name].diff("t2m_x_mean_diff_q")
    dz_dy_decile = qskew_ds[var_name].diff("t2m_x_skew_q")

    # align to a 9x9 grid to center the differences
    dz_dx_9x9 = dz_dx_decile.rolling(t2m_x_skew_q=2).mean().dropna("t2m_x_skew_q")
    dz_dy_9x9 = dz_dy_decile.rolling(t2m_x_mean_diff_q=2).mean().dropna("t2m_x_mean_diff_q")

    # ratio of the absolute per-decile gradients
    # >1 means metric changes more across mean decile, compared to var
    raw_ratio = np.abs(dz_dx_9x9 / dz_dy_9x9)
    # covert to log10, so that >0 means metric changes more across mean decile
    # and =1 means 10x difference
    grad_ratio = np.log10(raw_ratio)
    grad_ratio = grad_ratio.where(np.isfinite(grad_ratio))
    da_list_grad_ratio_skew.append(grad_ratio)

    # Reindex back to the 10x10 grid of the original dataset
    grad_ratio_10x10 = grad_ratio.reindex_like(qskew_ds)
    grad_ratio_10x10.name = f"grad_ratio_{short_name}_skew"

    is_first = i == 0

    # Slice off the first and last x-ticks for the 2nd and 3rd panels
    current_xticks = tx_xticks if is_first else tx_xticks[1:-1]

    # Only apply the bounding formatter hook to the first panel
    current_hooks = [cbar_ratio_skew, phelpers.ticks_bound_hv("both")] if is_first else [cbar_ratio_skew]

    fig_grad = hv.QuadMesh((mean_diff_qs, skew_qs, grad_ratio_10x10)).opts(
        colorbar=False,
        edgecolors="white",
        xlabel=r"$\Delta$ Tx (°C)",
        ylabel="Climatological Skew" if is_first else "",
        xlim=tx_lim,
        ylim=skew_lim,
        xticks=current_xticks,
        yticks=skew_yticks if is_first else 0,
        hooks=current_hooks,
        **fig_kwargs,
    )
    figlist_grad_ratio_skew.append(fig_grad)

fig_layout_grads_skew = hv.Layout(figlist_grad_ratio_skew).cols(3).opts(sublabel_format="", **layout_kwargs)
ds_grad_ratio_skew = xr.merge(da_list_grad_ratio_skew)

# proportion where gradient across mean deciles is larger than across skewness deciles
np.mean(ds_grad_ratio_skew > 0).values

#################################
# Combine Variance and Skewness into a single figure
#################################

# Combine the lists of HoloViews objects (3 variance + 3 skewness = 6 panels)
figlist_combined = figlist_grad_ratio + figlist_grad_ratio_skew

# Layout in 3 columns (which creates 2 rows)
fig_layout_combined = hv.Layout(figlist_combined).cols(3).opts(sublabel_format="", **layout_kwargs)

##########################
# Rendering & Combining  #
##########################

# 1. Render to matplotlib
fig_combined_mp = hv.render(fig_layout_combined, backend="matplotlib")
num_panels = len(figlist_combined)

# 2. Extract main axes
main_axes_combined = sorted(
    fig_combined_mp.axes,
    key=lambda ax: ax.get_position().width * ax.get_position().height,
    reverse=True,
)[0:num_panels]

# Order them strictly top-to-bottom, then left-to-right
# Using round() to avoid sorting errors if plotting jitter makes y-positions off by micro-fractions
main_axes_combined = sorted(
    main_axes_combined,
    key=lambda ax: (-round(ax.get_position().y0, 3), round(ax.get_position().x0, 3)),
)

# 3. Add manual subplot labels (a) through (f)
labels = [f"({i})" for i in string.ascii_lowercase[0:num_panels]]
for ax, lab in zip(main_axes_combined, labels, strict=False):
    ax.text(
        0.02,
        0.98,
        lab,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=phelpers.label_size,
        fontweight="normal",
    )

# 4. Attach a single unified colorbar bounding all main axes
mappable_combined = main_axes_combined[0].collections[0]

cbar_combined = fig_combined_mp.colorbar(
    mappable_combined, ax=main_axes_combined, orientation="vertical", fraction=0.03, pad=0.04, extend="both"
)
cbar_label = r"$\log_{10} \left( |\Delta z/\Delta x| / |\Delta z/\Delta y| \right)$"
cbar_combined.set_label(cbar_label, size=phelpers.label_size)

# fig_combined_mp.savefig(phelpers.fig_dir / f"fig_gradient_ratios_{flags.label}.png", dpi=200, bbox_inches="tight")
