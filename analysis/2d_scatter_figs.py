"""
generate scatterplots for a fixed 2 degree shift, with points of
- climatology moments on the x axis
- change in heatwave metrics on the y axis

inputs:
- moments_ds.nc from 2a_get_moments.py
- heatwave metrics for the synthetic 2degree shift experiment from 0_era_meanshift.py

outputs:
- fig_2deg.png
"""

from heatwave_mean_shift import flags
from heatwave_mean_shift import plot_helpers as phelpers
import xarray as xr
import polars as pl
import numpy as np
import holoviews as hv
import string

# import statsmodels.api as sm
import statsmodels.formula.api as smf
import hvplot.xarray  # noqa: F401
import hvplot.polars  # noqa: F401
from pathlib import Path

hvplot.extension(phelpers.backend_hv)


fig_dir = Path("figures")
data_dir = Path("processed_data")

fig_kwargs = dict(
    fig_inches=(phelpers.width_default, phelpers.height_default),
    **phelpers.global_kwargs,
)

layout_kwargs = dict(tight=True, tight_padding=4)


def fig_scatter(df, bin_var, bin_id_var, hw_metric, label, color, linestyle="-"):

    binned_df = (
        df.sort(bin_var)
        .group_by(bin_id_var)
        .agg(
            [
                pl.col(f"t2m_x.t2m_x_threshold.{hw_metric}").mean().alias(f"{hw_metric}_mean"),
                pl.col(f"t2m_x.t2m_x_threshold.{hw_metric}").std().alias(f"{hw_metric}_std"),
            ]
        )
    )
    # scatter = df.hvplot.scatter(
    #     x=bin_var, y=f"t2m_x.t2m_x_threshold.{hw_metric}", alpha=0.05, c=color
    # )
    # means = binned_df.hvplot.scatter(
    #     x="bin", y=f"{hw_metric}_mean", size=100, color=color
    # )
    sds = hv.ErrorBars(
        binned_df.select([bin_id_var, f"{hw_metric}_mean", f"{hw_metric}_std"]).to_numpy(),
        label=label,
    ).opts(
        alpha=0.5,
        capsize=3,
        edgecolor=color,
        linestyle=linestyle,
    )

    lines = (
        binned_df.sort(bin_id_var)
        .hvplot.line(
            x=bin_id_var,
            y=f"{hw_metric}_mean",
            color=color,
            marker=".",
            ms=10,
            linestyle=linestyle,
        )
        .opts(alpha=0.5)
    )

    # fig = scatter * means * sds
    fig = lines * sds
    return fig


###########################################3
# read in data from 2a_get_moments
###########################################

combined_ds = xr.open_dataset(data_dir / f"moments_ds_{flags.label}.nc")
combined_df = pl.from_pandas(combined_ds.to_dataframe(), include_index=True).drop_nulls()


# 2deg, observed --------------------------------


# variance, observed --------------


deg2_obs_df = combined_df.filter((pl.col("t2m_x_mean_diff") >= 1.75) & (pl.col("t2m_x_mean_diff") <= 2.25))

n_bins = 20
bins_var = np.linspace(deg2_obs_df["t2m_x_var"].min(), deg2_obs_df["t2m_x_var"].max(), n_bins)
midpoints_var = ((bins_var[:-1] + bins_var[1:]) / 2).round(1).astype(str)

bins_skew = np.linspace(deg2_obs_df["t2m_x_skew"].min(), deg2_obs_df["t2m_x_skew"].max(), n_bins)
midpoints_skew = ((bins_skew[:-1] + bins_skew[1:]) / 2).round(1).astype(str)

bins_ar1 = np.linspace(deg2_obs_df["t2m_x_ar1"].min(), deg2_obs_df["t2m_x_ar1"].max(), n_bins)
midpoints_ar1 = ((bins_ar1[:-1] + bins_ar1[1:]) / 2).round(1).astype(str)


deg2_obs_df = deg2_obs_df.with_columns(
    var_bin_id=pl.col("t2m_x_var").cut(breaks=bins_var[1:-1], labels=midpoints_var).cast(pl.String).cast(pl.Float64),
    skew_bin_id=pl.col("t2m_x_skew")
    .cut(breaks=bins_skew[1:-1], labels=midpoints_skew)
    .cast(pl.String)
    .cast(pl.Float64),
    ar1_bin_id=pl.col("t2m_x_ar1").cut(breaks=bins_ar1[1:-1], labels=midpoints_ar1).cast(pl.String).cast(pl.Float64),
)

fig_var_hwf_obs = fig_scatter(deg2_obs_df, "t2m_x_var", "var_bin_id", "HWF", "Observed", "red").opts(
    hv.opts.Scatter(alpha=0.2)
)
fig_var_hwd_obs = fig_scatter(deg2_obs_df, "t2m_x_var", "var_bin_id", "HWD", "Observed", "red").opts(
    hv.opts.Scatter(alpha=0.2)
)
fig_var_sumheat_obs = fig_scatter(deg2_obs_df, "t2m_x_var", "var_bin_id", "sumHeat", "Observed", "red").opts(
    hv.opts.Scatter(alpha=0.2)
)


# skewness, observed --------------
n_bins = 20
bin_size_skew = (deg2_obs_df["t2m_x_skew"].max() - deg2_obs_df["t2m_x_skew"].min()) / n_bins

fig_skew_hwf_obs = fig_scatter(deg2_obs_df, "t2m_x_skew", "skew_bin_id", "HWF", "Observed", "red").opts(
    hv.opts.Scatter(alpha=0.05)
)
fig_skew_hwd_obs = fig_scatter(deg2_obs_df, "t2m_x_skew", "skew_bin_id", "HWD", "Observed", "red").opts(
    hv.opts.Scatter(alpha=0.05)
)
fig_skew_sumheat_obs = fig_scatter(deg2_obs_df, "t2m_x_skew", "skew_bin_id", "sumHeat", "Observed", "red").opts(
    hv.opts.Scatter(alpha=0.05)
)


# 2 degree, synthetic ---------------------------------------------------


hw_synth_2deg = (
    xr.open_dataset(data_dir / f"hw_metrics_{flags.ref_years[0]}_{flags.new_years[1]}_synth_2deg_anom{flags.label}.nc")
    .sel(percentile=flags.percentile_threshold, definition="3-0-0")
    .drop_vars(["percentile", "definition"])
)

hw_old_2deg = hw_synth_2deg.sel(time=slice(str(flags.ref_years[0]), str(flags.ref_years[1])))
hw_new_2deg = hw_synth_2deg.sel(time=slice(str(flags.new_years[0]), str(flags.new_years[1])))
hw_mean_diff_2deg = hw_new_2deg.mean(dim="time") - hw_old_2deg.mean(dim="time")

# pull out just the climatology variables
climatology_stats = combined_ds[["t2m_x_skew", "t2m_x_kurt", "t2m_x_var", "t2m_x_ar1"]]
combined_synth_2deg_ds = xr.merge([climatology_stats, hw_mean_diff_2deg], join="exact")
# combined_synth_2deg_df = combined_synth_2deg_ds.to_dataframe().dropna(how="all")  # this just drops ocean gridcells
combined_synth_2deg_df = pl.from_pandas(combined_synth_2deg_ds.to_dataframe(), include_index=True).drop_nulls()

combined_synth_2deg_df = combined_synth_2deg_df.with_columns(
    var_bin_id=pl.col("t2m_x_var").cut(breaks=bins_var[1:-1], labels=midpoints_var).cast(pl.String).cast(pl.Float64),
    skew_bin_id=pl.col("t2m_x_skew")
    .cut(breaks=bins_skew[1:-1], labels=midpoints_skew)
    .cast(pl.String)
    .cast(pl.Float64),
    ar1_bin_id=pl.col("t2m_x_ar1").cut(breaks=bins_ar1[1:-1], labels=midpoints_ar1).cast(pl.String).cast(pl.Float64),
)


# variance -----------------------
fig_var_hwf_synth = fig_scatter(
    combined_synth_2deg_df, "t2m_x_var", "var_bin_id", "HWF", "Synthetic", "blue", "--"
).opts(hv.opts.Scatter(alpha=0.01))
fig_var_hwd_synth = fig_scatter(
    combined_synth_2deg_df, "t2m_x_var", "var_bin_id", "HWD", "Synthetic", "blue", "--"
).opts(hv.opts.Scatter(alpha=0.01))
fig_var_sumheat_synth = fig_scatter(
    combined_synth_2deg_df,
    "t2m_x_var",
    "var_bin_id",
    "sumHeat",
    "Synthetic",
    "blue",
    "--",
).opts(hv.opts.Scatter(alpha=0.01))

# skewness ----------------------
fig_skew_hwf_synth = fig_scatter(
    combined_synth_2deg_df,
    "t2m_x_skew",
    "skew_bin_id",
    "HWF",
    "Synthetic",
    "blue",
    "--",
).opts(hv.opts.Scatter(alpha=0.01))
fig_skew_hwd_synth = fig_scatter(
    combined_synth_2deg_df,
    "t2m_x_skew",
    "skew_bin_id",
    "HWD",
    "Synthetic",
    "blue",
    "--",
).opts(hv.opts.Scatter(alpha=0.01))
fig_skew_sumheat_synth = fig_scatter(
    combined_synth_2deg_df,
    "t2m_x_skew",
    "skew_bin_id",
    "sumHeat",
    "Synthetic",
    "blue",
    "--",
).opts(hv.opts.Scatter(alpha=0.01))


###########
# combine
##########
fig_var_hwf = (fig_var_hwf_synth * fig_var_hwf_obs).opts(
    xlabel="Climatological Variance (°C²)",
    ylabel=r"$\Delta$ HWF (days)",
    xlim=(0, 60),
    ylim=(0, 50),
    show_legend=False,
    hooks=[phelpers.shared_ylabel_hv(xpos=0.05)],
    **fig_kwargs,
)
fig_var_hwd = (fig_var_hwd_synth * fig_var_hwd_obs).opts(
    xlabel="Climatological Variance (°C²)",
    ylabel=r"$\Delta$ HWD (days)",
    xlim=(0, 60),
    ylim=(0, 20),
    show_legend=False,
    hooks=[phelpers.shared_ylabel_hv(xpos=0.36)],
    **fig_kwargs,
)
fig_var_sumheat = (fig_var_sumheat_synth * fig_var_sumheat_obs).opts(
    xlabel="Climatological Variance (°C²)",
    ylabel=r"$\Delta$ sumHeat (°C-days)",
    xlim=(0, 60),
    ylim=(0, 75),
    hooks=[phelpers.shared_ylabel_hv(xpos=0.66)],
    **fig_kwargs,
)

fig_skew_hwf = (fig_skew_hwf_synth * fig_skew_hwf_obs).opts(
    xlabel="Climatological Skew",
    ylabel="",
    # ylabel=r"$\Delta$ HWF (days)",
    xlim=(-1.25, 0.75),
    ylim=(-1, 50),
    show_legend=False,
    **fig_kwargs,
)
fig_skew_hwd = (fig_skew_hwd_synth * fig_skew_hwd_obs).opts(
    xlabel="Climatological Skew",
    ylabel="",
    # ylabel=r"$\Delta$ HWD (days)",
    xlim=(-1.25, 0.75),
    ylim=(-2, 20),
    show_legend=False,
    **fig_kwargs,
)
fig_skew_sumheat = (fig_skew_sumheat_synth * fig_skew_sumheat_obs).opts(
    xlabel="Climatological Skew",
    ylabel="",
    # ylabel=r"$\Delta$ sumHeat (°C-days)",
    xlim=(-1.25, 0.75),
    ylim=(-2, 75),
    show_legend=False,
    **fig_kwargs,
)

fig_scatter_combined = (
    (fig_var_hwf + fig_var_hwd + fig_var_sumheat + fig_skew_hwf + fig_skew_hwd + fig_skew_sumheat)
    .cols(3)
    .opts(
        shared_axes=False,
        sublabel_format="",
        # sublabel_format="({alpha})",
        # sublabel_position=(0.8, 0.8),
        # sublabel_size=phelpers.tick_size,
        **layout_kwargs,
    )
)
# fig_scatter
# hvplot.save(fig_scatter_combined, phelpers.fig_dir / f"fig_2deg_{flags.label}.png")

# convert to matplotlib and add labels manually
fig = hv.render(fig_scatter_combined, backend="matplotlib")
num_panels = len(fig_scatter_combined)
# Keep only main panel axes (exclude colorbar/small helper axes)
main_axes = sorted(
    fig.axes,
    key=lambda ax: ax.get_position().width * ax.get_position().height,
    reverse=True,
)[:num_panels]

# Ensure panel order is row-major: top-left -> top-right, then next row
main_axes = sorted(
    main_axes,
    key=lambda ax: (-ax.get_position().y0, ax.get_position().x0),
)

labels = [f"({c})" for c in string.ascii_lowercase[:num_panels]]
for ax, lab in zip(main_axes, labels, strict=False):
    ax.text(
        0.98,
        1.15,
        lab,  # top-right inside panel
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=phelpers.label_size,
        fontweight="normal",
        zorder=30,
    )

#! main output of this script.
fig.savefig(phelpers.fig_dir / f"fig_2deg_{flags.label}.png", dpi=200, bbox_inches="tight")
# fig.savefig(phelpers.fig_dir / "tmp.png", dpi=200, bbox_inches="tight")

#################################
# misc results cited in the paper
#################################

## fitting least squares lines ------

# skew, synthetic
ols_skew_synth_hwf = (
    smf.ols(
        formula="Q('t2m_x.t2m_x_threshold.HWF') ~ t2m_x_skew",
        data=combined_synth_2deg_df.to_pandas(),
    )
    .fit()
    .summary()
)
ols_skew_synth_hwd = (
    smf.ols(
        formula="Q('t2m_x.t2m_x_threshold.HWD') ~ t2m_x_skew",
        data=combined_synth_2deg_df.to_pandas(),
    )
    .fit()
    .summary()
)
ols_skew_synth_sumheat = (
    smf.ols(
        formula="Q('t2m_x.t2m_x_threshold.sumHeat') ~ t2m_x_skew",
        data=combined_synth_2deg_df.to_pandas(),
    )
    .fit()
    .summary()
)

# skew, obs
ols_skew_obs_hwf = (
    smf.ols(formula="Q('t2m_x.t2m_x_threshold.HWF') ~ t2m_x_skew", data=deg2_obs_df.to_pandas()).fit().summary()
)

smf.ols(formula="Q('t2m_x.t2m_x_threshold.HWD') ~ t2m_x_skew", data=deg2_obs_df.to_pandas()).fit().summary()

ols_skew_obs_sumheat = (
    smf.ols(
        formula="Q('t2m_x.t2m_x_threshold.sumHeat') ~ t2m_x_skew",
        data=deg2_obs_df.to_pandas(),
    )
    .fit()
    .summary()
)


# var, synthetic
ols_var_synth_hwf = smf.ols(
    formula="Q('t2m_x.t2m_x_threshold.HWF') ~ t2m_x_var",
    data=combined_synth_2deg_df.to_pandas(),
).fit()
ols_var_synth_hwd = smf.ols(
    formula="Q('t2m_x.t2m_x_threshold.HWD') ~ t2m_x_var",
    data=combined_synth_2deg_df.to_pandas(),
).fit()
ols_var_synth_sumheat = smf.ols(
    formula="Q('t2m_x.t2m_x_threshold.sumHeat') ~ t2m_x_var",
    data=combined_synth_2deg_df.to_pandas(),
).fit()

# var, obs
ols_var_obs_hwf = smf.ols(formula="Q('t2m_x.t2m_x_threshold.HWF') ~ t2m_x_var", data=deg2_obs_df.to_pandas()).fit()
ols_var_obs_hwd = smf.ols(formula="Q('t2m_x.t2m_x_threshold.HWD') ~ t2m_x_var", data=deg2_obs_df.to_pandas()).fit()
ols_var_obs_sumheat = smf.ols(
    formula="Q('t2m_x.t2m_x_threshold.sumHeat') ~ t2m_x_var",
    data=deg2_obs_df.to_pandas(),
).fit()


########################################################
# instead, let's split by var >20 and get separate slopes
########################################################


# 3. Define helper to fit and get slope
def get_slope(df, y_var, x_var):
    model = smf.ols(formula=f"Q('{y_var}') ~ {x_var}", data=df.to_pandas()).fit()
    return model.params[x_var]


metrics = ["HWF", "HWD", "sumHeat"]
results = {}  # will contain results, with order (synth, obs)

for metric in metrics:
    y_col = f"t2m_x.t2m_x_threshold.{metric}"

    # Split by variance 20
    obs_low = deg2_obs_df.filter(pl.col("t2m_x_var") < 20)
    obs_high = deg2_obs_df.filter(pl.col("t2m_x_var") >= 20)
    synth_low = combined_synth_2deg_df.filter(pl.col("t2m_x_var") < 20)
    synth_high = combined_synth_2deg_df.filter(pl.col("t2m_x_var") >= 20)

    # Slopes for low variance (< 20)
    slope_obs_low = get_slope(obs_low, y_col, "t2m_x_var")
    slope_synth_low = get_slope(synth_low, y_col, "t2m_x_var")

    # Slopes for high variance (>= 20)
    slope_obs_high = get_slope(obs_high, y_col, "t2m_x_var")
    slope_synth_high = get_slope(synth_high, y_col, "t2m_x_var")

    # order is (synth, obs)
    results[f"{metric}_low_var"] = [slope_synth_low, slope_obs_low]
    results[f"{metric}_high_var"] = [slope_synth_high, slope_obs_high]

results


########################################################
# scatterplots for ar1
########################################################

# 2deg, observed --------------------------------

# ar1, observed --------------
n_bins = 20
bin_size_ar1 = (deg2_obs_df["t2m_x_ar1"].max() - deg2_obs_df["t2m_x_ar1"].min()) / n_bins

fig_ar1_hwf_obs = fig_scatter(deg2_obs_df, "t2m_x_ar1", "ar1_bin_id", "HWF", "Observed", "red").opts(
    hv.opts.Scatter(alpha=0.05)
)
fig_ar1_hwd_obs = fig_scatter(deg2_obs_df, "t2m_x_ar1", "ar1_bin_id", "HWD", "Observed", "red").opts(
    hv.opts.Scatter(alpha=0.05)
)
fig_ar1_sumheat_obs = fig_scatter(deg2_obs_df, "t2m_x_ar1", "ar1_bin_id", "sumHeat", "Observed", "red").opts(
    hv.opts.Scatter(alpha=0.05)
)


# 2 degree, synthetic ---------------------------------------------------


# ar1 ----------------------
fig_ar1_hwf_synth = fig_scatter(
    combined_synth_2deg_df,
    "t2m_x_ar1",
    "ar1_bin_id",
    "HWF",
    "Synthetic",
    "blue",
    "--",
).opts(hv.opts.Scatter(alpha=0.01))
fig_ar1_hwd_synth = fig_scatter(
    combined_synth_2deg_df,
    "t2m_x_ar1",
    "ar1_bin_id",
    "HWD",
    "Synthetic",
    "blue",
    "--",
).opts(hv.opts.Scatter(alpha=0.01))
fig_ar1_sumheat_synth = fig_scatter(
    combined_synth_2deg_df,
    "t2m_x_ar1",
    "ar1_bin_id",
    "sumHeat",
    "Synthetic",
    "blue",
    "--",
).opts(hv.opts.Scatter(alpha=0.01))


###########
# combine
##########

fig_ar1_hwf = (fig_ar1_hwf_synth * fig_ar1_hwf_obs).opts(
    xlabel="Climatological AR(1)",
    ylabel="",
    # ylabel=r"$\Delta$ HWF (days)",
    xlim=(0, 1),
    ylim=(-1, 50),
    show_legend=False,
    **fig_kwargs,
)
fig_ar1_hwd = (fig_ar1_hwd_synth * fig_ar1_hwd_obs).opts(
    xlabel="Climatological AR(1)",
    ylabel="",
    # ylabel=r"$\Delta$ HWD (days)",
    xlim=(0, 1),
    ylim=(-2, 20),
    show_legend=False,
    **fig_kwargs,
)
fig_ar1_sumheat = (fig_ar1_sumheat_synth * fig_ar1_sumheat_obs).opts(
    xlabel="Climatological AR(1)",
    ylabel="",
    # ylabel=r"$\Delta$ sumHeat (°C-days)",
    xlim=(0, 1),
    ylim=(-2, 75),
    show_legend=False,
    **fig_kwargs,
)

fig_scatter_ar1 = (
    (fig_ar1_hwf + fig_ar1_hwd + fig_ar1_sumheat)
    .cols(3)
    .opts(
        shared_axes=False,
        sublabel_format="",
        # sublabel_format="({alpha})",
        # sublabel_position=(0.8, 0.8),
        # sublabel_size=phelpers.tick_size,
        **layout_kwargs,
    )
)
# hvplot.save(fig_scatter_ar1, phelpers.fig_dir / "supplemental" / f"fig_2deg_ar1_{flags.label}.png", dpi=200)

#############################################################
## supplemental figure:
# observed heatwave changes as a function of normalized mean shift
# that is, mean shift divided by climatological standard deviation.

# x axis: temperature mean shift / standard deviation
# y axis: heatwave metric change (e.g. HWF change)
#############################################################

# x-axis: temperature mean shift / standard deviation
combined_df = combined_df.with_columns(normalized_mean_shift=pl.col("t2m_x_mean_diff") / pl.col("t2m_x_var").sqrt())


n_bins = 20
bins_normalized_meanshift_all = np.linspace(
    combined_df["normalized_mean_shift"].min(),
    combined_df["normalized_mean_shift"].max(),
    n_bins,
)
midpoints_normalized_meanshift_all = (
    ((bins_normalized_meanshift_all[:-1] + bins_normalized_meanshift_all[1:]) / 2).round(1).astype(str)
)

all_obs_df = combined_df.with_columns(
    normalized_mean_shift_bin_id=pl.col("normalized_mean_shift")
    .cut(
        breaks=bins_normalized_meanshift_all[1:-1],
        labels=midpoints_normalized_meanshift_all,
    )
    .cast(pl.String)
    .cast(pl.Float64),
)

fig_normalized_meanshift_hwf_obs = fig_scatter(
    all_obs_df,
    "normalized_mean_shift",
    "normalized_mean_shift_bin_id",
    "HWF",
    "Observed",
    "red",
).opts(hv.opts.Scatter(alpha=0.2))
fig_normalized_meanshift_hwd_obs = fig_scatter(
    all_obs_df,
    "normalized_mean_shift",
    "normalized_mean_shift_bin_id",
    "HWD",
    "Observed",
    "red",
).opts(hv.opts.Scatter(alpha=0.2))
fig_normalized_meanshift_sumheat_obs = fig_scatter(
    all_obs_df,
    "normalized_mean_shift",
    "normalized_mean_shift_bin_id",
    "sumHeat",
    "Observed",
    "red",
).opts(hv.opts.Scatter(alpha=0.2))


(fig_normalized_meanshift_hwf_obs + fig_normalized_meanshift_hwd_obs + fig_normalized_meanshift_sumheat_obs).opts(
    shared_axes=False
).cols(1)


# alte rvsion with not-normalized mean shift on x

n_bins = 20
bins_meanshift_all = np.linspace(combined_df["t2m_x_mean_diff"].min(), combined_df["t2m_x_mean_diff"].max(), n_bins)
midpoints_meanshift_all = ((bins_meanshift_all[:-1] + bins_meanshift_all[1:]) / 2).round(1).astype(str)

all_obs_df = combined_df.with_columns(
    t2m_x_mean_diff_bin_id=pl.col("t2m_x_mean_diff")
    .cut(breaks=bins_meanshift_all[1:-1], labels=midpoints_meanshift_all)
    .cast(pl.String)
    .cast(pl.Float64),
)

fig_meanshift_hwf_obs = fig_scatter(
    all_obs_df, "t2m_x_mean_diff", "t2m_x_mean_diff_bin_id", "HWF", "Observed", "red"
).opts(hv.opts.Scatter(alpha=0.2))
fig_meanshift_hwd_obs = fig_scatter(
    all_obs_df, "t2m_x_mean_diff", "t2m_x_mean_diff_bin_id", "HWD", "Observed", "red"
).opts(hv.opts.Scatter(alpha=0.2))
fig_meanshift_sumheat_obs = fig_scatter(
    all_obs_df,
    "t2m_x_mean_diff",
    "t2m_x_mean_diff_bin_id",
    "sumHeat",
    "Observed",
    "red",
).opts(hv.opts.Scatter(alpha=0.2))


(fig_meanshift_hwf_obs + fig_meanshift_hwd_obs + fig_meanshift_sumheat_obs).opts(shared_axes=False).cols(1)
