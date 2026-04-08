"""
maps of
- climatological variance
- climatoloical skew
- how much the mean of Tx has shifted in ERA5 between ref_years and new_years
"""

from changing_heat_extremes import flags
from changing_heat_extremes import plot_helpers as phelpers
import xarray as xr
import holoviews as hv
import hvplot.xarray  # noqa: F401
import hvplot.pandas  # noqa: F401
from pathlib import Path
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

hvplot.extension(phelpers.backend_hv)


fig_dir = Path("figures")
data_dir = Path("processed_data")


combined_ds = xr.open_dataset(data_dir / f"moments_ds_{flags.label}.nc")

### also calculate the correlations for each, and add as labels ---
cor_mean_diff_var = xr.corr(
    combined_ds["t2m_x_mean_diff"],
    combined_ds["t2m_x_var"],
).values.round(2)
cor_mean_diff_skew = xr.corr(
    combined_ds["t2m_x_mean_diff"],
    combined_ds["t2m_x_skew"],
).values.round(2)

# shared plotting arguments
qm_kwargs = dict(coastline=True, projection=ccrs.PlateCarree())
fig_kwargs = dict(
    xlabel="",
    ylabel="",
    fig_inches=(phelpers.width_default, phelpers.height_wide),
    xaxis=None,
    yaxis=None,
    **phelpers.global_kwargs,
)

################
# mean shift map
################

cbar_kwargs_meanshift = phelpers.cbar_helper_hv(-0.5, 2, cmap="RdBu_r", cmap_center=0)
horizontal_cbar_meanshift = phelpers.horizontal_cbar_hv(
    clabel="Mean Shift (°C)", shrink=0.7
)

fig_meanshift = (
    combined_ds["t2m_x_mean_diff"]
    .hvplot.quadmesh(**qm_kwargs)
    .opts(
        hv.opts.QuadMesh(
            colorbar=False,
            hooks=[cbar_kwargs_meanshift, horizontal_cbar_meanshift],
            **fig_kwargs,
        )
    )
)

# add label to bottom left
# hv.Text(0.02, 0.02, 'BL', halign='left', valign='bottom')
# corner_text = hv.Text(0.02, 0.02, '(a)', halign='left', valign='bot')


################
# variance shift map
################


cbar_kwargs_var = phelpers.cbar_helper_hv(
    0, 50, cmap=phelpers.reds_cmap, extension=phelpers.backend_hv
)
horizontal_cbar_var = phelpers.horizontal_cbar_hv(
    clabel="Climatological Variance (°C²)", shrink=0.7
)

fig_var = (
    combined_ds["t2m_x_var"]
    .hvplot.quadmesh(**qm_kwargs)
    .opts(
        hv.opts.QuadMesh(
            colorbar=False,
            hooks=[cbar_kwargs_var, horizontal_cbar_var],
            **fig_kwargs,
        )
    )
)


var_text = hv.Text(
    -180 + 180,
    -60 + 11,
    f"r={str(cor_mean_diff_var)}",
    fontsize=phelpers.label_size - 2,
)

fig_var_final = (fig_var * var_text).opts(ylim=(-59, 80), xlim=(-180, 180))

################
# skew shift map
################

cbar_kwargs_skew = phelpers.cbar_helper_hv(
    -1.1, 0.5, cmap="RdBu_r", cmap_center=0, extension=phelpers.backend_hv
)
horizontal_cbar_skew = phelpers.horizontal_cbar_hv(
    clabel="Climatological Skewness", shrink=0.7
)
fig_skew = (
    combined_ds["t2m_x_skew"]
    .hvplot.quadmesh(**qm_kwargs)
    .opts(
        hv.opts.QuadMesh(
            colorbar=False,
            hooks=[cbar_kwargs_skew, horizontal_cbar_skew],
            **fig_kwargs,
        )
    )
)

skew_text = hv.Text(
    -180 + 180,
    -60 + 11,
    f"r={str(cor_mean_diff_skew)}",
    fontsize=phelpers.label_size - 2,
)

fig_skew_final = (fig_skew * skew_text).opts(ylim=(-59, 80), xlim=(-180, 180))

###################
# Combining figures
###################

# # original method. works fine as long as you're ok with things being in 1 row
# fig_moments = (fig_meanshift + fig_var_final + fig_skew_final).cols(3)
# # these subplot labels end up getting ignored if we use the mpl version.
# fig_moments.opts(
#     sublabel_format="({alpha})",
#     sublabel_position=(-0.04, -0.11),
#     sublabel_size=phelpers.tick_size,
#     tight=True,
#     tight_padding=1,
# )
# # hvplot.save(fig_moments, fig_dir / f"fig_moments_{flags.label}.png", dpi = 200)


## my attempt at manual layout with panel.
## it works, but would need to manually input subplot labels.
# import panel as pn
# fig_moments = pn.Column(
#     pn.Row(
#         pn.Spacer(width = 225),
#         fig_meanshift,
#         margin = 0
#     ),
#     pn.Row(fig_var_final, fig_skew_final, margin = 0),
# ).show()


## matplotlib layout
# my figure has 3 subplots. i want the first one on its own, and the second two below it. here's how to arrange them:
# 2 rows x 4 cols:
# - top panel spans middle 2 cols
# - bottom-left spans first 2 cols
# - bottom-right spans last 2 cols
fig = plt.figure(figsize=(10, 5), constrained_layout=False)
gs = fig.add_gridspec(
    nrows=2,
    ncols=4,
    wspace=-0.3,
    hspace=0.30,
)

ax_mean = fig.add_subplot(gs[0, 1:3], projection=ccrs.PlateCarree())
ax_var = fig.add_subplot(gs[1, 0:2], projection=ccrs.PlateCarree())
ax_skew = fig.add_subplot(gs[1, 2:4], projection=ccrs.PlateCarree())

renderer = hv.renderer("matplotlib")
renderer.get_plot(fig_meanshift, fig=fig, axis=ax_mean)
renderer.get_plot(fig_var_final, fig=fig, axis=ax_var)
renderer.get_plot(fig_skew_final, fig=fig, axis=ax_skew)

# add in the subplot labels.
for ax, lab in [(ax_mean, "(a)"), (ax_var, "(b)"), (ax_skew, "(c)")]:
    ax.text(
        0.01,
        0.02,
        lab,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=phelpers.label_size,
        # fontweight="bold",
    )

# remove whitespace outside of the plot
# need bottom=0.10 to make room for the horizontal colorbars
fig.subplots_adjust(
    left=0,
    right=1,
    bottom=0.10,
    top=1,
    wspace=0.0,
    hspace=0.0,
)
# ! uncomment to save fig!
# fig.savefig(fig_dir / f"fig_moments_{flags.label}.png", dpi=200, bbox_inches="tight")

###################################
# supplemental analyses used in the paper
####################################

# # adding contour at 20C^2, where there seems to be a elbow in the variance (in 2d_scatter_figs.py)
# fig_var_contour = combined_ds["t2m_x_var"].hvplot.contour(
#     levels=[20],
#     linewidth=2,
#     cmap=["blue"],
#     colorbar=False,
#     legend=False,
#     **qm_kwargs,
#     **phelpers.global_kwargs,
# )

# fig_var_contour_final = (fig_var * fig_var_contour).opts(
#     title="Climatological Variance\ncontour at 20°C²"
# )
# # hvplot.save(fig_var_contour_final, fig_dir / "supplemental" /  f"fig_var_contour_{flags.label}.png")


# # autocorrelation map

cor_mean_diff_ar1 = xr.corr(
    combined_ds["t2m_x_mean_diff"],
    combined_ds["t2m_x_ar1"],
).values.round(2)

cbar_kwargs_ar = phelpers.cbar_helper_hv(
    0.5, 0.9, cmap=phelpers.reds_cmap, extension=phelpers.backend_hv
)
horizontal_cbar_ar = phelpers.horizontal_cbar_hv(
    clabel="Climatological AR(1)", shrink=0.7
)
fig_ar = (
    combined_ds["t2m_x_ar1"]
    .hvplot.quadmesh(**qm_kwargs)
    .opts(
        hv.opts.QuadMesh(
            colorbar=False,
            hooks=[cbar_kwargs_ar, horizontal_cbar_ar],
            **fig_kwargs,
        )
    )
)

ar_text = hv.Text(
    -180 + 180,
    -60 + 11,
    f"r={str(cor_mean_diff_ar1)}",
    fontsize=phelpers.label_size - 2,
)

fig_ar_final = (fig_ar * ar_text).opts(ylim=(-59, 80), xlim=(-180, 180))
# # hvplot.save(fig_ar_final, fig_dir / "supplemental" /  f"fig_ar_map_{flags.label}.png", dpi = 200)
