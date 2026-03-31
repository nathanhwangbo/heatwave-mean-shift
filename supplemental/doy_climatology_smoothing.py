from changing_heat_extremes import analysis_helpers as ahelpers
import xarray as xr
import hvplot.xarray
import tastymap
import holoviews as hv
import glob
import hvplot.pandas
import hdp

rdbu_discrete = tastymap.cook_tmap("RdYlBu_r", num_colors=12).cmap
reds_discrete = tastymap.cook_tmap("cet_CET_L18", num_colors=11)[1:10].cmap  # get rid of white
blues_discrete = tastymap.cook_tmap("blues", num_colors=10).cmap


#######################################################################
# Calculate mean differences (1986-2021) - (1950-1985) for temperature
#######################################################################

era_filelist = glob.glob("D:data\\ERA5\\t2m_x_1x1\\*.nc")
era = xr.open_mfdataset(era_filelist).drop_vars("expver")

era = era.convert_calendar(calendar="noleap", use_cftime=True)
era = era.sel(lat=slice(-60, 80))  # matching karen's doy mask

era_land = ahelpers.add_landmask(era).compute()

##### calculate anomalies relative to 1950-1985 #######
era_land_ref = era_land.sel(time=slice("1950", "1985"))
# Note! we're taking anomalies with respect to the REFERENCE time period
ref_doy_climatology = era_land_ref.groupby("time.dayofyear").mean()

# pull out a few test cases


def get_doy_climatology_casestudy(da, label="", ylab=""):
    """
    da is an xarray data array, with dimensions (dayofyear, lat, lon)
    lon is assumed to be in (0, 360)
    """
    # test_lon = -60
    # test_lon = np.mod(test_lon, 360)
    lon_sw = 245
    lat_sw = 35
    test_sw = da.sel(lat=lat_sw, lon=lon_sw, method="nearest")
    fig_sw = test_sw.hvplot(
        title=f"north american southwest (lonlat is ({lon_sw}, {lat_sw}))",
        label=label,
        legend="right",
        ylabel=ylab,
    )

    # cambodia
    lon_khm = 104
    lat_khm = 11
    tmax_clim_khm = da.sel(lat=lat_khm, lon=lon_khm, method="nearest")
    fig_khm = tmax_clim_khm.hvplot(
        title=f"cambodia  (lonlat is ({lon_khm}, {lat_khm}))",
        label=label,
        legend="right",
        ylabel=ylab,
    )

    # norway
    lon_nor = 10
    lat_nor = 59
    tmax_clim_nor = da.sel(lat=lat_nor, lon=lon_nor, method="nearest")
    fig_nor = tmax_clim_nor.hvplot(
        title=f"norway (lonlat is ({lon_nor}, {lat_nor}))",
        label=label,
        legend="right",
        ylabel=ylab,
    )

    # bolivia
    lon_bol = 300
    lat_bol = -17
    tmax_clim_bol = da.sel(lat=lat_bol, lon=lon_bol, method="nearest")
    fig_bol = tmax_clim_bol.hvplot(
        title=f"bolivia  (lonlat is ({lon_bol}, {lat_bol}))",
        label=label,
        legend="right",
        ylabel=ylab,
    )

    fig_tmax_climatology_baseline = (fig_sw + fig_khm + fig_nor + fig_bol).cols(2)
    return fig_tmax_climatology_baseline


fig_tmax_climatology_baseline = get_doy_climatology_casestudy(ref_doy_climatology["t2m_x"])
hvplot.save(fig_tmax_climatology_baseline, "fig_tmax_climatology_casestudy.html")


### compare with smoothed climatology -----------------------

ref_doy_climatology = ahelpers.fourier_climatology_smoother(era_land_ref["t2m_x"], n_time=365, n_bases=5)
fig_tmax_climatology_smoothed = get_doy_climatology_casestudy(ref_doy_climatology)

figlist_tmax_climatology_compare = []
for i in range(len(fig_tmax_climatology_baseline)):
    figlist_tmax_climatology_compare.append(fig_tmax_climatology_baseline[i] * fig_tmax_climatology_smoothed[i])

fig_tmax_climatology_compare = hv.Layout(figlist_tmax_climatology_compare).cols(2)
hvplot.save(fig_tmax_climatology_compare, "fig_tmax_climatology_casestudy.html")
############################################


# take anomalies
era_land_ref_anom = (era_land_ref.groupby("time.dayofyear") - ref_doy_climatology).drop_vars("dayofyear")
era_land_ref_anom["t2m_x"].attrs = {"units": "C"}  # hdp package needs units

# conversion to celcius
measures_ref = hdp.measure.format_standard_measures(temp_datasets=[era_land_ref_anom["t2m_x"]])
percentiles = [0.9]


# try 3 different windows
thresholds_ref7 = (
    hdp.threshold.compute_thresholds(measures_ref, percentiles, rolling_window_size=7)["t2m_x_threshold"]
    .isel(percentile=0)
    .transpose("doy", "lat", "lon")
    .compute()
)

thresholds_ref15 = (
    hdp.threshold.compute_thresholds(measures_ref, percentiles, rolling_window_size=15)["t2m_x_threshold"]
    .isel(percentile=0)
    .transpose("doy", "lat", "lon")
    .compute()
)

thresholds_ref30 = (
    hdp.threshold.compute_thresholds(measures_ref, percentiles, rolling_window_size=30)["t2m_x_threshold"]
    .isel(percentile=0)
    .transpose("doy", "lat", "lon")
    .compute()
)


# take a look
fig_q90_baseline7 = get_doy_climatology_casestudy(
    thresholds_ref7,
    label="7day",
    ylab="q90 threshold (anom C)",
)
fig_q90_baseline15 = get_doy_climatology_casestudy(
    thresholds_ref15,
    label="15day",
    ylab="q90 threshold (anom C)",
)
fig_q90_baseline30 = get_doy_climatology_casestudy(
    thresholds_ref30,
    label="30day",
    ylab="q90 threshold (anom C)",
)

figlist_q90_climatology_compare = []
for i in range(len(fig_q90_baseline7)):
    figlist_q90_climatology_compare.append(fig_q90_baseline7[i] * fig_q90_baseline15[i] * fig_q90_baseline30[i])

fig_q90_climatology_compare = hv.Layout(figlist_q90_climatology_compare).cols(2)
hvplot.save(fig_q90_climatology_compare, "fig_q90_climatology_casestudy.html")


#### now try smoothing all the thresholds and see how they look
q90_smoothed_climatology7 = ahelpers.fourier_climatology_smoother(
    thresholds_ref7, n_time=365, n_bases=5, is_time_dim=False
)
q90_smoothed_climatology15 = ahelpers.fourier_climatology_smoother(
    thresholds_ref15, n_time=365, n_bases=5, is_time_dim=False
)
q90_smoothed_climatology30 = ahelpers.fourier_climatology_smoother(
    thresholds_ref30, n_time=365, n_bases=5, is_time_dim=False
)
# take a look
fig_q90_baseline7_smooth = get_doy_climatology_casestudy(
    q90_smoothed_climatology7,
    label="7day",
    ylab="q90 threshold (anom C)",
)
fig_q90_baseline15_smooth = get_doy_climatology_casestudy(
    q90_smoothed_climatology15,
    label="15day",
    ylab="q90 threshold (anom C)",
)
fig_q90_baseline30_smooth = get_doy_climatology_casestudy(
    q90_smoothed_climatology30,
    label="30day",
    ylab="q90 threshold (anom C)",
)

figlist_q90_climatology_compare_smooth = []
for i in range(len(fig_q90_baseline7_smooth)):
    figlist_q90_climatology_compare_smooth.append(
        fig_q90_baseline7_smooth[i] * fig_q90_baseline15_smooth[i] * fig_q90_baseline30_smooth[i]
    )

fig_q90_climatology_compare_smooth = hv.Layout(figlist_q90_climatology_compare_smooth).cols(2)
# hvplot.save(fig_q90_climatology_compare_smooth, "fig_q90_climatology_smooth_casestudy.html")
