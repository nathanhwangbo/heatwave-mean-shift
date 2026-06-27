################################################################################################
#  Goal: justify the choice to use a 1980-1999 reference period for calcualting q90 thresholds
#     compared to the more standard 1960-1990.
#
# Idea: there were more ERA data artifacts if we included the 60s, esp near the tropics.
#
# More generally: apriori, we expect that the q90 thresholds should be spatially smooth (at the coarse resolutions we're looking at, where topography is unlikely to pose huge problems)
#   so we plot an avg of the q90 threshold over the periods we're interested in (JJA in north hemsiphere and DJF in southern hemisphere)
#   using both of the reference periods, and compare
##################################################################################################
from heatwave_mean_shift import analysis_helpers as ahelpers
import xarray as xr
import numpy as np
import glob
import hdp
import cartopy.crs as ccrs
import tastymap

reds_discrete = tastymap.cook_tmap("cet_CET_L18", num_colors=12).cmap
rdbu_discrete = tastymap.cook_tmap("RdYlBu_r", num_colors=10).cmap

################################
# read in data
################################

era_filelist = glob.glob("D:data\\ERA5\\t2m_x_1x1_2025\\*.nc")
era = xr.open_mfdataset(era_filelist).rename({"t2m": "t2m_x"})
era = era.convert_calendar(calendar="noleap", use_cftime=True)
# convert to (-180, 180) lon.
era = era.assign_coords(lon=(((era.lon + 180) % 360) - 180)).sortby("lon")
era_land = ahelpers.add_landmask(era).compute()

#######################################################
# plot time series example  of problematic area in 1960s
########################################################

sudan_lon = 30
sudan_lat = 6

era_sudan = era_land.sel(lon=sudan_lon, lat=sudan_lat, method="nearest")

era_sudan.sel(time=slice("1963", "1964")).hvplot(title="see the big drop in late july 1964")

era_sudan.sel(time=slice("1964", "1965")).hvplot(title="repeated again in '65")

# this messes with the q90 threshold quite a lot!

####################################################################
# plot annual avg map of q90 threshold using 1980-1999 and 1960-1990
####################################################################


def get_q90_threshold(era_land, start_year, end_year):
    """
    add the start and end year as strings
    eg start_year = "1960" and end_year = "1990"
    """
    era_land_ref = era_land.sel(time=slice(start_year, end_year))
    ref_doy_climatology = ahelpers.fourier_climatology_smoother(era_land_ref["t2m_x"], n_time=365, n_bases=5)

    era_land_ref_anom = (era_land_ref.groupby("time.dayofyear") - ref_doy_climatology).drop_vars("dayofyear")
    era_land_ref_anom["t2m_x"].attrs = {"units": "C"}  # hdp package needs units

    # conversion to celcius
    measures_ref = hdp.measure.format_standard_measures(temp_datasets=[era_land_ref_anom["t2m_x"]])
    percentiles = [0.9]

    #### calculate doy thresholds, and smooth ----------------------
    thresholds_ref_unsmooth = hdp.threshold.compute_thresholds(
        measures_ref, percentiles, rolling_window_size=7
    ).compute()

    ## smoothing out the the threshold climatology as well
    thresholds_ref_smoothed = ahelpers.fourier_climatology_smoother(
        thresholds_ref_unsmooth["t2m_x_threshold"].sel(percentile=percentiles[0]).drop_vars("percentile"),
        n_time=365,
        n_bases=5,
    )
    # match the formatting of the original hdp function -----
    thresholds_ref = (
        thresholds_ref_smoothed.to_dataset()
        .expand_dims(percentile=percentiles)
        .transpose("lat", "lon", "doy", "percentile")
    )
    return thresholds_ref.isel(percentile=0)


# calculate thresholds over these two periods -----------------
threshold_1960_1990 = get_q90_threshold(era_land, "1960", "1990")
threshold_1980_1999 = get_q90_threshold(era_land, "1980", "1999")

# plot JJA/DJF avg maps -----------------------------------------

# get corresponding doys for JJA
jja_doy = np.unique(
    era_land.where(
        era_land.time.dt.month.isin([6, 7, 8]),
        drop=True,
    ).time.dt.dayofyear.values
)
is_in_jja = threshold_1960_1990["t2m_x_threshold"].doy.isin(jja_doy)

# get corresponding doys for DJF
djf_doy = np.unique(
    era_land.where(
        era_land.time.dt.month.isin([12, 1, 2]),
        drop=True,
    ).time.dt.dayofyear.values
)
is_in_djf = threshold_1960_1990["t2m_x_threshold"].doy.isin(djf_doy)


# get hemisphere averages
is_in_nh = threshold_1960_1990["t2m_x_threshold"].lat >= 0
jja_avg_1960_1990_nh = (
    threshold_1960_1990["t2m_x_threshold"].where(np.logical_and(is_in_nh, is_in_jja), drop=True).mean(dim="doy")
)

djf_avg_1960_1990_sh = (
    threshold_1960_1990["t2m_x_threshold"].where(np.logical_and(~is_in_nh, is_in_djf), drop=True).mean(dim="doy")
)

# repeat for 1980-1999
jja_avg_1980_1999_nh = (
    threshold_1980_1999["t2m_x_threshold"].where(np.logical_and(is_in_nh, is_in_jja), drop=True).mean(dim="doy")
)

djf_avg_1980_1999_sh = (
    threshold_1980_1999["t2m_x_threshold"].where(np.logical_and(~is_in_nh, is_in_djf), drop=True).mean(dim="doy")
)


# combine hemispheres
summer_avg_1960_1990 = xr.concat([jja_avg_1960_1990_nh, djf_avg_1960_1990_sh], dim="lat").sortby("lat")
summer_avg_1980_1999 = xr.concat([jja_avg_1980_1999_nh, djf_avg_1980_1999_sh], dim="lat").sortby("lat")
# note. we could also take avg of the diff. this is diff of the avg
summer_avg_diff = summer_avg_1960_1990 - summer_avg_1980_1999

fig_1960_1990 = summer_avg_1960_1990.hvplot(
    projection=ccrs.PlateCarree(),
    coastline=True,
    cmap=reds_discrete,
    clim=(1, 7),
    title="q90 tmax anomaly threshold\nsummer avg, 1960-1990",
    clabel="q90 anom threshold (C)",
    xlabel="",
    ylabel="",
).opts(fontscale=2.5, ylim=(-60, None))

fig_1980_1999 = summer_avg_1980_1999.hvplot(
    projection=ccrs.PlateCarree(),
    coastline=True,
    cmap=reds_discrete,
    clim=(1, 7),
    title="q90 tmax anomaly threshold\nsummer avg, 1980-1999",
    clabel="q90 anom threshold (C)",
    xlabel="",
    ylabel="",
).opts(fontscale=2.5, ylim=(-60, None))


fig_diff = summer_avg_diff.hvplot(
    projection=ccrs.PlateCarree(),
    coastline=True,
    cmap=rdbu_discrete,
    clim=(-1, 1),
    title="a - b",
    clabel="q90 anom threshold (C)",
    xlabel="",
    ylabel="",
).opts(fontscale=2.5, ylim=(-60, None))


fig_threshold = (fig_1960_1990 + fig_1980_1999).cols(2)
fig_diff
