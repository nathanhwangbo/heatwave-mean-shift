"""script to calculate summer land heatwave metrics from ERA

inputs:
    - daily maximum temperature from ERA on a 1x1 grid (downloaded in in src/00_get_era_daily_tmax.py and processed in src/01_process_era.py)
    - (optional) a doy-mask specifying the "summer" season, if using the heat-based definition of summer instead of JJA.

outputs (for use in subsequent scripts in /analysis/):
    - a map with the q90(tmax) threshold used to define heatwaves (uses reference period 1960-1990)
    - heatwave metrics per year                                   (uses all years)

- To estimate heatwave metrics, we:
    - 1. estimate the day-of-year climatology by taking the mean across 1960:1990 and smoothing via 5 fourier basis functions
    - 2. remove this day-of-year climatology, then calculate the q90 thresholds
        - q90 is specific to each gridcell and each day-of-year, using 7 day windows around each day-of-year
        - i.e. for each day of year, q90 is estimated from 7 days * 31 years
        - The q90 thresholds are then smoothed via 5 fourier basis functions
    - 3. use this threshold to calculate heatwave metrics across 1960-2025 (one value per summer)
        - heatwaves are defined as 3+ consecutive days where the anomalies exceed q90


We note that by calculating q90 over just the 1960-1990 period, we guarantee that every gridcell has days which exceed this threshold in the 1960-1990 period.
"""

from heatwave_mean_shift import flags
from heatwave_mean_shift import analysis_helpers as ahelpers
import xarray as xr
import hdp  # pip install -e "E:\\Projects\\HDP\\"
from pathlib import Path
from dask.diagnostics import ProgressBar
import resource

# Set limit to 200 GB to make sure we don't blow up the server
memory_limit_gb = 200
limit_bytes = memory_limit_gb * 1024 * 1024 * 1024
resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))

pbar = ProgressBar(dt=1)
pbar.register()  # show a progress bar on dask computations

era5_path = Path(flags.era5_path)
processed_data_path = Path("processed_data")  # outputs are saved here.


#########################################################
#### DECISION! is summer defined via calendar year (JJA?)
# if false, summer is defined via 90 days surrounding the hottest day (for each latitude)
# flag is set in flags.py
#########################################################
if flags.use_calendar_summer is False:
    # day of year summer mask
    era_summer_path = era5_path / "ERA5_hottest_doys_t2m_x.nc"
    era_summer_mask = xr.open_dataarray(era_summer_path)


#######################################
# read in ERA data
#######################################

if flags.use_1x1:
    # to use 1x1 regridded data ---
    era_filelist = sorted(list((era5_path / "t2m_x_1x1").glob("*.nc")))
    era = (
        xr.open_mfdataset(era_filelist)
        .rename({"__xarray_dataarray_variable__": "t2m_x", "valid_time": "time"})
        .reset_coords(names="number", drop=True)
    )
else:
    ## use native resolution data ---
    era_filelist = sorted(list((era5_path / "t2m_x_daily").glob("t2m_x*.nc")))
    era = (
        xr.open_mfdataset(era_filelist)
        .rename(
            {
                "t2m": "t2m_x",
                "valid_time": "time",
                "latitude": "lat",
                "longitude": "lon",
            }
        )
        .sortby("lat")
    )

    era = era.drop_vars("number")  # remove empty coord


# fixing formatting for the hdp package
era["t2m_x"].attrs = {"units": "K"}  # hdp package needs units
era = era.convert_calendar(calendar="noleap", use_cftime=True)
era = era.sel(lat=slice(-60, 80)).chunk({"time": -1, "lat": 30, "lon": 30})  # matching karen's doy mask

# convert to (-180, 180) lon.
era = era.assign_coords(lon=(((era.lon + 180) % 360) - 180)).sortby("lon")


# add landmask
era_land = ahelpers.add_landmask(era)  # .compute()


#################################################################
# Calculate base period (1960-1990) characteristics
#### calculate doy thresholds, and smooth the threshold ---------
##################################################################

thresholds_path = processed_data_path / f"thresholds_{flags.ref_years[0]}_{flags.ref_years[1]}_{flags.label}.nc"

# skip calculating thresholds if precomputed, to save time.
if thresholds_path.exists():
    print("loading existing threshold file")
    thresholds_ref = xr.open_dataset(thresholds_path).chunk({"lat": 30, "lon": 30})
else:
    print("calculating thresholds")

    era_land_ref_years = era_land.sel(time=slice(str(flags.ref_years[0] - 1), str(flags.ref_years[1])))
    # take doy anomalies
    ref_doy_climatology = ahelpers.fourier_climatology_smoother(
        era_land_ref_years["t2m_x"], n_time=365, n_bases=5
    ).chunk({"dayofyear": -1, "lat": 30, "lon": 30})
    era_land_ref_anom = (era_land_ref_years.groupby("time.dayofyear") - ref_doy_climatology).drop_vars("dayofyear")
    era_land_ref_anom["t2m_x"].attrs = {"units": "C"}  # hdp package needs units. K or C is same for anoms

    # conversion to celcius
    measures_ref = hdp.measure.format_standard_measures(temp_datasets=[era_land_ref_anom["t2m_x"]])
    thresholds_ref_unsmooth = hdp.threshold.compute_thresholds(
        measures_ref, percentiles=[flags.percentile_threshold], rolling_window_size=7
    ).compute()

    ## smoothing out the the threshold climatology as well --------

    thresholds_ref_smoothed = ahelpers.fourier_climatology_smoother(
        thresholds_ref_unsmooth["t2m_x_threshold"].sel(percentile=flags.percentile_threshold).drop_vars("percentile"),
        n_time=365,
        n_bases=5,
    )

    # match the formatting of the original hdp function ---
    thresholds_ref = (
        thresholds_ref_smoothed.to_dataset()
        .expand_dims(percentile=[flags.percentile_threshold])
        .transpose("lat", "lon", "doy", "percentile")
    )
    thresholds_ref["t2m_x_threshold"].attrs["baseline_variable"] = "t2m_x"
    thresholds_ref["t2m_x_threshold"].attrs["hdp_type"] = "threshold"
    thresholds_ref["t2m_x_threshold"].attrs["baseline_calendar"] = "noleap"
    thresholds_ref = thresholds_ref.chunk({"lat": 30, "lon": 30})

    # ! uncomment to save output
    ahelpers.write_nc(
        thresholds_ref,
        thresholds_path,
    )


################################################
# calculate extremal metrics at each gridcell
###############################################

############################
# observations (not synthetic)
# time period ALL 1960-2025,
##############################

land_anom_path = processed_data_path / f"land_anom_{flags.ref_years[0]}_{flags.ref_years[1]}.nc"
# skip calculating anomalies if precomputed, to save time.
if land_anom_path.exists():
    print("loading existing land anomaly file")
    era_land_all_anom = xr.open_dataset(land_anom_path).chunk({"time": -1, "lat": 30, "lon": 30})

else:
    era_land_all = era_land.sel(time=slice(str(flags.ref_years[0] - 1), str(flags.new_years[1])))
    era_land_all_anom = (era_land_all.groupby("time.dayofyear") - ref_doy_climatology).drop_vars("dayofyear")

    # ! uncomment to save output. this is is the data that's used to calculate heatwave metrics
    ahelpers.write_nc(
        era_land_all_anom,
        f"processed_data/land_anom_{flags.ref_years[0]}_{flags.ref_years[1]}.nc",
    )

era_land_all_anom["t2m_x"].attrs = {"units": "C"}  # hdp package needs units
# use c when dealing with anomalies (bc anomalies are the same in either units)


# calculate hw metrics if it doesn't already exist ----------
hw_metrics_obs_path = processed_data_path / f"hw_metrics_{flags.ref_years[0]}_{flags.new_years[1]}_anom{flags.label}.nc"
if hw_metrics_obs_path.exists():
    print("loading existing heatwave metrics for obs")
    metrics_all_land = xr.open_dataset(hw_metrics_obs_path).chunk({"lat": 30, "lon": 30})
else:
    # calculate heatwave metrics on time period.
    measures_all = hdp.measure.format_standard_measures(
        temp_datasets=[era_land_all_anom["t2m_x"]]
    )  # .chunk({"time": -1, "lat": 30, "lon": 30})

    print("calculating heatwave metrics for obs, 1960-2025")
    if flags.use_calendar_summer:
        metrics_dataset_all = hdp.metric.compute_group_metrics(
            measures_all, thresholds_ref, flags.hw_def, start=(6, 1), end=(9, 1)
        )
    else:
        metrics_dataset_all = hdp.metric.compute_group_metrics(
            measures_all,
            thresholds_ref,
            flags.hw_def,
            use_doy=True,
            doy_mask=era_summer_mask,
        )

    metrics_all_land = ahelpers.process_heatwave_metrics(metrics_dataset_all)

    # ! uncomment to save output
    print("saving heatwave metrics for obs, 1960-2025")
    ahelpers.write_nc(
        metrics_all_land,
        hw_metrics_obs_path,
    )

# ! clean up for the next section to same some ram
# del metrics_all_land, metrics_dataset_all, measures_all


###########
# synthetic, meanshift
#############

# note that we do this on the shifted, doy-anomaly removed ts.

hw_metrics_meanshift_path = (
    processed_data_path / f"hw_metrics_{flags.ref_years[0]}_{flags.new_years[1]}_synth_anom{flags.label}.nc"
)

if hw_metrics_meanshift_path.exists():
    print("loading existing heatwave metrics for mean shift")
    metrics_synth_land = xr.open_dataset(hw_metrics_meanshift_path).chunk({"lat": 30, "lon": 30})
else:
    # time period 2,
    # era_land_new = era_land.sel(time=slice(str(flags.new_years[0]), str(flags.new_years[1])))
    era_land_ref_anom = era_land_all_anom.sel(time=slice(str(flags.ref_years[0] - 1), str(flags.ref_years[1])))
    era_land_new_anom = era_land_all_anom.sel(time=slice(str(flags.new_years[0] - 1), str(flags.new_years[1])))

    ## shifting the center for each grid cell.
    ## note: this is in anomaly space!! shifted, doy-anomaly removed.
    if flags.use_mean_shift:
        old_centers = era_land_ref_anom["t2m_x"].mean(dim=["time"])
        new_centers = era_land_new_anom["t2m_x"].mean(dim=["time"])
        # old_centers = era_land_ref["t2m_x"].mean(dim=["time"])
        # new_centers = era_land_new["t2m_x"].mean(dim=["time"])
    else:  # use median
        old_centers = era_land_ref_anom["t2m_x"].median(dim=["time"])
        new_centers = era_land_new_anom["t2m_x"].median(dim=["time"])

    # update the "time" coordinate in the future to pretend it's the "future"
    era_land_synth_new = (era_land_ref_anom - old_centers) + new_centers

    print("calculating heatwave metrics for mean shift")
    metrics_synth_land = ahelpers.get_synthetic_hw_metrics(
        era_land_ref_anom,
        era_land_synth_new,
        flags.new_years,
        thresholds_ref,
        flags.hw_def,
        use_calendar_summer=flags.use_calendar_summer,
    )

    # ! uncomment to save output
    ahelpers.write_nc(
        metrics_synth_land,
        hw_metrics_meanshift_path,
    )

# ! clean up for the next section to same some ram
# del era_land_synth_new, metrics_synth_land


###########
# synthetic, 1deg shift
# i.e. the "future" time period (1995-2025) is the same as 1960-1990 but shifted by 1deg
#############
hw_metrics_1deg_path = (
    processed_data_path / f"hw_metrics_{flags.ref_years[0]}_{flags.new_years[1]}_synth_1deg_anom{flags.label}.nc"
)

if hw_metrics_1deg_path.exists():
    print("loading existing heatwave metrics for 1deg shift")
    metrics_synth_land_1deg = xr.open_dataset(hw_metrics_1deg_path).chunk({"lat": 30, "lon": 30})
else:
    # update the "time" coordinate in the future to pretend it's the "future"
    # shifted by 1 degree.
    era_land_synth_new_1deg = era_land_ref_anom + 1

    metrics_synth_land_1deg = ahelpers.get_synthetic_hw_metrics(
        era_land_ref_anom,
        era_land_synth_new_1deg,
        flags.new_years,
        thresholds_ref,
        flags.hw_def,
        use_calendar_summer=flags.use_calendar_summer,
    )

    print("calculating heatwave metrics for 1deg shift")
    # ! uncomment to save output
    ahelpers.write_nc(
        metrics_synth_land_1deg,
        hw_metrics_1deg_path,
    )

# ! clean up for the next section to same some ram
# del era_land_synth_new_1deg, metrics_synth_land_1deg

###########
# synthetic, 2deg shift
#############

hw_metrics_2deg_path = (
    processed_data_path / f"hw_metrics_{flags.ref_years[0]}_{flags.new_years[1]}_synth_2deg_anom{flags.label}.nc"
)

if hw_metrics_2deg_path.exists():
    print("loading existing heatwave metrics for 2deg shift")
    metrics_synth_land_2deg = xr.open_dataset(hw_metrics_2deg_path).chunk({"lat": 30, "lon": 30})
else:
    # update the "time" coordinate in the future to pretend it's the "future"
    # shift by 2 degrees
    era_land_synth_new_2deg = era_land_ref_anom + 2
    metrics_synth_land_2deg = ahelpers.get_synthetic_hw_metrics(
        era_land_ref_anom,
        era_land_synth_new_2deg,
        flags.new_years,
        thresholds_ref,
        flags.hw_def,
        use_calendar_summer=flags.use_calendar_summer,
    )

    # ! uncomment to save output
    print("calculating heatwave metrics for 2deg shift")
    ahelpers.write_nc(
        metrics_synth_land_2deg,
        hw_metrics_2deg_path,
    )
