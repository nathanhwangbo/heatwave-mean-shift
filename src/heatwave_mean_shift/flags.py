"""
Define the following parameters, which will be used downstream by all of the analysis scripts

- use_1x1: boolean
    - true uses 1x1 degree regridded data (the regridding is done in 01_process_era.py)
    - false uses the native ERA5 grid (0.25x0.25 degree)

- use_ref_years_for_climatology: boolean
    - true uses the ref_years to calculate the day-of-year climatology (i.e the day-of-year means are calculated using only the ref_years)
    - false uses the entire time period (1960-2025) to calculate the day-of-year climatology. This is the option we went with in the main text

- use_calendar_summer: boolean
    - true uses JJA as summer in the northern hemisphere and DJF in southern
    - false uses 90 days surrounding hottest climatological day, ala mckinnon et al 2024

- ref_years
    - list of 2 years.
    - period used to define heatwave metrics in the "base period"
    - tested option: [1960, 1990]

- new_years
    - list of 2 years.
    - gap between the two years should probably match ref_years
    - tested option: [1995, 2025]

- percentile_threshold
    - threshold used to define extreme heat at each gridcell
    - tested options: 0.9, 0.95

- hw_def:
    - heatwave definitions, as in the HDP package.
    - tested options: "3-0-0", "2-0-0", "6-0-0"
    - eg. 3-0-0 means a heatwave is defined as 3+ consecutive days of extreme heat

- era5_path: string
    - where to save downloaded ERA5 data to (used in 00_get_era.py and 01_process_era.py, and read from in 02_era_climatology.py and 2a_get_moments.py)
    - NOTE: this path will house daily tmax data in the original resolution AND 1x1 regridded data (if use_1x1 is True)
    - so.. this will eat up a lot of hard drive space!
- label:
    - string
    - used to label your choice of flags! Will show up in saved file names.


note: if you try an option and it doesn't work, look at modifying 0_era_meanshift.py

note: if you only care about `percentile_threshold` and the `hw_def` args, it might be faster for you to use the built-in vectorization in the HDP package
    - i.e. you don't need to pass in one config at a time to the hdp functions.
    - (this is why the hw_def has an extra set of brackets)

"""


###################
# analysis flags
####################

## this is the config for the main analysis in the paper --------------------------------
label = "q90_300"
use_1x1 = False  # if true, use 1x1 degree regridded data. else use native ERA5 grid
use_calendar_summer = True  # if true, use JJA as summer. else use dayofyear mask
use_ref_years_for_climatology = False  # if false, use all years to calculate day-of-year climatology
ref_years = [1960, 1990]  # the time period the thresholds are calculated over
new_years = [1995, 2025]  # the time period we're gonna compare to
percentile_threshold = 0.9
hw_def = [[3, 0, 0]]
use_mean_shift = True  # if false, use the median
era5_path = "/home/nhwangbo/data_to_organize_later/ERA5"  # where we want to save downloaded data to

##################################################
# examples of other configs
# these are in the supplement as sensitivity tests
##################################################

## sensitivity to mean shift vs median shift
# label = "q90_300_median"
# use_1x1 = True
# use_calendar_summer = True  # if true, use JJA as summer. else use dayofyear mask
# ref_years = [1960, 1990]  # the time period the thresholds are calculated over
# new_years = [1995, 2025]  # the time period we're gonna compare to
# percentile_threshold = 0.9
# hw_def = [[3, 0, 0]]
# use_mean_shift = False  # if false, use the median # CHANGED (compared to the main config)
# era5_path = "/home/nhwangbo/data_to_organize_later/ERA5"

## sensitivity to summer definition --------------------
# label = "doy_q90_300"
# use_1x1 = True
# use_calendar_summer = False  # CHANGED
# ref_years = [1960, 1990]
# new_years = [
#     1995 + 1,
#     2025,
# ]  # CHANGED: + 1 accounts for calendar year wrapping in my modified hdp functions
# percentile_threshold = 0.9
# hw_def = [[3, 0, 0]]
# use_mean_shift = True
# era5_path = "/home/nhwangbo/data_to_organize_later/ERA5"

## sensitivity to "extreme heat" definition (i.e. quantile) --------------------------------
# label = "q95_300"
# use_1x1 = True
# use_calendar_summer = True
# ref_years = [1960, 1990]
# new_years = [1995, 2025]
# percentile_threshold = 0.95  # CHANGED
# hw_def = [[3, 0, 0]]
# use_mean_shift = True
# era5_path = "/home/nhwangbo/data_to_organize_later/ERA5"

## sensitivity to heatwave definition --------------------------------
# label = "q90_200"
# use_1x1 = True
# use_calendar_summer = True
# ref_years = [1960, 1990]
# new_years = [1995, 2025]
# percentile_threshold = 0.90
# hw_def = [[2, 0, 0]]  # CHANGED
# use_mean_shift = True
# era5_path = "/home/nhwangbo/data_to_organize_later/ERA5"

## sensitivity to heatwave definition --------------------------------
# label = "q90_500"
# use_1x1 = True
# use_calendar_summer = True
# ref_years = [1960, 1990]
# new_years = [1995, 2025]
# percentile_threshold = 0.90
# hw_def = [[5, 0, 0]]  # CHANGED
# use_mean_shift = True
# era5_path = "/home/nhwangbo/data_to_organize_later/ERA5"

## sensitivity to using skewness/variance from the reference period, instead of the whole period --------------------------------
# label = "q90_300_ref_climatology"
# use_1x1 = False  # if true, use 1x1 degree regridded data. else use native ERA5 grid
# use_calendar_summer = True  # if true, use JJA as summer. else use dayofyear mask
# use_ref_years_for_climatology = True  # CHANGED
# ref_years = [1960, 1990]  # the time period the thresholds are calculated over
# new_years = [1995, 2025]  # the time period we're gonna compare to
# percentile_threshold = 0.9
# hw_def = [[3, 0, 0]]
# use_mean_shift = True  # if false, use the median
# era5_path = "/home/nhwangbo/data_to_organize_later/ERA5"


# print out all of the flags whenever this module is imported (so always)
if __name__ != "__main__":  # Only print when imported, not when run directly
    print("=" * 60)
    print("ANALYSIS FLAGS")
    print("=" * 60)
    print(f"label: {label}")
    print(f"use_1x1: {use_1x1}")
    print(f"use_calendar_summer: {use_calendar_summer}")
    print(f"use_ref_years_for_climatology: {use_ref_years_for_climatology}")
    print(f"ref_years: {ref_years}")
    print(f"new_years: {new_years}")
    print(f"percentile_threshold: {percentile_threshold}")
    print(f"hw_def: {hw_def}")
    print(f"use_mean_shift: {use_mean_shift}")
    print("=" * 60)
