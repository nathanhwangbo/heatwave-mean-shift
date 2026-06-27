#############################
# the ERA data I downloaded on the full .25 degree grid
# Let's put this on a 1 degree grid, to match karen's ERA summer mask
# from vortex (the mckinnon group server):  modified from /kmckinnon/summer_extremes/summer_extremes/scripts/regrid_era5_1x1.py
################################################################3

import xesmf as xe
import xarray as xr
from glob import glob
import os
import numpy as np
from heatwave_mean_shift import flags
from pathlib import Path
from subprocess import check_call

if flags.use_1x1:
    era5_dir = Path(flags.era5_path)
    varnames = "t2m_x"

    lat1x1 = np.arange(-89.5, 90)
    lon1x1 = np.arange(0.5, 360)

    for varname in varnames:
        # files in original resolution
        files = glob(str(era5_dir / "t2m_x_daily" / "*.nc"))

        # # make 1x1 dir if not already present
        era5_dir_1x1 = era5_dir / "t2m_x_1x1"
        era5_dir_1x1 = "%s/%s/1x1" % (era5_dir, varname)
        cmd = "mkdir -p %s" % era5_dir_1x1
        check_call(cmd.split())

        for f in files:
            f_new = f.replace(".nc", "_1x1.nc").split("\\")[-1]
            f_new = "%s%s" % (era5_dir_1x1, f_new)

            if os.path.isfile(f_new):
                continue
            else:
                print(f)
                wgt_file = "%s/xe_weights_1x1.nc" % (era5_dir)
                if os.path.isfile(wgt_file):
                    reuse_weights = True
                else:
                    reuse_weights = False

                da = xr.open_dataarray(f)

                da = da.rename({"latitude": "lat", "longitude": "lon"})
                da = da.sortby("lat")

                regridder = xe.Regridder(
                    {"lat": da.lat, "lon": da.lon},
                    {"lat": lat1x1, "lon": lon1x1},
                    "bilinear",
                    periodic=True,
                    reuse_weights=reuse_weights,
                    filename=wgt_file,
                )

                da = regridder(da)
                da.to_netcdf(f_new)

    ## regrid landmask if needed
    # land_file = "/home/data/ERA5/fx/era5_lsmask.nc"
    # f_new = land_file.replace(".nc", "_1x1.nc")
    # if not os.path.isfile(f_new):
    #     da = xr.open_dataarray(land_file)
    #     da = da.rename({"latitude": "lat", "longitude": "lon"})
    #     da = da.sortby("lat")

    #     regridder = xe.Regridder(
    #         {"lat": da.lat, "lon": da.lon},
    #         {"lat": lat1x1, "lon": lon1x1},
    #         "bilinear",
    #         periodic=True,
    #         reuse_weights=reuse_weights,
    #         filename=wgt_file,
    #     )

    #     da = regridder(da)
    #     da.to_netcdf(f_new)
