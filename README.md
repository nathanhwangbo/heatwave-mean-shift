This repo is setup with `pixi` as the package manager. After installing [`pixi`](https://pixi.prefix.dev/latest/installation/) and cloning this repo, you can reproduce the analysis using the following steps:

0. open up `src/heatwave-mean-shift/flags.py`
    - This script defines different global variables. At minimum, the `era5_path` variable should be specified to a valid location on your files system. This is where ERA5 data will be downloaded. If your main goal is just to play around with the repo to get a sense of how it works, I recommend picking a much smaller set of years (e.g. 1960:1962 for `ref_years`, and 1963:1965 for `new_years`). 

1. in your terminal, in the root folder (i.e. the same folder as the `pyproject.toml`), run `pixi run dl-pipeline`
    - this downloads daily ERA tmax to the specified`era5_path`.
    - (this just runs `dl_data.py` and `regrid_era.py`, in order.)
2. Once that finished, run `pixi run processing-pipeline`
    - this processes the downloaded data and calculates the heatwave metrics
    - (this just runs `get_climatology.py`, `get_hw_metrics.py`, and `get_moments.py`, in order.)
3. And then `pixi run plotting-pipeline`
    - (this just runs `plot_hw_meanshift.py`, `plot_moments.py`, `plot_heatmaps.py`, and `plot_scatter.py`, in order.)

After all of that is done, you should have the main figures generated in the `figures` directory. 