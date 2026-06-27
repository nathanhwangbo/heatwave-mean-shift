from heatwave_mean_shift import flags
from pathlib import Path
import cdsapi

# make a target directory to save downloaded data
target_dir = Path(flags.era5_path) / "t2m_x_daily"
target_dir.mkdir(parents=True, exist_ok=True)

dataset = "derived-era5-single-levels-daily-statistics"
first_year = flags.ref_years[0]  # typically 1950
last_year = flags.new_years[1]  # typically 2025
for year in range(first_year, last_year + 1):
    print(f"working on year {year}")
    request = {
        "product_type": "reanalysis",
        "variable": ["2m_temperature"],
        "year": str(year),
        "month": [
            "01",
            "02",
            "03",
            "04",
            "05",
            "06",
            "07",
            "08",
            "09",
            "10",
            "11",
            "12",
        ],
        "day": [
            "01",
            "02",
            "03",
            "04",
            "05",
            "06",
            "07",
            "08",
            "09",
            "10",
            "11",
            "12",
            "13",
            "14",
            "15",
            "16",
            "17",
            "18",
            "19",
            "20",
            "21",
            "22",
            "23",
            "24",
            "25",
            "26",
            "27",
            "28",
            "29",
            "30",
            "31",
        ],
        "daily_statistic": "daily_maximum",
        "time_zone": "utc+00:00",
        "frequency": "1_hourly",
        "grid": [1.0, 1.0],
    }

    client = cdsapi.Client()
    client.retrieve(dataset, request).download(target=target_dir / f"t2m_x_daily_{year}.nc")
