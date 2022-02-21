import os
import re
import multiprocessing
from typing import NamedTuple, Tuple

import tqdm
from netCDF4 import Dataset, num2date
import numpy as np
import pandas as pd
from cftime import DatetimeNoLeap
import datetime
import xarray as xr

from winter_wheat.climate_change.feature_point import get_selected_points
from winter_wheat.util import make_dirs, get_project_root


def check_duplicated(model_name: str, base_dir: str) -> None:
    df = pd.read_csv(os.path.join(base_dir, "{}.csv".format(model_name)))
    print(df.head())
    # Select duplicate rows except last occurrence based on all columns
    # df = df.drop_duplicates(subset=['date', 'county', "state"])
    duplicateRowsDF = df[df.duplicated(subset=["scenario", 'date', 'county', "state"],)]
    print("Duplicate Rows except last occurrence based on all columns are :")
    print(duplicateRowsDF)
    # check for CanESM2, it has two historical dataset, 1850-2005, 1979-2005
    print(df[(df['date'] >= "1979-01-01") & (df['date'] <= '1979-01-02')])


def read_npy(base_dir: str, model_name: str, scenario: str, year: int) -> np.array:
    return np.load(os.path.join(base_dir, "snw_{}_{}_{}.npy".format(model_name, scenario, year)))


def process_model(base_dir: str, output_dir: str) -> None:
    fips_array = xr.open_rasterio(os.path.join(get_project_root() / "input/raster", "county_FIPS_0.125deg.tif"))
    fips_array = fips_array.to_masked_array()[0]
    fips_array_flatten = fips_array.flatten()
    # print(fips_array.shape)
    # return
    # np.savetxt(os.path.join(get_project_root() / "output/cc_swe_all", "test_fips_us.csv"), fips_array,
    #            delimiter=",")
    df_model = None

    for year in tqdm.tqdm(range(1999, 2020)):
        start = datetime.datetime.strptime("{}/08/01".format(year-1), "%Y/%m/%d")
        end = datetime.datetime.strptime("{}/07/01".format(year), "%Y/%m/%d")
        date_generated = [start + datetime.timedelta(days=x) for x in range(0, (end - start).days)]
        # print(date_generated)

        for i, date in enumerate(date_generated):
            date_str = date.strftime("%Y%m%d")

            today_snw_grid = np.load(os.path.join(base_dir, "cmc_us_sdepth_dly_{}.npy".format(date_str)))
            # print(fips_array.shape)
            # print(today_snw_grid.shape)
            # print(today_snw_grid)
            today_snw_grid = np.where((today_snw_grid > -1) & (today_snw_grid < 0), 0, today_snw_grid)

            # Downscaling for matching FIPS map
            # np.repeat is faster than np.kron about 10x
            # today_snw_grid = np.kron(today_snw_grid, [[1, 1], [1, 1]])
            today_snw_grid = np.repeat(np.repeat(today_snw_grid, 2, axis=0), 2, axis=1)
            assert today_snw_grid.flatten().shape == fips_array_flatten.shape

            df_day_snw = pd.DataFrame({"county_fips": fips_array_flatten, "snow_depth_cm": today_snw_grid.flatten()})
            # print(df_day_snw.head())
            df_day_snw_county = df_day_snw.groupby(["county_fips"]).mean()
            df_day_snw_county = df_day_snw_county.reset_index()
            df_day_snw_county["date"] = pd.to_datetime(date_str, format='%Y%m%d')
            df_day_snw_county = df_day_snw_county[df_day_snw_county.county_fips != 65535]
            # df_day_snw_county["date"] = df_day_snw_county["date"].dt.date
            if df_model is None:
                df_model = df_day_snw_county
            else:
                df_model = pd.concat([df_model, df_day_snw_county])

            if np.min(today_snw_grid) < 0:
                print("There is negative value. Check file.")
                print(os.path.join(output_dir, "test_dataframe_raw_CMC_dly_{}.csv".format(date_str)))
                df_day_snw.to_csv(os.path.join(output_dir, "test_dataframe_raw_CMC_dly_{}.csv".format(date_str)))
                df_day_snw_county = df_day_snw.groupby(["FIPS"]).mean()
                df_day_snw_county = df_day_snw_county.reset_index()
                df_day_snw_county.to_csv(os.path.join(output_dir, "test_dataframe_groupby_CMC_dly_{}.csv".format(date_str)))
    df_model.to_csv(os.path.join(output_dir, "CMC_snw_county.csv"), index=False)


def main(base_dir: str, output_dir: str) -> None:
        process_model(base_dir=base_dir,
                      output_dir=output_dir)


if __name__ == "__main__":
    # base_dir = "/mnt/n/CMIP5/snw"
    # output_dir = "/mnt/n/CMIP5/snw_points"
    base_dir = "N:/WinterWheat/CMC_US_daily_npy"
    output_dir = "N:/WinterWheat/CMC_snw_county_shift"
    make_dirs(output_dir)

    main(base_dir, output_dir)

    # merge(output_dir)
