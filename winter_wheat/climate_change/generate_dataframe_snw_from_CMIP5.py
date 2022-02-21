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


def process_model(model_name: str, base_dir: str, output_dir: str) -> None:
    output_dir = os.path.join(output_dir, model_name)
    make_dirs(output_dir)
    scenarios = ["historical", "rcp45", "rcp85"]
    # periods = {
    #     "historical": (1951, 2005),
    #     "rcp45": (2007, 2100),
    #     "rcp85": (2007, 2100),
    # }
    periods = {
        "historical": (1980, 2005),
        "rcp45": (2080, 2100),
        "rcp85": (2080, 2100),
    }

    fips_array = xr.open_rasterio(os.path.join(get_project_root() / "input/raster", "county_FIPS_0.125deg.tif"))
    fips_array = fips_array.to_masked_array()[0]
    fips_array_flatten = fips_array.flatten()
    # print(fips_array.shape)
    # return
    # np.savetxt(os.path.join(get_project_root() / "output/cc_swe_all", "test_fips_us.csv"), fips_array,
    #            delimiter=",")

    for scenario in scenarios:
        prev_year_dataset = None
        for year in tqdm.tqdm(range(periods[scenario][0], periods[scenario][1] + 1), desc="Processing {} - {}".format(model_name, scenario)):

            output_filename = os.path.join(output_dir, "df-snw-{}-{}-{}.csv".format(model_name, scenario, year))
            if os.path.exists(output_filename):
                prev_year_dataset = None
                continue
            df_model = None

            leap_mode = False
            if model_name in ["CNRM-CM5", "MPI-ESM-MR", "MIROC-ESM", "MRI-CGCM3"]:
                start = datetime.datetime.strptime("{}0801".format(year-1), "%Y%m%d")
                end = datetime.datetime.strptime("{}0701".format(year), "%Y%m%d")
                # assert ((end - start).days + 1) == ds.shape[0]
                leap_mode = True
            else:
                start = DatetimeNoLeap(year - 1, 8, 1)
                end = DatetimeNoLeap(year, 7, 1)
            date_generated = [start + datetime.timedelta(days=x) for x in range(0, (end - start).days)]

            if prev_year_dataset is None:
                prev_year_dataset = read_npy(base_dir, model_name, scenario, year - 1)
            else:
                prev_year_dataset = current_year_dataset
            current_year_dataset = read_npy(base_dir, model_name, scenario, year)
            # print(current_year_dataset.shape)
            # np.savetxt(os.path.join(get_project_root() / "output/cc_swe_all", "test_snw_us.csv"), current_year_dataset[100],
            #            delimiter=",")

            for i, date in enumerate(date_generated):
                date_str = date.strftime("%Y%m%d")
                working_year = date.timetuple().tm_year
                doy = date.timetuple().tm_yday

                if working_year == year:
                    today_snw_grid = current_year_dataset[doy - 1]
                else:
                    today_snw_grid = prev_year_dataset[doy - 1]
                if model_name == "inmcm4" or model_name == "GFDL-ESM2G":
                    today_snw_grid = np.where((today_snw_grid > -1) & (today_snw_grid < 0), 0, today_snw_grid)
                elif model_name == "CNRM-CM5" or model_name == "MIROC-ESM":
                    today_snw_grid = np.where(today_snw_grid == -9999, np.nan, today_snw_grid)
                elif model_name == "GFDL-CM3":
                    today_snw_grid = np.where((today_snw_grid > -5) & (today_snw_grid < 0), 0, today_snw_grid)

                df_day_snw = pd.DataFrame({"FIPS": fips_array_flatten, "snw": today_snw_grid.flatten()})
                # print(df_day_snw.head())
                df_day_snw_county = df_day_snw.groupby(["FIPS"]).mean()
                df_day_snw_county = df_day_snw_county.reset_index()
                df_day_snw_county["model"] = model_name
                df_day_snw_county["scenario"] = scenario
                df_day_snw_county["date"] = pd.to_datetime(date_str, format='%Y%m%d')
                df_day_snw_county = df_day_snw_county[df_day_snw_county.FIPS != 65535]
                # df_day_snw_county["date"] = df_day_snw_county["date"].dt.date
                if df_model is None:
                    df_model = df_day_snw_county
                else:
                    df_model = pd.concat([df_model, df_day_snw_county], ignore_index=True)

                if np.min(today_snw_grid) < 0:
                    df_day_snw.to_csv(os.path.join(get_project_root() / "output/cc_swe_all", "test_dataframe_raw_{}_{}_{}.csv".format(model_name, scenario, date_str)))
                    df_day_snw_county = df_day_snw.groupby(["FIPS"]).mean()
                    df_day_snw_county = df_day_snw_county.reset_index()
                    df_day_snw_county.to_csv(os.path.join(get_project_root() / "output/cc_swe_all", "test_dataframe_groupby_{}_{}_{}.csv".format(model_name, scenario, date_str)))
            df_model.to_csv(output_filename, index=False)


    #         df = pd.DataFrame(list(zip(dates, snw.flatten())),
    #                           columns=['date', 'snw'])
    #         # print(df.head())
    #         df["model"] = model_name
    #         df["scenario"] = scenario
    #         df["date"] = df["date"].dt.date
    #         df["county"] = county_name.split("_")[0]
    #         df["state"] = county_name.split("_")[1]
    #
    #         if df_model is None:
    #             df_model = df
    #         else:
    #             df_model = pd.concat([df_model, df], ignore_index=True)
    #
    # # print(df_model.head())
    # df_model = df_model.drop_duplicates(subset=["scenario", 'date', 'county', "state"])
    # df_model = df_model[["model", "scenario", "date", "county", "state", "snw", ]]
    # df_model.to_csv(os.path.join(output_dir, "{}.csv".format(model_name)), index=False)


def main(base_dir: str, output_dir: str, selected_model_idx: int = -1) -> None:
    models = [
        # Phase 1: Using 3 GCMs in first step
        "CanESM2",
        "CSIRO-Mk3-6-0",
        "inmcm4",

        # Phase 2: Add remained GCMs
        "CNRM-CM5",
        # "GFDL-CM3",  # incomplete dataset (GEE has no 2097-2100, RCP45)
        "GFDL-ESM2G",
        "MIROC-ESM",
        # "MIROC-ESM-CHEM",
        # "MPI-ESM-LR",
        "MPI-ESM-MR",
        "MRI-CGCM3",
        "NorESM1-M",

        # This GCMs are ignored because of different time
        # "GISS-E2-H", This model use 360 days, so we decide to ignore it.
        # "GISS-E2-R",
    ]

    if selected_model_idx < -1 or selected_model_idx >= len(models):
        print("selected_model cannot run. please check model idx.")
        print("it should be -1 or from 0 to less than {}".format(len(models)))
        return

    if selected_model_idx == -1:
        pool = multiprocessing.Pool(4)
        results = []
        # print(models_for_mp)
        for model in models:
            base_model_dir = os.path.join(base_dir, model)
            results.append(pool.apply_async(process_model, args=(model, base_model_dir, output_dir)))
        for i, result in enumerate(results):
            result.get()
            print("Result: Model (idx={}) is processed.".format(i, ))
    else:
        base_model_dir = os.path.join(base_dir, models[selected_model_idx])
        process_model(model_name=models[selected_model_idx],
                      base_dir=base_model_dir,
                      output_dir=output_dir)


def merge(base_dir: str) -> None:
    models = [
        # Phase 1: Using 3 GCMs in first step
        "CanESM2",
        "CSIRO-Mk3-6-0",
        "inmcm4",

        # Phase 2: Add remained GCMs
        "CNRM-CM5",
        # "GFDL-CM3",  # incomplete dataset (GEE has no 2097-2100, RCP45)
        "GFDL-ESM2G",
        "MIROC-ESM",
        # "MIROC-ESM-CHEM",
        # "MPI-ESM-LR",
        "MPI-ESM-MR",
        "MRI-CGCM3",
        "NorESM1-M",
        #
        # This GCMs are ignored because of different time
        # "GISS-E2-H", This model use 360 days, so we decide to ignore it.
        # "GISS-E2-R",
    ]

    df_all = None
    for model_name in tqdm.tqdm(models):
        filename = base_dir, "{}.csv".format(model_name)
        if os.path.exists(filename):
            df = pd.read_csv(os.path.join())
            if df_all is None:
                df_all = df
            else:
                df_all = pd.concat([df_all, df], ignore_index=True)
    df_all.to_csv(os.path.join(base_dir, "cc_swe.csv"), index=False)


def test_datetime():
    start = DatetimeNoLeap(2004, 2, 1)
    end = DatetimeNoLeap(2004, 3, 2)
    date_generated = [start + datetime.timedelta(days=x) for x in range(0, (end - start).days)]
    assert len(date_generated) == 29
    # print(date_generated)
    # print(date_generated[-2].strftime("%Y%m%d"))
    assert date_generated[-2].strftime("%Y%m%d") == "20040228"
    # print(date_generated[-1].timetuple())
    assert date_generated[-1].timetuple().tm_yday == 31 + 28 + 1

    start = datetime.datetime.strptime("{}/02/01".format(2004), "%Y/%m/%d")
    end = datetime.datetime.strptime("{}/03/02".format(2004), "%Y/%m/%d")
    date_generated = [start + datetime.timedelta(days=x) for x in range(0, (end - start).days)]
    assert len(date_generated) == 30
    # print(date_generated)
    # print(date_generated[-2].strftime("%Y%m%d"))
    assert date_generated[-2].strftime("%Y%m%d") == "20040229"
    assert date_generated[-1].timetuple().tm_yday == 31 + 29 + 1


if __name__ == "__main__":
    # base_dir = "/mnt/n/CMIP5/snw"
    # output_dir = "/mnt/n/CMIP5/snw_points"

    base_dir = "N:/CMIP5/snw_daily_npy_0125"
    output_dir = "N:/WinterWheat/CMIP5_snw_county_shift"
    make_dirs(output_dir)

    test_datetime()
    main(base_dir, output_dir, selected_model_idx=-1)

    # merge(output_dir)
