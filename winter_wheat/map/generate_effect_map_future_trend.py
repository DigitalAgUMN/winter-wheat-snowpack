import os
from typing import NamedTuple
import multiprocessing

import pandas as pd

from winter_wheat.util import get_project_root, make_dirs


class Coefficient(NamedTuple):
    name: str
    value: float


def generate_effect_data(df_all, var_name, output_dir, output_filename, df_fips=None):
    scenarios = ["rcp45", "rcp85"]
    periods = {
        # "2021-2040": (2021, 2040),
        # "2040-2060": (2040, 2060),
        "2080-2100": (2080, 2100),
    }

    for scenario in scenarios:
        for period_name, period_year in periods.items():
            dataframe = df_all[(df_all["scenario"] == scenario) & (df_all["year"] >= period_year[0])
                               & (df_all["year"] <= period_year[1])]
            df_effect = dataframe.groupby(["county_fips"], as_index=False)[[var_name]].mean()
            # print(df_effect.head(10))
            df_effect = df_effect.reset_index()
            df_effect = df_effect.sort_values(by=['county_fips', ])
            if df_fips is not None:
                df_effect = pd.merge(df_fips, df_effect, on="county_fips", how="left")

            df_effect = df_effect[["county_fips", var_name, ]]

            # df_effect = df_effect[["county_fips", "effect"]]
            output_dir_each = os.path.join(output_dir, scenario)
            output_dir_each = os.path.join(output_dir_each, period_name)
            make_dirs(output_dir_each)
            df_effect.to_csv(os.path.join(output_dir_each, output_filename), index=False)


def generate_effect_data_rank(df_all, var_name, rank_percent, output_dir, output_filename, df_fips=None):
    scenarios = ["rcp45", "rcp85"]
    # periods = {
    #     "2021-2040": (2021, 2040),
    #     "2041-2070": (2041, 2070),
    #     "2071-2100": (2071, 2100),
    # }
    periods = {
        "2040-2060": (2040, 2060),
        "2080-2100": (2080, 2100),
    }

    for scenario in scenarios:
        for period_name, period_year in periods.items():
            rank = round((period_year[1] - period_year[0] + 1) * rank_percent)
            # print(rank,  period_year[0],  period_year[1])
            dataframe = df_all[(df_all["scenario"] == scenario) & (df_all["year"] >= period_year[0])
                               & (df_all["year"] <= period_year[1])]

            df_effect = dataframe.copy()
            df_effect["rank"] = df_effect.groupby("county_fips")[[var_name]].rank("first", ascending=False)
            df_effect = df_effect[df_effect["rank"] == rank]
            # print(df_effect.head(10))
            df_effect = df_effect.reset_index()
            df_effect = df_effect.sort_values(by=['county_fips', ])
            if df_fips is not None:
                df_effect = pd.merge(df_fips, df_effect, on="county_fips", how="left")
                # df_effect = df_effect.astype({"year": int, })  # ValueError: Cannot convert non-finite values (NA or inf) to integer

            df_effect = df_effect[["county_fips", var_name, "year"]]

            output_dir_each = os.path.join(output_dir, scenario)
            output_dir_each = os.path.join(output_dir_each, period_name)
            make_dirs(output_dir_each)

            df_effect.to_csv(os.path.join(output_dir_each, output_filename), index=False)


def make_effect_avg(df, output_dir, df_fips=None):
    variables = [
        "tasmin_fall", "tasmax_fall", "tasavg_fall", "prcp_fall", "snow_depth_mm_fall",
        "tasmin_winter", "tasmax_winter", "tasavg_winter", "prcp_winter", "snow_depth_mm_winter",
        "tasmin_spring", "tasmax_spring", "tasavg_spring", "prcp_spring", "snow_depth_mm_spring",
        "tasmin", "tasmax", "tasavg", "pr", "snow_depth_mm",
    ]
    for variable in variables:
        generate_effect_data(df_all=df,
                             var_name=variable,
                             output_dir=output_dir,
                             output_filename="gridMET_{}_avg.csv".format(variable),
                             df_fips=df_fips)


def make_effect_rank(df, output_dir, df_fips=None):
    variables = [
        "tasmin_fall", "tasmax_fall", "tasavg_fall", "prcp_fall", "snow_depth_mm_fall",
        "tasmin_winter", "tasmax_winter", "tasavg_winter", "prcp_winter", "snow_depth_mm_winter",
        "tasmin_spring", "tasmax_spring", "tasavg_spring", "prcp_spring", "snow_depth_mm_spring",
        "tasmin", "tasmax", "tasavg", "pr", "snow_depth_mm",
    ]
    for variable in variables:
        generate_effect_data_rank(df_all=df,
                                  var_name=variable,
                                  rank_percent=.1,
                                  output_dir=output_dir,
                                  output_filename="gridMET_{}_top10p.csv".format(variable),
                                  df_fips=df_fips)


def process(model_name):
    base_dir = os.path.join("N:/WinterWheat/model-input-swe-bc", model_name)
    output_dir = get_project_root() / "output/county_map/v20200726-gdd-a"  # all states
    output_dir = get_project_root() / "output/county_map/v20200730"  # all states + all missing counties
    output_dir = get_project_root() / "output/county_map/v20200814-future-trend"  # all states + all missing counties including variables not only effect
    output_dir = get_project_root() / "output/county_map/v20201014-future-trend"
    output_dir = "N:/WinterWheat/county_map/v20210129-future-trend"

    output_dir = os.path.join(output_dir, model_name)
    data_filename = os.path.join(base_dir, "winter_wheat_climate_trend_sdepth_bc_v20200814_{}.csv".format(model_name))
    data_filename = os.path.join(base_dir, "winter_wheat_climate_trend_sdepth_bc_v20210129_{}.csv".format(model_name))

    if not os.path.exists(data_filename):
        print("{} is not exist.".format(data_filename))
        return

    df = pd.read_csv(data_filename)
    df_fips = pd.read_csv(get_project_root() / "input/county_fips_3107.csv")
    make_effect_avg(df, output_dir, df_fips=df_fips)
    make_effect_rank(df, output_dir, df_fips=df_fips)


def main():
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
    pool = multiprocessing.Pool(3)
    results = []
    # print(models_for_mp)
    for model in models:
        results.append(pool.apply_async(process, args=(model,)))
    for i, result in enumerate(results):
        result.get()
        print("Result: Model (idx={}) is processed.".format(i, ))


if __name__ == "__main__":
    main()
