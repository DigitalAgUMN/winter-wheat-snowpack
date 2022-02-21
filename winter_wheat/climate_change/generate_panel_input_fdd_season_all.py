import os
import multiprocessing
import datetime
import sys

import pandas as pd
import numpy as np
import tqdm

from winter_wheat.util import get_project_root, make_dirs


hourly_interpolation = np.sin(np.linspace(-np.pi / 2, np.pi / 2, num=24)) + 1


def get_fdd(tmin, tmax, criteria=0):
    # tsa_kk = tmin + hourly_interpolation * (tmax - tmin) / 2 - criteria
    # return -tsa_kk[tsa_kk < 0].sum() / 24
    tsa_kk = criteria - (tmin + hourly_interpolation * (tmax - tmin) / 2)
    return tsa_kk[tsa_kk > 0].sum() / 24


def append_season_year(df):
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df.loc[df['month'] < 9, 'season_year_offset'] = 0
    df.loc[df['month'] >= 9, 'season_year_offset'] = 1
    df["season_year"] = df["year"] + df["season_year_offset"]
    df["season_year"] = df["season_year"].astype('int')
    return df


def generate_panel_output(output_filename, df_all):
    df_all = append_season_year(df_all)
    # df_all = df_all[((df_all["month"] >= 9) | (df_all["month"] < 6)) & (df_all["year"] >= 1950)]
    df_all = df_all.copy()
    # print(df_all.head())
    # print(df_all.columns)
    # df_all["snow_depth_cm"] = df_all.apply(lambda x: x.snow_depth_mm / 10.0, axis=1)
    df_all["snow_depth_cm"] = df_all["snow_depth_mm"] / 10.0
    # tsa_kk = df_all['tasmin'] + hourly_interpolation * (df_all['tasmax'] - df_all['tasmin']) / 2
    # print("starting fdd1")
    df_all["fdd1"] = df_all.apply(lambda x: get_fdd(x.tasmin, x.tasmax, criteria=0), axis=1)
    df_all["fdd1_fall"] = np.where(((df_all["month"] >= 9) & (df_all["month"] < 12)), df_all[
        "fdd1"], 0)
    df_all["fdd1_winter"] = np.where(((df_all["month"] >= 12) | (df_all["month"] < 3)), df_all[
        "fdd1"], 0)
    df_all["fdd1_spring"] = np.where(((df_all["month"] >= 3) & (df_all["month"] < 6)), df_all[
        "fdd1"], 0)

    df_all = df_all.groupby(["scenario", "county_fips", "season_year"]).agg({
        'fdd1': 'sum',
        'fdd1_fall': 'sum',
        'fdd1_winter': 'sum',
        'fdd1_spring': 'sum',
    })
    # print(df_all.columns)
    # print(df_all.dtypes)
    # print(df_all.head())

    df_all = df_all[
        [
            "fdd1",
            "fdd1_fall",
            "fdd1_winter",
            "fdd1_spring",
        ]
    ]
    df_all = df_all.reset_index()
    # mask = df_all['season_year'].isin([1950, 2006])
    # df_all = df_all[~mask]
    # print(df_all.head())
    df_all.to_csv(output_filename, index=False)


def read_nex_gddp(filename):
    df_nex = pd.read_csv(filename)
    df_nex[['scenario', 'model', "date", "emsemble", "ee_id"]] = df_nex["system:index"].str.split("_", expand=True)
    df_nex["date"] = pd.to_datetime(df_nex["date"], format='%Y%m%d')
    df_nex["tasmax"] = df_nex["tasmax"] - 273.15
    df_nex["tasmin"] = df_nex["tasmin"] - 273.15
    df_nex["tasavg"] = (df_nex["tasmin"] + df_nex["tasmax"]) / 2
    df_nex["pr"] = df_nex["pr"] * 86400
    df_nex = df_nex.rename(columns={"GEOID": "county_fips", })
    # print(df_nex[['scenario', 'model', "date", "emsemble", "ee_id"]].head())
    return df_nex[["county_fips", "scenario", "date", "tasmin", "tasmax", "tasavg", "pr"]]


def process_annual_fdd(weather_filename, swe_filename, output_filename, other_weather_file=None):
    df_nex = read_nex_gddp(weather_filename)
    if other_weather_file is not None:
        df_nex_other = read_nex_gddp(other_weather_file)
        df_nex = pd.concat([df_nex, df_nex_other], ignore_index=True)

    df_swe = pd.read_csv(swe_filename, parse_dates=['date'])
    df_swe["snow_depth_mm"] = df_swe["snw"] / 0.274
    df_swe = df_swe.rename(columns={"FIPS": "county_fips", })

    df_all = pd.merge(df_nex, df_swe, on=["scenario", "date", "county_fips"],
                      how='inner')

    generate_panel_output(output_filename, df_all)


def process(model_name):
    base_drive = "N:/WinterWheat"
    if sys.platform == "linux" or sys.platform == "linux2":  # MSI
        base_drive = "/home/jinzn/taegon/WinterWheat"
    # output_dir = get_project_root() / "output/panel_input_future_all"
    output_dir = os.path.join(os.path.join(base_drive, "NEX-GDDP-annual"), model_name)
    make_dirs(output_dir)
    base_dir_swe = os.path.join(os.path.join(base_drive, "CMIP5_snw_county"), model_name)
    base_dir_nex = os.path.join(os.path.join(base_drive, "NEX-GDDP"), model_name)
    cmc_filename = os.path.join(base_drive, "CMC_snw_county/CMC_snw_county.csv")

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

    number_of_process = 4
    if len(sys.argv) >= 2:
        number_of_process = int(sys.argv[1])

    pool = multiprocessing.Pool(number_of_process)
    results = []

    for scenario in scenarios:
        for year in range(periods[scenario][0], periods[scenario][1] + 1):
            weather_filename = os.path.join(base_dir_nex,
                                            "NEX-GDDP-all-w25000-{}-{}-{}.csv".format(model_name, scenario, year))
            weather_other_filename = os.path.join(base_dir_nex,
                                            "NEX-GDDP-all-other-w25000-{}-{}-{}.csv".format(model_name, scenario, year))
            swe_filename = os.path.join(base_dir_swe,
                                            "df-snw-{}-{}-{}.csv".format(model_name, scenario, year))

            output_filename = os.path.join(output_dir,
                                           "climate_input_data_fdd_season_{}_{}_{}_20210109.csv".format(model_name,
                                                                                                        scenario,
                                                                                                        year))
            if not os.path.exists(weather_filename):
                continue
            if not os.path.exists(weather_other_filename):
                weather_other_filename = None
            if os.path.exists(output_filename):
                continue
            # process_annual_gdd(base_dir_nex, model_name, output_dir, scenario, year)
            results.append(pool.apply_async(process_annual_fdd, args=(weather_filename, swe_filename, output_filename,weather_other_filename)))

    for i, result in enumerate(tqdm.tqdm(results)):
        result.get()
        # print("Result: Model (idx={}) is processed.".format(i, ))


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

    # for model in models:
    #     process(model)
    if len(sys.argv) >= 3:
        model_idx = int(sys.argv[2])
        process(models[model_idx])
    else:
        for model in models:
            process(model)


if __name__ == "__main__":
    main()
