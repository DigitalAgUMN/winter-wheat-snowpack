import os
import multiprocessing
import datetime
import sys

import pandas as pd
import numpy as np
import tqdm

from winter_wheat.util import get_project_root, make_dirs


hourly_interpolation = np.sin(np.linspace(-np.pi / 2, np.pi / 2, num=24)) + 1


def get_igdd(tmin, tmax, criteria=0):
    tsa_kk = tmin + hourly_interpolation * (tmax - tmin) / 2 - criteria
    return tsa_kk[tsa_kk > 0].sum() / 24


def partition_rain(pr, tavg):
    t_rain = 3.0
    t_snow = -1.0

    pr_fraction = (t_rain - tavg) / (t_rain - t_snow)
    pr_fraction = 0 if pr_fraction < 0 else pr_fraction
    pr_fraction = 1 if pr_fraction > 1 else pr_fraction
    return pr * (1 - pr_fraction)


def append_season_year(df):
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df.loc[df['month'] < 9, 'season_year_offset'] = 0
    df.loc[df['month'] >= 9, 'season_year_offset'] = 1
    df["season_year"] = df["year"] + df["season_year_offset"]
    df["season_year"] = df["season_year"].astype('int')
    df['season'] = "None"
    df.loc[(df['month'] >= 9) & (df['month'] < 12), 'season'] = "fall"
    df.loc[(df['month'] >= 12) | (df['month'] < 3), 'season'] = "winter"
    df.loc[(df['month'] >= 3) & (df['month'] < 6), 'season'] = "spring"
    df = df[(df['month'] <= 5) | (df['month'] >= 9)]

    return df


def generate_panel_output(output_filename, df_all):
    df_all = append_season_year(df_all)
    # df_all = df_all[((df_all["month"] >= 9) | (df_all["month"] < 6)) & (df_all["year"] >= 1950)]
    df_all = df_all.copy()
    # print(df_all.head())
    # print(df_all.columns)
    # tsa_kk = df_all['tasmin'] + hourly_interpolation * (df_all['tasmax'] - df_all['tasmin']) / 2
    # print("starting fdd1")

    criteria_lmh = {
        "fall": [0, 10, 17],
        "winter": [0, 5, 10],
        "spring": [0, 18, 30],
    }

    criteria_lmh_avg = {
        "fall": [0, 18, 23],
        "winter": [0, 5, 10],
        "spring": [0, 17, 28],
    }

    df_all["rainfall"] = df_all.apply(lambda x: partition_rain(x.pr, x.tasavg), axis=1)
    df_all["snowfall"] = df_all["pr"] - df_all["rainfall"]

    for season_name in ("fall", "winter", "spring"):
        for i, criteria_level in enumerate(("low", "med", "high")):
            # df_all["gdd_{}_{}".format(criteria_level, season_name)]\
            #     = df_all.apply(lambda x: x.tasavg - criteria_lmh[season_name][i] if x.season == season_name and x.tasavg - criteria_lmh[season_name][i] > 0 else 0, axis=1)
            # df_all["prcp_{}".format(season_name)] = df_all.apply(lambda x: x.pr if x.season == season_name else 0, axis=1)

            # df_all["gdd_{}_{}".format(criteria_level, season_name)] = df_all["tasavg"] - criteria_lmh[season_name][i]
            df_all["gdd_{}_{}".format(criteria_level, season_name)] = df_all.apply(lambda x: get_igdd(x.tasmin, x.tasmax, criteria=criteria_lmh[season_name][i]), axis=1)
            df_all["gdd_{}_{}".format(criteria_level, season_name)] = np.where(df_all['season'] == season_name, df_all["gdd_{}_{}".format(criteria_level, season_name)], 0)
            df_all["gdd_{}_{}".format(criteria_level, season_name)] = np.where(df_all["gdd_{}_{}".format(criteria_level, season_name)] > 0, df_all["gdd_{}_{}".format(criteria_level, season_name)], 0)
            df_all['prcp_{}'.format(season_name)] = np.where(df_all['season'] == season_name, df_all["rainfall"], 0)
            df_all['snowfall_{}'.format(season_name)] = np.where(df_all['season'] == season_name, df_all["snowfall"], 0)


    #         df_all["gdd_a_{}_{}".format(criteria_level, season_name)] = df_all["tasavg"] - criteria_lmh_avg[season_name][i]
    #         df_all["gdd_a_{}_{}".format(criteria_level, season_name)] = np.where(df_all['season'] == season_name, df_all["gdd_a_{}_{}".format(criteria_level, season_name)], 0)
    #         df_all["gdd_a_{}_{}".format(criteria_level, season_name)] = np.where(df_all["gdd_a_{}_{}".format(criteria_level, season_name)] > 0, df_all["gdd_a_{}_{}".format(criteria_level, season_name)], 0)
    #
    # if "1951" in output_filename:
    #     df_all.to_csv("C:/Users/taegon/Desktop/test_igdd_daily.csv")
    #     print("test file generated.")

    df_all = df_all.groupby(["scenario", "county_fips", "season_year"]).agg({
        "gdd_low_fall": "sum", "gdd_med_fall": "sum", "gdd_high_fall": "sum", "prcp_fall": "sum", "snowfall_fall": "sum",
        "gdd_low_winter": "sum", "gdd_med_winter": "sum", "gdd_high_winter": "sum", "prcp_winter": "sum", "snowfall_winter": "sum",
        "gdd_low_spring": "sum", "gdd_med_spring": "sum", "gdd_high_spring": "sum", "prcp_spring": "sum", "snowfall_spring": "sum",
    })

    df_all = df_all[
        [
            "gdd_low_fall", "gdd_med_fall", "gdd_high_fall", "prcp_fall", "snowfall_fall",
            "gdd_low_winter", "gdd_med_winter", "gdd_high_winter", "prcp_winter", "snowfall_winter",
            "gdd_low_spring", "gdd_med_spring", "gdd_high_spring", "prcp_spring", "snowfall_spring",
        ]
    ]
    df_all = df_all.reset_index()
    # mask = df_all['season_year'].isin([1950, 2006])
    # df_all = df_all[~mask]
    # print(df_all.head())

    df_all.to_csv(output_filename, index=False)


def process_annual_gdd(weather_filename, output_filename, other_weather_file=None):
    df_nex = pd.read_csv(weather_filename)
    if other_weather_file is not None:
        df_nex_other = pd.read_csv(other_weather_file)
        df_nex = pd.concat([df_nex, df_nex_other], ignore_index=True)

    df_nex[['scenario', 'model', "date", "emsemble", "ee_id"]] = df_nex["system:index"].str.split("_", expand=True)
    df_nex["date"] = pd.to_datetime(df_nex["date"], format='%Y%m%d')
    df_nex["tasmax"] = df_nex["tasmax"] - 273.15
    df_nex["tasmin"] = df_nex["tasmin"] - 273.15
    df_nex["tasavg"] = (df_nex["tasmin"] + df_nex["tasmax"]) / 2
    df_nex["pr"] = df_nex["pr"] * 86400
    df_nex = df_nex.rename(columns={"GEOID": "county_fips", })

    scenarios = df_nex["scenario"].unique()
    # print(df_nex["scenario"].unique())
    if len(scenarios) > 1:
        if "rcp45" in scenarios:
            df_nex["scenario"] = "rcp45"
        elif "rcp85" in scenarios:
            df_nex["scenario"] = "rcp85"
        else:
            print("this should be checked.")
            print(scenarios)

    # print(df_nex[['scenario', 'model', "date", "emsemble", "ee_id"]].head())
    df_nex = df_nex[["county_fips", "scenario", "date", "tasmin", "tasmax", "tasavg", "pr"]]
    # print(df_nex.head())
    # print(output_filename)
    generate_panel_output(output_filename, df_nex)


def process(model_name):
    base_drive = "N:/WinterWheat"
    if sys.platform == "linux" or sys.platform == "linux2":  # MSI
        base_drive = "/home/jinzn/taegon/WinterWheat"
    # output_dir = get_project_root() / "output/panel_input_future_all"
    output_dir = os.path.join(os.path.join(base_drive, "NEX-GDDP-annual"), model_name)
    make_dirs(output_dir)
    base_dir_nex = os.path.join(os.path.join(base_drive, "NEX-GDDP"), model_name)

    scenarios = ["rcp45", "rcp85"]
    # periods = {
    #     "historical": (1951, 2005),
    #     "rcp45": (2007, 2100),
    #     "rcp85": (2007, 2100),
    # }
    periods = {
        "rcp45": (1999, 2019),
        "rcp85": (1999, 2019),
    }

    number_of_process = 4
    if len(sys.argv) >= 2:
        number_of_process = int(sys.argv[1])

    pool = multiprocessing.Pool(number_of_process)
    results = []

    for scenario in scenarios:
        for year in range(periods[scenario][0], periods[scenario][1] + 1):
            if year <= 2005:
                scenario_hist = "historical"
            else:
                scenario_hist = scenario
            # if year == 2006:
            #     continue
    # for scenario in reversed(scenarios):
    #     for year in reversed(range(periods[scenario][0], periods[scenario][1] + 1)):
            weather_filename = os.path.join(base_dir_nex,
                                            "NEX-GDDP-all-w25000-{}-{}-{}.csv".format(model_name, scenario_hist, year))
            weather_other_filename = os.path.join(base_dir_nex,
                                            "NEX-GDDP-all-other-w25000-{}-{}-{}.csv".format(model_name, scenario_hist, year))
            output_filename = os.path.join(output_dir,
                                           "climate_input_data_igdd_{}_{}_{}_20210925.csv".format(model_name, scenario,
                                                                                                 year))
            if not os.path.exists(weather_filename):
                continue
            if not os.path.exists(weather_other_filename):
                weather_other_filename = None
            if os.path.exists(output_filename):
                continue
            # process_annual_gdd(base_dir_nex, model_name, output_dir, scenario, year)
            results.append(pool.apply_async(process_annual_gdd, args=(weather_filename, output_filename, weather_other_filename)))

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
