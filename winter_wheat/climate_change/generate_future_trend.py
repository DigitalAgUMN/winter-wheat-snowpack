import os
import multiprocessing
import datetime
import sys

import pandas as pd
import numpy as np
import tqdm

from winter_wheat.util import get_project_root, make_dirs


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

    return df


def generate_panel_output(output_filename, df_all):
    df_all = append_season_year(df_all)
    # df_all = df_all[((df_all["month"] >= 9) | (df_all["month"] < 6)) & (df_all["year"] >= 1950)]
    df_all = df_all.copy()
    # print(df_all.head())
    # print(df_all.columns)
    # df_all["snow_depth_cm"] = df_all.apply(lambda x: x.snow_depth_mm / 10.0, axis=1)
    df_all["snow_depth_cm"] = df_all["snow_depth_mm"] / 10.0


    for season_name in ("fall", "winter", "spring"):
        df_all['tasmin_{}'.format(season_name)] = np.where(df_all['season'] == season_name, df_all["tasmin"], np.nan)
        df_all['tasmax_{}'.format(season_name)] = np.where(df_all['season'] == season_name, df_all["tasmax"], np.nan)
        df_all['tasavg_{}'.format(season_name)] = np.where(df_all['season'] == season_name, df_all["tasavg"], np.nan)
        df_all['prcp_{}'.format(season_name)] = np.where(df_all['season'] == season_name, df_all["pr"], np.nan)
        df_all['snow_depth_mm_{}'.format(season_name)] = np.where(df_all['season'] == season_name, df_all["snow_depth_mm"], np.nan)
    df_all = df_all.groupby(["scenario", "county_fips", "season_year"]).agg({
        "tasmin_fall": "mean", "tasmax_fall": "mean", "tasavg_fall": "mean", "prcp_fall": "sum", "snow_depth_mm_fall": "mean",
        "tasmin_winter": "mean", "tasmax_winter": "mean", "tasavg_winter": "mean", "prcp_winter": "sum", "snow_depth_mm_winter": "mean",
        "tasmin_spring": "mean", "tasmax_spring": "mean", "tasavg_spring": "mean", "prcp_spring": "sum", "snow_depth_mm_spring": "mean",
        "tasmin": "mean", "tasmax": "mean", "tasavg": "mean", "pr": "sum", "snow_depth_mm": "mean",
    })

    # print(df_all.columns)
    # print(df_all.dtypes)
    # print(df_all.head())


    df_all = df_all[
        [
            "tasmin_fall", "tasmax_fall", "tasavg_fall", "prcp_fall", "snow_depth_mm_fall",
            "tasmin_winter", "tasmax_winter", "tasavg_winter", "prcp_winter", "snow_depth_mm_winter",
            "tasmin_spring", "tasmax_spring", "tasavg_spring", "prcp_spring", "snow_depth_mm_spring",
            "tasmin", "tasmax", "tasavg", "pr", "snow_depth_mm",
        ]
    ]
    df_all = df_all.reset_index()
    # mask = df_all['season_year'].isin([1950, 2006])
    # df_all = df_all[~mask]
    # print(df_all.head())
    df_all.to_csv(output_filename, index=False)


def bias_correction(df_cc, reference_data, cchist_data):
    df_cc['mean_obs'] = df_cc["county_fips"].map(reference_data[('snow_depth_mm', "mean")])
    df_cc['std_obs'] = df_cc["county_fips"].map(reference_data[('snow_depth_mm', "std")])
    df_cc['mean_cchist'] = df_cc["county_fips"].map(cchist_data[('snow_depth_mm', "mean")])
    df_cc['std_cchist'] = df_cc["county_fips"].map(cchist_data[('snow_depth_mm', "std")])

    df_cc['snow_depth_corrected_mm'] = (df_cc["snow_depth_mm"] - df_cc['mean_cchist']) * (df_cc['std_obs'] / df_cc['std_cchist']) + df_cc['mean_obs']
    df_cc['snow_depth_corrected_mm'] = np.where(df_cc['snow_depth_corrected_mm'] < 0, 0, df_cc['snow_depth_corrected_mm'])
    df_cc['snow_depth_corrected_mm'] = np.where(df_cc["snow_depth_mm"] <= 0, 0, df_cc['snow_depth_corrected_mm'])
    df_cc["snow_depth_raw_mm"] = df_cc["snow_depth_mm"]
    df_cc["snow_depth_mm"] = df_cc['snow_depth_corrected_mm']

    # print(df_cc[['mean_cchist', 'std_cchist', "snow_depth_raw_mm", "snow_depth_mm"]].head(10))

    # df_ref["mean_obs"] = reference_data[]
    # df_ref["std_obs"] = reference_data[]
    # df_ref["mean_ref"] = reference_data[]
    # df_ref["std_ref"] = reference_data[]

    return df_cc


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


def process_annual_fdd(weather_filename, swe_filename, output_filename, other_weather_file):
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


def process_annual_fdd_with_corrected_swe(weather_filename, swe_filename, reference_data, cchist_data, output_filename, swe_corrected_filename, other_weather_file):
    df_nex = read_nex_gddp(weather_filename)
    if other_weather_file is not None:
        df_nex_other = read_nex_gddp(other_weather_file)
        df_nex = pd.concat([df_nex, df_nex_other], ignore_index=True)

    if os.path.exists(swe_corrected_filename):
        df_swe_bias_corr = pd.read_csv(swe_corrected_filename, parse_dates=['date'])
    else:
        df_swe = pd.read_csv(swe_filename, parse_dates=['date'])
        df_swe["snow_depth_mm"] = df_swe["snw"] / 0.274
        df_swe = df_swe.rename(columns={"FIPS": "county_fips", })

        df_swe_bias_corr = bias_correction(df_swe, reference_data, cchist_data)
        df_swe_bias_corr.to_csv(swe_corrected_filename, index=False)

    df_all = pd.merge(df_nex, df_swe_bias_corr, on=["scenario", "date", "county_fips"],
                      how='inner')

    generate_panel_output(output_filename, df_all)


def get_bias_correction(df_cc, df_ref):
    df_cc["year"] = df_cc["date"].dt.year
    df_cc["month"] = df_cc["date"].dt.month
    df_cc["key"] = df_cc["county_fips"]
    df_ref["year"] = df_ref["date"].dt.year
    df_ref["month"] = df_ref["date"].dt.month
    df_ref["key"] = df_ref["county_fips"]

    reference_data = df_ref[((df_ref["date"] >= "1998-09-01") & (df_ref["date"] < "2006-01-01") & ((df_ref["month"] < 6) | (df_ref["month"] >= 9)))]

    reference_data = reference_data.groupby(["key"]).agg({
        'snow_depth_mm': ['mean', 'std',],
    })

    target_data = df_cc[((df_cc["date"] >= "1998-08-01") & (df_cc["date"] < "2006-01-01") & (df_cc["scenario"] == "historical") & ((df_cc["month"] < 6) | (df_cc["month"] >= 9)))]
    # print(df_cc[["scenario", "date", "snow_depth_mm",]].head())
    # print(target_data.head())

    # target_data = target_data.groupby(["model", "county", "state"]).agg({
    #     'snow_depth_mm': ['mean', 'std',],
    # })
    target_data = target_data.groupby(["key"]).agg({
        'snow_depth_mm': ['mean', 'std',],
    })
    # df_cc['mean_obs'] = df_cc["key"].map(reference_data[('snow_depth_mm', "mean")])
    # df_cc['std_obs'] = df_cc["key"].map(reference_data[('snow_depth_mm', "std")])
    # df_cc['mean_cchist'] = df_cc["key"].map(target_data[('snow_depth_mm', "mean")])
    # df_cc['std_cchist'] = df_cc["key"].map(target_data[('snow_depth_mm', "std")])
    return reference_data, target_data


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
    periods = {
        "historical": (1951, 2005),
        "rcp45": (2007, 2100),
        "rcp85": (2007, 2100),
    }

    pool = multiprocessing.Pool(4)
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
                                           "climate_trend_{}_{}_{}_20210129.csv".format(model_name, scenario,
                                                                                        year))
            if not os.path.exists(weather_filename):
                continue
            if not os.path.exists(weather_other_filename):
                weather_other_filename = None
            if os.path.exists(output_filename):
                continue
            # process_annual_gdd(base_dir_nex, model_name, output_dir, scenario, year)
            # mpi_config.append((weather_filename, swe_filename, output_filename, weather_other_filename))
            results.append(pool.apply_async(process_annual_fdd, args=(weather_filename, swe_filename, output_filename,weather_other_filename)))


    output_dir_swe = os.path.join(os.path.join(base_drive, "CMIP5_snw_county_corrected"), model_name)
    output_dir = os.path.join(os.path.join(base_drive, "NEX-GDDP-annual-swe-corrected"), model_name)
    make_dirs(output_dir)
    make_dirs(output_dir_swe)

    df_cmc = pd.read_csv(cmc_filename, parse_dates=['date'])
    # df_cmc = df_cmc.rename(columns={"FIPS": "county_fips", })
    df_cmc["snow_depth_mm"] = df_cmc["snow_depth_cm"] * 10.
    df_cmc = df_cmc[(df_cmc["date"] >= "1998-09-01") & (df_cmc["date"] < "2006-01-01")]
    df_swe_hist = None

    for year in range(1999, 2006):
        swe_filename = os.path.join(base_dir_swe,
                                    "df-snw-{}-{}-{}.csv".format(model_name, "historical", year))
        df_swe_temp = pd.read_csv(swe_filename, parse_dates=['date'])
        df_swe_temp["snow_depth_mm"] = df_swe_temp["snw"] / 0.274
        df_swe_temp = df_swe_temp.rename(columns={"FIPS": "county_fips", })
        if df_swe_hist is None:
            df_swe_hist = df_swe_temp
        else:
            df_swe_hist = pd.concat([df_swe_hist, df_swe_temp])

    reference_data, cchist_data = get_bias_correction(df_swe_hist, df_cmc)

    for scenario in scenarios:
        for year in range(periods[scenario][0], periods[scenario][1] + 1):
            weather_filename = os.path.join(base_dir_nex,
                                            "NEX-GDDP-all-w25000-{}-{}-{}.csv".format(model_name, scenario, year))
            weather_other_filename = os.path.join(base_dir_nex,
                                            "NEX-GDDP-all-other-w25000-{}-{}-{}.csv".format(model_name, scenario, year))
            swe_filename = os.path.join(base_dir_swe,
                                            "df-snw-{}-{}-{}.csv".format(model_name, scenario, year))
            output_filename = os.path.join(output_dir,
                                           "climate_trend_swe_corrected_{}_{}_{}_20210129.csv".format(model_name, scenario,
                                                                                                 year))
            swe_corrected_filename = os.path.join(output_dir_swe,
                                        "df-snw-corrected-{}-{}-{}.csv".format(model_name, scenario, year))
            if not os.path.exists(weather_filename):
                continue
            if not os.path.exists(weather_other_filename):
                weather_other_filename = None
            if os.path.exists(output_filename):
                continue
            results.append(pool.apply_async(process_annual_fdd_with_corrected_swe,
                                            args=(weather_filename, swe_filename, reference_data, cchist_data, output_filename, swe_corrected_filename, weather_other_filename)))


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
    for model in models:
        process(model)


if __name__ == "__main__":
    main()
