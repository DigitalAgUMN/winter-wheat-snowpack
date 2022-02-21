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
    # print("starting fdd2")
    df_all["fdd2"] = df_all.apply(lambda x: get_fdd(x.tasmin, x.tasmax, criteria=-5), axis=1)
    # print("starting fdd3")
    df_all["fdd3"] = df_all.apply(lambda x: get_fdd(x.tasmin, x.tasmax, criteria=-10), axis=1)
    for fdd_key in ["fdd1", "fdd2", "fdd3"]:
        for snow_covered in (2, 5, 10, 15, 20):
            sc_key = "sc{}".format(snow_covered)
            # df_all["{}_{}".format(fdd_key, sc_key)] = df_all.apply(lambda x: 1 if x[fdd_key] > 0 and x["snow_depth_cm"] >= snow_covered else 0, axis=1)
            df_all["{}_{}".format(fdd_key, sc_key)] = np.where((df_all["snow_depth_cm"] >= snow_covered) & (df_all[fdd_key] > 0), 1, 0)

    df_all["fdd1_day"] = 0
    df_all["fdd2_day"] = 0
    df_all["fdd3_day"] = 0
    df_all.loc[df_all['fdd1'] > 0, 'fdd1_day'] = 1
    df_all.loc[df_all['fdd2'] > 0, 'fdd2_day'] = 1
    df_all.loc[df_all['fdd3'] > 0, 'fdd3_day'] = 1

    df_all = df_all.groupby(["scenario", "county_fips", "season_year"]).agg({
        'fdd1': 'sum',
        'fdd2': 'sum',
        'fdd3': 'sum',
        "fdd1_sc2": 'sum', "fdd1_sc5": 'sum', "fdd1_sc10": 'sum', "fdd1_sc15": 'sum', "fdd1_sc20": 'sum',
        "fdd2_sc2": 'sum', "fdd2_sc5": 'sum', "fdd2_sc10": 'sum', "fdd2_sc15": 'sum', "fdd2_sc20": 'sum',
        "fdd3_sc2": 'sum', "fdd3_sc5": 'sum', "fdd3_sc10": 'sum', "fdd3_sc15": 'sum', "fdd3_sc20": 'sum',
        "fdd1_day": 'sum', "fdd2_day": 'sum', "fdd3_day": 'sum',
    })
    # print(df_all.columns)
    # print(df_all.dtypes)
    # print(df_all.head())

    for fdd_key in ["fdd1", "fdd2", "fdd3"]:
        for snow_covered in (2, 5, 10, 15, 20):
            sc_key = "sc{}".format(snow_covered)
            df_all["{}_{}_sctf".format(fdd_key, sc_key)] = df_all.apply(lambda x:
                                                                        x["{}_{}".format(fdd_key, sc_key)]
                                                                        / x["{}_day".format(fdd_key)]
                                                                        if x["{}_day".format(fdd_key)] > 0 else np.NaN, axis=1)

    df_all = df_all[
        [
            "fdd1", "fdd2", "fdd3",
            "fdd1_sc2", "fdd1_sc5", "fdd1_sc10", "fdd1_sc15", "fdd1_sc20",
            "fdd2_sc2", "fdd2_sc5", "fdd2_sc10", "fdd2_sc15", "fdd2_sc20",
            "fdd3_sc2", "fdd3_sc5", "fdd3_sc10", "fdd3_sc15", "fdd3_sc20",
            "fdd1_day", "fdd2_day", "fdd3_day",
            "fdd1_sc2_sctf", "fdd1_sc5_sctf", "fdd1_sc10_sctf", "fdd1_sc15_sctf", "fdd1_sc20_sctf",
            "fdd2_sc2_sctf", "fdd2_sc5_sctf", "fdd2_sc10_sctf", "fdd2_sc15_sctf", "fdd2_sc20_sctf",
            "fdd3_sc2_sctf", "fdd3_sc5_sctf", "fdd3_sc10_sctf", "fdd3_sc15_sctf", "fdd3_sc20_sctf",
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


def process_annual_fdd_with_corrected_swe(weather_filename, swe_filename, reference_data, cchist_data, output_filename, swe_corrected_filename, other_weather_file=None):
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
    # periods = {
    #     "historical": (1980, 2005),
    #     "rcp45": (2080, 2100),
    #     "rcp85": (2080, 2100),
    # }

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
                                           "climate_input_data_fdd_{}_{}_{}_20210109.csv".format(model_name, scenario,
                                                                                                 year))
            if not os.path.exists(weather_filename):
                continue
            if not os.path.exists(weather_other_filename):
                weather_other_filename = None
            if os.path.exists(output_filename):
                continue
            # process_annual_gdd(base_dir_nex, model_name, output_dir, scenario, year)
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
                                           "climate_input_data_fdd_swe_corrected_{}_{}_{}_20210109.csv".format(model_name, scenario,
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
    if len(sys.argv) >= 3:
        model_idx = int(sys.argv[2])
        process(models[model_idx])
    else:
        for model in models:
            process(model)


if __name__ == "__main__":
    main()
