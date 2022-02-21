import os
from typing import NamedTuple
import multiprocessing

import pandas as pd

from winter_wheat.util import get_project_root, make_dirs


class Coefficient(NamedTuple):
    name: str
    value: float


def generate_effect_data(df_all, var_name, output_dir, output_filename, df_fips=None):
    # county_fips,year,yield_survey,production_survey,acre_survey,acre_irrigated_survey,
    # irr_acre_percent,fdd1,fdd2,fdd3,fdd1_sc2,fdd1_sc5,fdd1_sc10,fdd2_sc2,fdd2_sc5,fdd2_sc10,
    # fdd3_sc2,fdd3_sc5,fdd3_sc10,fdd1_day,fdd2_day,fdd3_day,
    # fdd1_sc2_sctf,fdd1_sc5_sctf,fdd1_sc10_sctf,
    # fdd2_sc2_sctf,fdd2_sc5_sctf,fdd2_sc10_sctf,
    # fdd3_sc2_sctf,fdd3_sc5_sctf,fdd3_sc10_sctf,
    # gdd_low_fall,gdd_med_fall,gdd_high_fall,prcp_fall,
    # gdd_low_winter,gdd_med_winter,gdd_high_winter,prcp_winter,
    # gdd_low_spring,gdd_med_spring,gdd_high_spring,prcp_spring
    coeff = {
        # "fdd1": Coefficient("fdd1", -9.1434e-06),
        # "fdd1_sc2_sctf": Coefficient("fdd1_sc2_sctf", 8.8191e-06),
        # "fdd2": Coefficient("fdd2", -9.823e-06),
        # "fdd2_sc2_sctf": Coefficient("fdd2_sc2_sctf", 8.8274e-06),
        # "gdd3_spring": Coefficient("gdd_high_spring", -0.0039149),
        "gdd3_spring": Coefficient("gdd_high_spring", 1),
    }
    scenarios = ["rcp45", "rcp85"]
    periods = {
        "2021-2040": (2021, 2040),
        "2041-2070": (2041, 2070),
        "2071-2100": (2071, 2100),
    }

    for scenario in scenarios:
        for period_name, period_year in periods.items():
            dataframe = df_all[(df_all["scenario"] == scenario) & (df_all["year"] >= period_year[0])
                               & (df_all["year"] <= period_year[1])]
            if var_name == "fdd1_sc2_sctf":
                df_effect = dataframe.groupby(["county_fips"], as_index=False)[["fdd1", "fdd1_sc2_sctf"]].mean()
                df_effect["effect"] = df_effect["fdd1"] * df_effect["fdd1_sc2_sctf"] * coeff[var_name].value
            elif var_name == "fdd2_sc2_sctf":
                df_effect = dataframe.groupby(["county_fips"], as_index=False)[["fdd2", "fdd2_sc2_sctf"]].mean()
                df_effect["effect"] = df_effect["fdd2"] * df_effect["fdd2_sc2_sctf"] * coeff[var_name].value
            elif var_name == "fdd1_sc2_overlap":
                df_effect = dataframe.groupby(["county_fips"], as_index=False)[["fdd1", "fdd1_sc2_sctf"]].mean()
                df_effect["effect"] = df_effect["fdd1"] * coeff["fdd1"].value\
                                      + df_effect["fdd1"] * df_effect["fdd1_sc2_sctf"] * coeff["fdd1_sc2_sctf"].value
            elif var_name == "fdd2_sc2_overlap":
                df_effect = dataframe.groupby(["county_fips"], as_index=False)[["fdd2", "fdd2_sc2_sctf"]].mean()
                df_effect["effect"] = df_effect["fdd2"] * coeff["fdd2"].value\
                                      + df_effect["fdd2"] * df_effect["fdd2_sc2_sctf"] * coeff["fdd2_sc2_sctf"].value
            else:
                df_effect = dataframe.groupby(["county_fips"], as_index=False)[[coeff[var_name].name]].mean()
                df_effect["effect"] = df_effect[coeff[var_name].name] * coeff[var_name].value
            # print(df_effect.head(10))
            df_effect = df_effect.reset_index()
            df_effect = df_effect.sort_values(by=['county_fips', ])
            if df_fips is not None:
                df_effect = pd.merge(df_fips, df_effect, on="county_fips", how="left")

            # df_effect = df_effect[["county_fips", "effect"]]
            if var_name.startswith("fdd1_sc2"):
                df_effect = df_effect[["county_fips", "fdd1", "fdd1_sc2_sctf", "effect"]]
            elif var_name.startswith("fdd2_sc2"):
                df_effect = df_effect[["county_fips", "fdd2", "fdd2_sc2_sctf", "effect"]]
            else:
                df_effect = df_effect[["county_fips", coeff[var_name].name, "effect"]]
            output_dir_each = os.path.join(output_dir, scenario)
            output_dir_each = os.path.join(output_dir_each, period_name)
            make_dirs(output_dir_each)
            df_effect.to_csv(os.path.join(output_dir_each, output_filename), index=False)


def generate_effect_data_rank(df_all, var_name, rank_percent, output_dir, output_filename, df_fips=None):
    # county_fips,year,yield_survey,production_survey,acre_survey,acre_irrigated_survey,
    # irr_acre_percent,fdd1,fdd2,fdd3,fdd1_sc2,fdd1_sc5,fdd1_sc10,fdd2_sc2,fdd2_sc5,fdd2_sc10,
    # fdd3_sc2,fdd3_sc5,fdd3_sc10,fdd1_day,fdd2_day,fdd3_day,
    # fdd1_sc2_sctf,fdd1_sc5_sctf,fdd1_sc10_sctf,
    # fdd2_sc2_sctf,fdd2_sc5_sctf,fdd2_sc10_sctf,
    # fdd3_sc2_sctf,fdd3_sc5_sctf,fdd3_sc10_sctf,
    # gdd_low_fall,gdd_med_fall,gdd_high_fall,prcp_fall,
    # gdd_low_winter,gdd_med_winter,gdd_high_winter,prcp_winter,
    # gdd_low_spring,gdd_med_spring,gdd_high_spring,prcp_spring
    coeff = {
        # "fdd1": Coefficient("fdd1", -9.1434e-06),
        # "fdd1_sc2_sctf": Coefficient("fdd1_sc2_sctf", 8.8191e-06),
        # "fdd2": Coefficient("fdd2", -9.823e-06),
        # "fdd2_sc2_sctf": Coefficient("fdd2_sc2_sctf", 8.8274e-06),
        # "gdd3_spring": Coefficient("gdd_high_spring", -0.0039149),
        "gdd3_spring": Coefficient("gdd_high_spring", 1),
    }
    scenarios = ["rcp45", "rcp85"]
    periods = {
        "2021-2040": (2021, 2040),
        "2041-2070": (2041, 2070),
        "2071-2100": (2071, 2100),
    }

    for scenario in scenarios:
        for period_name, period_year in periods.items():
            rank = round((period_year[1] - period_year[0] + 1) * rank_percent)
            # print(rank,  period_year[0],  period_year[1])
            dataframe = df_all[(df_all["scenario"] == scenario) & (df_all["year"] >= period_year[0])
                               & (df_all["year"] <= period_year[1])]

            if var_name == "fdd1_sc2_sctf":
                df_effect = dataframe.copy()
                df_effect["rank"] = df_effect.groupby("county_fips")[["fdd1"]].rank("first", ascending=False)
                df_effect = df_effect[df_effect["rank"] == rank]
                df_effect["effect"] = df_effect["fdd1"] * df_effect["fdd1_sc2_sctf"] * coeff[var_name].value
            elif var_name == "fdd2_sc2_sctf":
                df_effect = dataframe.copy()
                df_effect["rank"] = df_effect.groupby("county_fips")[["fdd2"]].rank("first", ascending=False)
                df_effect = df_effect[df_effect["rank"] == rank]
                df_effect["effect"] = df_effect["fdd2"] * df_effect["fdd2_sc2_sctf"] * coeff[var_name].value
            elif var_name == "fdd1_sc2_overlap":
                df_effect = dataframe.copy()
                df_effect["rank"] = df_effect.groupby("county_fips")[["fdd1"]].rank("first", ascending=False)
                df_effect = df_effect[df_effect["rank"] == rank]
                df_effect["effect"] = df_effect["fdd1"] * coeff["fdd1"].value\
                                      + df_effect["fdd1"] * df_effect["fdd1_sc2_sctf"] * coeff["fdd1_sc2_sctf"].value
            elif var_name == "fdd2_sc2_overlap":
                df_effect = dataframe.copy()
                df_effect["rank"] = df_effect.groupby("county_fips")[["fdd2"]].rank("first", ascending=False)
                df_effect = df_effect[df_effect["rank"] == rank]
                df_effect["effect"] = df_effect["fdd2"] * coeff["fdd2"].value\
                                      + df_effect["fdd2"] * df_effect["fdd2_sc2_sctf"] * coeff["fdd2_sc2_sctf"].value
            elif var_name == "gdd3_spring":
                df_effect = dataframe.copy()
                df_effect["rank"] = df_effect.groupby("county_fips")[["fdd2"]].rank("first", ascending=False)
                df_effect = df_effect[df_effect["rank"] == rank]
                df_effect["effect"] = df_effect[coeff[var_name].name] * coeff[var_name].value
            else:
                df_effect = dataframe.copy()
                df_effect["rank"] = df_effect.groupby("county_fips")[[coeff[var_name].name]].rank("first", ascending=False)
                df_effect = df_effect[df_effect["rank"] == rank]
                df_effect["effect"] = df_effect[coeff[var_name].name] * coeff[var_name].value
            # print(df_effect.head(10))
            df_effect = df_effect.reset_index()
            df_effect = df_effect.sort_values(by=['county_fips', ])
            if df_fips is not None:
                df_effect = pd.merge(df_fips, df_effect, on="county_fips", how="left")
                # df_effect = df_effect.astype({"year": int, })  # ValueError: Cannot convert non-finite values (NA or inf) to integer

            if var_name.startswith("fdd1_sc2"):
                df_effect = df_effect[["county_fips", "fdd1", "fdd1_sc2_sctf", "effect", "year"]]
            elif var_name.startswith("fdd2_sc2"):
                df_effect = df_effect[["county_fips", "fdd2", "fdd2_sc2_sctf", "effect", "year"]]
            else:
                df_effect = df_effect[["county_fips", coeff[var_name].name, "effect", "year"]]

            output_dir_each = os.path.join(output_dir, scenario)
            output_dir_each = os.path.join(output_dir_each, period_name)
            make_dirs(output_dir_each)

            df_effect.to_csv(os.path.join(output_dir_each, output_filename), index=False)


def make_effect_avg(df, output_dir, df_fips=None):
    variables = ["gdd3_spring",]

    for variable in variables:
        generate_effect_data(df_all=df,
                             var_name=variable,
                             output_dir=output_dir,
                             output_filename="gridMET_i{}_avg.csv".format(variable),
                             df_fips=df_fips)


def make_effect_rank(df, output_dir, df_fips=None):
    variables = ["gdd3_spring",]
    for variable in variables:
        generate_effect_data_rank(df_all=df,
                                  var_name=variable,
                                  rank_percent=.1,
                                  output_dir=output_dir,
                                  output_filename="gridMET_i{}_top10p.csv".format(variable),
                                  df_fips=df_fips)


def process(model_name):
    base_dir = os.path.join("N:/WinterWheat/model-input-swe-bc", model_name)
    output_dir = get_project_root() / "output/county_map/v20200726-gdd-a"  # all states
    output_dir = get_project_root() / "output/county_map/v20200730"  # all states + all missing counties
    output_dir = get_project_root() / "output/county_map/v20200806-for-Peng-rawdata"  # all states + all missing counties including variables not only effect
    output_dir = get_project_root() / "output/county_map/v20200814-for-Peng-rawdata"  # fix igdd/ifdd errors
    output_dir = os.path.join(output_dir, model_name)
    data_filename = os.path.join(base_dir, "winter_wheat_input_igdd_sdepth_bc_v20200814_{}.csv".format(model_name))

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
        # "GFDL-CM3",
        # "GFDL-ESM2G",
        # "MIROC-ESM",
        # "MIROC-ESM-CHEM",
        # "MPI-ESM-LR",
        "MPI-ESM-MR",
        # "MRI-CGCM3",
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
