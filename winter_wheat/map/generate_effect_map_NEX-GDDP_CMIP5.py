import os
from typing import NamedTuple
import multiprocessing

import numpy as np
import pandas as pd

from winter_wheat.map.generate_effect_map_gridMET import aggregate_rank, aggregate
from winter_wheat.util import get_project_root, make_dirs


class Coefficient(NamedTuple):
    name: str
    value: float
    SE: float

# coeff = {
#     "fdd1": Coefficient("fdd1", -0.0005494),
#     "gdd1_spring": Coefficient("gdd_low_spring", 5.31e-5),
#     "gdd1_winter": Coefficient("gdd_low_winter", 6.1958e-7),
#     "gdd1_fall": Coefficient("gdd_low_fall", 0.00036712),
#     "gdd2_spring": Coefficient("gdd_med_spring", -0.0010333),
#     "gdd2_winter": Coefficient("gdd_med_winter", 0.00069274),
#     "gdd2_fall": Coefficient("gdd_med_fall", -0.0020253),
#     "gdd3_spring": Coefficient("gdd_high_spring", -0.0026711),
#     "gdd3_winter": Coefficient("gdd_high_winter", -0.00077021),
#     "gdd3_fall": Coefficient("gdd_high_fall", 0.00096057),
#     "prcp_fall": Coefficient("prcp_fall", 0.00027297),
#     "prcp_winter": Coefficient("prcp_winter", 0.00049957),
#     "prcp_spring": Coefficient("prcp_spring", 0.00039303),
#     "fdd1_sc2_sctf": Coefficient("fdd1_sc2_sctf", 0.00059592),
# }

coeff = {
    "fdd1": Coefficient("fdd1", -0.00050632, 3.0832e-5),
    "gdd1_spring": Coefficient("gdd_low_spring", 2.1181e-5, 2.9334e-5),
    "gdd1_winter": Coefficient("gdd_low_winter", 0.00033099, 0.0001576),
    "gdd1_fall": Coefficient("gdd_low_fall", 0.00048502, 8.2739e-5),
    "gdd2_spring": Coefficient("gdd_med_spring", -0.00091571, 8.1836e-5),
    "gdd2_winter": Coefficient("gdd_med_winter", 0.00053237, 0.00023883),
    "gdd2_fall": Coefficient("gdd_med_fall", -0.00212, 0.00012421),
    "gdd3_spring": Coefficient("gdd_high_spring", -0.0029815, 0.00038717),
    "gdd3_winter": Coefficient("gdd_high_winter", -0.00076423, 0.00011462),
    "gdd3_fall": Coefficient("gdd_high_fall", 0.00093829, 5.724e-5),
    "prcp_fall": Coefficient("prcp_fall", 0.00026098, 2.1384e-5),
    "prcp_winter": Coefficient("prcp_winter", 0.00037849, 3.532e-5),
    "prcp_spring": Coefficient("prcp_spring", 0.00040059, 2.0319e-5),
    "snowfall_fall": Coefficient("snowfall_fall", 0.0016007, 0.00020739),
    "snowfall_winter": Coefficient("snowfall_winter", 0.00075981, 6.8538e-5),
    "snowfall_spring": Coefficient("snowfall_spring", -0.00029942, 0.00016845),
    "fdd1_sc2_sctf": Coefficient("fdd1_sc2_sctf", 0.00055051, 2.8933e-5),
}

scenarios = ["rcp45", "rcp85"]
# periods = {
#     "2021-2040": (2021, 2040),
#     "2041-2070": (2041, 2070),
#     "2071-2100": (2071, 2100),
# }
periods = {
    # "2021-2040": (2021, 2040),
    # "2040-2060": (2040, 2060),
    "2080-2100": (2080, 2100),
}

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

    for scenario in scenarios:
        for period_name, period_year in periods.items():
            dataframe = df_all[(df_all["scenario"] == scenario) & (df_all["year"] >= period_year[0])
                               & (df_all["year"] <= period_year[1])]
            if var_name == "fdd1_sc2_sctf":
                df_effect = dataframe.groupby(["county_fips"], as_index=False)[["fdd1", "fdd1_sc2_sctf"]].mean()
                df_effect["fdd1:sc2_sctf"] = df_effect["fdd1"] * df_effect["fdd1_sc2_sctf"]
                df_effect["effect"] = df_effect["fdd1"] * df_effect["fdd1_sc2_sctf"] * coeff[var_name].value
                df_effect["ci"] = np.abs(df_effect["fdd1"] * df_effect["fdd1_sc2_sctf"] * coeff[var_name].SE)
            elif var_name == "fdd1_sc2_overlap":
                df_effect = dataframe.groupby(["county_fips"], as_index=False)[["fdd1", "fdd1_sc2_sctf"]].mean()
                df_effect["fdd1:sc2_sctf"] = df_effect["fdd1"] * df_effect["fdd1_sc2_sctf"]
                df_effect["effect"] = df_effect["fdd1"] * coeff["fdd1"].value\
                                      + df_effect["fdd1"] * df_effect["fdd1_sc2_sctf"] * coeff["fdd1_sc2_sctf"].value
                df_effect["ci"] = np.sqrt(np.square(df_effect["fdd1"] * coeff["fdd1"].SE) \
                                  + np.square(
                    df_effect["fdd1"] * df_effect["fdd1_sc2_sctf"] * coeff["fdd1_sc2_sctf"].SE))
            else:
                df_effect = dataframe.groupby(["county_fips"], as_index=False)[[coeff[var_name].name]].mean()
                df_effect["effect"] = df_effect[coeff[var_name].name] * coeff[var_name].value
                df_effect["ci"] = np.abs(df_effect[coeff[var_name].name] * coeff[var_name].SE)
            # print(df_effect.head(10))
            df_effect = df_effect.reset_index()
            df_effect = df_effect.sort_values(by=['county_fips', ])
            if df_fips is not None:
                df_effect = pd.merge(df_fips, df_effect, on="county_fips", how="left")

            # df_effect = df_effect[["county_fips", "effect"]]
            if var_name.startswith("fdd1_sc2"):
                df_effect = df_effect[["county_fips", "effect", "fdd1", "fdd1_sc2_sctf", "fdd1:sc2_sctf", "ci"]]
            else:
                df_effect = df_effect[["county_fips", "effect", coeff[var_name].name, "ci"]]
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
                df_effect["fdd1:sc2_sctf"] = df_effect["fdd1"] * df_effect["fdd1_sc2_sctf"]
                df_effect["effect"] = df_effect["fdd1"] * df_effect["fdd1_sc2_sctf"] * coeff[var_name].value
                df_effect["ci"] = np.abs(df_effect["fdd1"] * df_effect["fdd1_sc2_sctf"] * coeff[var_name].SE)
            elif var_name == "fdd1_sc2_overlap":
                df_effect = dataframe.copy()
                df_effect["rank"] = df_effect.groupby("county_fips")[["fdd1"]].rank("first", ascending=False)
                df_effect = df_effect[df_effect["rank"] == rank]
                df_effect["fdd1:sc2_sctf"] = df_effect["fdd1"] * df_effect["fdd1_sc2_sctf"]
                df_effect["effect"] = df_effect["fdd1"] * coeff["fdd1"].value\
                                      + df_effect["fdd1"] * df_effect["fdd1_sc2_sctf"] * coeff["fdd1_sc2_sctf"].value
                df_effect["ci"] = np.sqrt(np.square(df_effect["fdd1"] * coeff["fdd1"].SE) \
                                  + np.square(
                    df_effect["fdd1"] * df_effect["fdd1_sc2_sctf"] * coeff["fdd1_sc2_sctf"].SE))
            else:
                df_effect = dataframe.copy()
                df_effect["rank"] = df_effect.groupby("county_fips")[["fdd1"]].rank("first", ascending=False)
                # df_effect["rank"] = df_effect.groupby("county_fips")[[coeff[var_name].name]].rank("first", ascending=False)
                df_effect = df_effect[df_effect["rank"] == rank]
                df_effect["effect"] = df_effect[coeff[var_name].name] * coeff[var_name].value
                df_effect["ci"] = np.abs(df_effect[coeff[var_name].name] * coeff[var_name].SE)

                # df_effect_avg = dataframe.copy()
                # df_effect_avg = df_effect_avg.groupby(["county_fips"], as_index=False)[[coeff[var_name].name]].mean()
                # df_effect_avg["effect"] = df_effect_avg[coeff[var_name].name] * coeff[var_name].value

                # if var_name == "fdd1":
                #     print(df_effect[(df_effect["county_fips"]==1001)&(df_effect["rank"]<10)][["fdd1", "effect", "rank"]].head(10))
                # df_effect = df_effect[df_effect["rank"] == rank]
                # if var_name == "fdd1":
                #     print("top10p")
                #     print(df_effect[(df_effect["county_fips"]==1001)&(df_effect["rank"]<10)][["fdd1", "effect", "rank"]].head(10))
                #     print("avg")
                #     print(df_effect_avg[(df_effect_avg["county_fips"] == 1001)][
                #               ["fdd1", "effect"]].head(10))

            # print(df_effect.head(10))
            df_effect = df_effect.reset_index()
            df_effect = df_effect.sort_values(by=['county_fips', ])
            if df_fips is not None:
                df_effect = pd.merge(df_fips, df_effect, on="county_fips", how="left")
                # df_effect = df_effect.astype({"year": int, })  # ValueError: Cannot convert non-finite values (NA or inf) to integer

            if var_name.startswith("fdd1_sc2"):
                df_effect = df_effect[["county_fips", "effect", "fdd1", "fdd1_sc2_sctf", "fdd1:sc2_sctf", "year", "ci"]]
            else:
                df_effect = df_effect[["county_fips", "effect", coeff[var_name].name, "year", "ci"]]

            output_dir_each = os.path.join(output_dir, scenario)
            output_dir_each = os.path.join(output_dir_each, period_name)
            make_dirs(output_dir_each)

            df_effect.to_csv(os.path.join(output_dir_each, output_filename), index=False)


def generate_effect_data_rank_gdd(df_all, var_name, rank_percent, output_dir, output_filename, df_fips=None):
    for scenario in scenarios:
        for period_name, period_year in periods.items():
            rank = round((period_year[1] - period_year[0] + 1) * rank_percent)
            # print(rank,  period_year[0],  period_year[1])
            dataframe = df_all[(df_all["scenario"] == scenario) & (df_all["year"] >= period_year[0])
                               & (df_all["year"] <= period_year[1])]

            df_effect = dataframe.copy()
            df_effect["gdd_effect"] = \
                df_effect[coeff["gdd1_spring"].name] * coeff["gdd1_spring"].value \
                + df_effect[coeff["gdd1_winter"].name] * coeff["gdd1_winter"].value \
                + df_effect[coeff["gdd1_fall"].name] * coeff["gdd1_fall"].value \
                + df_effect[coeff["gdd2_spring"].name] * coeff["gdd2_spring"].value \
                + df_effect[coeff["gdd2_winter"].name] * coeff["gdd2_winter"].value \
                + df_effect[coeff["gdd2_fall"].name] * coeff["gdd2_fall"].value \
                + df_effect[coeff["gdd3_spring"].name] * coeff["gdd3_spring"].value \
                + df_effect[coeff["gdd3_winter"].name] * coeff["gdd3_winter"].value \
                + df_effect[coeff["gdd3_fall"].name] * coeff["gdd3_fall"].value

            if var_name == "fdd1_sc2_sctf":
                df_effect["rank"] = df_effect.groupby("county_fips")[["gdd_effect"]].rank("first", ascending=True)
                df_effect = df_effect[df_effect["rank"] == rank]
                df_effect["fdd1:sc2_sctf"] = df_effect["fdd1"] * df_effect["fdd1_sc2_sctf"]
                df_effect["effect"] = df_effect["fdd1"] * df_effect["fdd1_sc2_sctf"] * coeff[var_name].value
                df_effect["ci"] = np.abs(df_effect["fdd1"] * df_effect["fdd1_sc2_sctf"] * coeff[var_name].SE)
            elif var_name == "fdd1_sc2_overlap":
                df_effect["rank"] = df_effect.groupby("county_fips")[["gdd_effect"]].rank("first", ascending=True)
                df_effect = df_effect[df_effect["rank"] == rank]
                df_effect["fdd1:sc2_sctf"] = df_effect["fdd1"] * df_effect["fdd1_sc2_sctf"]
                df_effect["effect"] = df_effect["fdd1"] * coeff["fdd1"].value\
                                      + df_effect["fdd1"] * df_effect["fdd1_sc2_sctf"] * coeff["fdd1_sc2_sctf"].value
                df_effect["ci"] = np.sqrt(np.square(df_effect["fdd1"] * coeff["fdd1"].SE) \
                                  + np.square(
                    df_effect["fdd1"] * df_effect["fdd1_sc2_sctf"] * coeff["fdd1_sc2_sctf"].SE))
            else:
                df_effect["rank"] = df_effect.groupby("county_fips")[["gdd_effect"]].rank("first", ascending=True)
                # df_effect["rank"] = df_effect.groupby("county_fips")[[coeff[var_name].name]].rank("first", ascending=False)
                df_effect = df_effect[df_effect["rank"] == rank]
                df_effect["effect"] = df_effect[coeff[var_name].name] * coeff[var_name].value
                df_effect["ci"] = np.abs(df_effect[coeff[var_name].name] * coeff[var_name].SE)
            # print(df_effect.head(10))
            df_effect = df_effect.reset_index()
            df_effect = df_effect.sort_values(by=['county_fips', ])
            if df_fips is not None:
                df_effect = pd.merge(df_fips, df_effect, on="county_fips", how="left")
                # df_effect = df_effect.astype({"year": int, })  # ValueError: Cannot convert non-finite values (NA or inf) to integer

            if var_name.startswith("fdd1_sc2"):
                df_effect = df_effect[["county_fips", "effect", "fdd1", "fdd1_sc2_sctf", "fdd1:sc2_sctf", "year", "ci"]]
            elif var_name.startswith("fdd2_sc2"):
                df_effect = df_effect[["county_fips", "effect", "fdd2", "fdd2_sc2_sctf", "fdd1:sc2_sctf", "year", "ci"]]
            else:
                df_effect = df_effect[["county_fips", "effect", coeff[var_name].name, "year", "ci"]]

            output_dir_each = os.path.join(output_dir, scenario)
            output_dir_each = os.path.join(output_dir_each, period_name)
            make_dirs(output_dir_each)

            df_effect.to_csv(os.path.join(output_dir_each, output_filename), index=False)


def make_effect_avg(df, output_dir, df_fips=None):
    variables = [
        "fdd1",
        "gdd1_spring",
        "gdd1_winter",
        "gdd1_fall",
        "gdd2_spring",
        "gdd2_winter",
        "gdd2_fall",
        "gdd3_spring",
        "gdd3_winter",
        "gdd3_fall",
        "prcp_fall",
        "prcp_winter",
        "prcp_spring",
        "snowfall_fall",
        "snowfall_winter",
        "snowfall_spring",
        "fdd1_sc2_sctf",
        "fdd1_sc2_overlap",
    ]

    for variable in variables:
        generate_effect_data(df_all=df,
                             var_name=variable,
                             output_dir=output_dir,
                             output_filename="gridMET_{}_avg.csv".format(variable),
                             df_fips=df_fips)

    for scenario in scenarios:
        for period_name, period_year in periods.items():
            output_dir_each = os.path.join(output_dir, scenario)
            output_dir_each = os.path.join(output_dir_each, period_name)
            aggregate(os.path.join(output_dir_each, "gridMET_{}_avg.csv"), "gdd_all", [x for x in variables if "gdd" in x])
            aggregate(os.path.join(output_dir_each, "gridMET_{}_avg.csv"), "gdd1_all",
                      [x for x in variables if "gdd1" in x])
            aggregate(os.path.join(output_dir_each, "gridMET_{}_avg.csv"), "gdd2_all",
                      [x for x in variables if "gdd2" in x])
            aggregate(os.path.join(output_dir_each, "gridMET_{}_avg.csv"), "gdd3_all",
                      [x for x in variables if "gdd3" in x])
            aggregate(os.path.join(output_dir_each, "gridMET_{}_avg.csv"), "gdd_fall_all",
                      [x for x in variables if "gdd" in x and "fall" in x])
            aggregate(os.path.join(output_dir_each, "gridMET_{}_avg.csv"), "gdd_winter_all",
                      [x for x in variables if "gdd" in x and "winter" in x])
            aggregate(os.path.join(output_dir_each, "gridMET_{}_avg.csv"), "gdd_spring_all",
                      [x for x in variables if "gdd" in x and "spring" in x])
            aggregate(os.path.join(output_dir_each, "gridMET_{}_avg.csv"), "prcp_all", [x for x in variables if "prcp" in x])
            aggregate(os.path.join(output_dir_each, "gridMET_{}_avg.csv"), "snowfall_all", [x for x in variables if "snowfall" in x])

            aggregate(os.path.join(output_dir_each, "gridMET_{}_avg.csv"), "effect_all", [x for x in variables if x != "fdd1_sc2_overlap"])


def make_effect_rank(df, output_dir, df_fips=None):
    variables = [
        "fdd1",
        "gdd1_spring",
        "gdd1_winter",
        "gdd1_fall",
        "gdd2_spring",
        "gdd2_winter",
        "gdd2_fall",
        "gdd3_spring",
        "gdd3_winter",
        "gdd3_fall",
        "prcp_fall",
        "prcp_winter",
        "prcp_spring",
        "snowfall_fall",
        "snowfall_winter",
        "snowfall_spring",
        "fdd1_sc2_sctf",
        "fdd1_sc2_overlap",
    ]
    for variable in variables:
        generate_effect_data_rank(df_all=df,
                                  var_name=variable,
                                  rank_percent=.1,
                                  output_dir=output_dir,
                                  output_filename="gridMET_{}_top10p.csv".format(variable),
                                  df_fips=df_fips)

    for scenario in scenarios:
        for period_name, period_year in periods.items():
            output_dir_each = os.path.join(output_dir, scenario)
            output_dir_each = os.path.join(output_dir_each, period_name)
            aggregate_rank(os.path.join(output_dir_each, "gridMET_{}_top10p.csv"), "gdd_all",
                           [x for x in variables if "gdd" in x])
            aggregate_rank(os.path.join(output_dir_each, "gridMET_{}_top10p.csv"), "gdd1_all",
                           [x for x in variables if "gdd1" in x])
            aggregate_rank(os.path.join(output_dir_each, "gridMET_{}_top10p.csv"), "gdd2_all",
                           [x for x in variables if "gdd2" in x])
            aggregate_rank(os.path.join(output_dir_each, "gridMET_{}_top10p.csv"), "gdd3_all",
                           [x for x in variables if "gdd3" in x])
            aggregate_rank(os.path.join(output_dir_each, "gridMET_{}_top10p.csv"), "gdd_fall_all",
                           [x for x in variables if "gdd" in x and "fall" in x])
            aggregate_rank(os.path.join(output_dir_each, "gridMET_{}_top10p.csv"), "gdd_winter_all",
                           [x for x in variables if "gdd" in x and "winter" in x])
            aggregate_rank(os.path.join(output_dir_each, "gridMET_{}_top10p.csv"), "gdd_spring_all",
                           [x for x in variables if "gdd" in x and "spring" in x])
            aggregate_rank(os.path.join(output_dir_each, "gridMET_{}_top10p.csv"), "prcp_all",
                           [x for x in variables if "prcp" in x])
            aggregate_rank(os.path.join(output_dir_each, "gridMET_{}_top10p.csv"), "snowfall_all",
                           [x for x in variables if "snowfall" in x])
            aggregate_rank(os.path.join(output_dir_each, "gridMET_{}_top10p.csv"), "effect_all",
                           [x for x in variables if x != "fdd1_sc2_overlap"])


def make_effect_rank_gdd(df, output_dir, df_fips=None):
    variables = [
        "fdd1",
        "gdd1_spring",
        "gdd1_winter",
        "gdd1_fall",
        "gdd2_spring",
        "gdd2_winter",
        "gdd2_fall",
        "gdd3_spring",
        "gdd3_winter",
        "gdd3_fall",
        "prcp_fall",
        "prcp_winter",
        "prcp_spring",
        "snowfall_fall",
        "snowfall_winter",
        "snowfall_spring",
        "fdd1_sc2_sctf",
        "fdd1_sc2_overlap",
    ]
    for variable in variables:
        generate_effect_data_rank_gdd(df_all=df,
                                      var_name=variable,
                                      rank_percent=.1,
                                      output_dir=output_dir,
                                      output_filename="gridMET_{}_gdd10p.csv".format(variable),
                                      df_fips=df_fips)

    for scenario in scenarios:
        for period_name, period_year in periods.items():
            output_dir_each = os.path.join(output_dir, scenario)
            output_dir_each = os.path.join(output_dir_each, period_name)
            aggregate_rank(os.path.join(output_dir_each, "gridMET_{}_gdd10p.csv"), "gdd_all",
                           [x for x in variables if "gdd" in x])
            aggregate_rank(os.path.join(output_dir_each, "gridMET_{}_gdd10p.csv"), "gdd1_all",
                           [x for x in variables if "gdd1" in x])
            aggregate_rank(os.path.join(output_dir_each, "gridMET_{}_gdd10p.csv"), "gdd2_all",
                           [x for x in variables if "gdd2" in x])
            aggregate_rank(os.path.join(output_dir_each, "gridMET_{}_gdd10p.csv"), "gdd3_all",
                           [x for x in variables if "gdd3" in x])
            aggregate_rank(os.path.join(output_dir_each, "gridMET_{}_gdd10p.csv"), "gdd_fall_all",
                           [x for x in variables if "gdd" in x and "fall" in x])
            aggregate_rank(os.path.join(output_dir_each, "gridMET_{}_gdd10p.csv"), "gdd_winter_all",
                           [x for x in variables if "gdd" in x and "winter" in x])
            aggregate_rank(os.path.join(output_dir_each, "gridMET_{}_gdd10p.csv"), "gdd_spring_all",
                           [x for x in variables if "gdd" in x and "spring" in x])
            aggregate_rank(os.path.join(output_dir_each, "gridMET_{}_gdd10p.csv"), "prcp_all",
                           [x for x in variables if "prcp" in x])
            aggregate_rank(os.path.join(output_dir_each, "gridMET_{}_gdd10p.csv"), "snowfall_all",
                           [x for x in variables if "snowfall" in x])
            aggregate_rank(os.path.join(output_dir_each, "gridMET_{}_gdd10p.csv"), "effect_all",
                           [x for x in variables if x != "fdd1_sc2_overlap"])


def process(model_name):
    base_dir = os.path.join("N:/WinterWheat/model-input-swe-bc", model_name)
    output_dir = get_project_root() / "output/county_map/v20200726-gdd-a"  # all states
    output_dir = get_project_root() / "output/county_map/v20200730"  # all states + all missing counties
    output_dir = get_project_root() / "output/county_map/v20200806-for-Peng-rawdata"  # all states + all missing counties including variables not only effect
    output_dir = get_project_root() / "output/county_map/v20200814-for-Peng-rawdata"  # fix igdd/ifdd errors
    output_dir = get_project_root() / "output/county_map/v20200927-for-Peng-rawdata"  # add snowfall variables
    output_dir = "N:/WinterWheat/county_map/v20210111-for-Peng-rawdata"  # add more GCMs

    output_dir = os.path.join(output_dir, model_name)
    data_filename = os.path.join(base_dir, "winter_wheat_input_igdd_sdepth_bc_v20210111_{}.csv".format(model_name))

    if not os.path.exists(data_filename):
        print("{} is not exist.".format(data_filename))
        return

    df = pd.read_csv(data_filename)
    df_fips = pd.read_csv(get_project_root() / "input/county_fips_3107.csv")
    make_effect_avg(df, output_dir, df_fips=df_fips)
    make_effect_rank(df, output_dir, df_fips=df_fips)
    make_effect_rank_gdd(df, output_dir, df_fips=df_fips)


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
