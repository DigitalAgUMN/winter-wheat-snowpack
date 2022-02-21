import os
from typing import NamedTuple

import numpy as np
import pandas as pd

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


def generate_effect_data(dataframe, var_name, output_filename, df_fips=None):
    # county_fips,year,yield_survey,production_survey,acre_survey,acre_irrigated_survey,
    # irr_acre_percent,fdd1,fdd2,fdd3,fdd1_sc2,fdd1_sc5,fdd1_sc10,fdd2_sc2,fdd2_sc5,fdd2_sc10,
    # fdd3_sc2,fdd3_sc5,fdd3_sc10,fdd1_day,fdd2_day,fdd3_day,
    # fdd1_sc2_sctf,fdd1_sc5_sctf,fdd1_sc10_sctf,
    # fdd2_sc2_sctf,fdd2_sc5_sctf,fdd2_sc10_sctf,
    # fdd3_sc2_sctf,fdd3_sc5_sctf,fdd3_sc10_sctf,
    # gdd_low_fall,gdd_med_fall,gdd_high_fall,prcp_fall,
    # gdd_low_winter,gdd_med_winter,gdd_high_winter,prcp_winter,
    # gdd_low_spring,gdd_med_spring,gdd_high_spring,prcp_spring
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
        df_effect["ci"] = np.sqrt(np.square(df_effect["fdd1"] * coeff["fdd1"].SE)\
                              + np.square(df_effect["fdd1"] * df_effect["fdd1_sc2_sctf"] * coeff["fdd1_sc2_sctf"].SE))
    else:
        df_effect = dataframe.groupby(["county_fips"], as_index=False)[coeff[var_name].name].mean()
        df_effect["effect"] = df_effect[coeff[var_name].name] * coeff[var_name].value
        df_effect["ci"] = np.abs(df_effect[coeff[var_name].name] * coeff[var_name].SE)
    # print(df_effect.head(10))
    df_effect = df_effect.reset_index()
    df_effect = df_effect.sort_values(by=['county_fips', ])

    if df_fips is not None:
        df_effect = pd.merge(df_fips, df_effect, on="county_fips", how="left")

    if var_name.startswith("fdd1_sc2"):
        df_effect = df_effect[["county_fips", "effect", "fdd1", "fdd1_sc2_sctf", "fdd1:sc2_sctf", "ci"]]
    else:
        df_effect = df_effect[["county_fips", "effect", coeff[var_name].name, "ci"]]
    df_effect.to_csv(output_filename, index=False)


def generate_effect_data_rank(dataframe, var_name, rank, output_filename, df_fips=None):
    # county_fips,year,yield_survey,production_survey,acre_survey,acre_irrigated_survey,
    # irr_acre_percent,fdd1,fdd2,fdd3,fdd1_sc2,fdd1_sc5,fdd1_sc10,fdd2_sc2,fdd2_sc5,fdd2_sc10,
    # fdd3_sc2,fdd3_sc5,fdd3_sc10,fdd1_day,fdd2_day,fdd3_day,
    # fdd1_sc2_sctf,fdd1_sc5_sctf,fdd1_sc10_sctf,
    # fdd2_sc2_sctf,fdd2_sc5_sctf,fdd2_sc10_sctf,
    # fdd3_sc2_sctf,fdd3_sc5_sctf,fdd3_sc10_sctf,
    # gdd_low_fall,gdd_med_fall,gdd_high_fall,prcp_fall,
    # gdd_low_winter,gdd_med_winter,gdd_high_winter,prcp_winter,
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
        df_effect["ci"] = np.sqrt(np.square(df_effect["fdd1"] * coeff["fdd1"].SE)\
                              + np.square(df_effect["fdd1"] * df_effect["fdd1_sc2_sctf"] * coeff["fdd1_sc2_sctf"].SE))
    else:
        df_effect = dataframe.copy()
        df_effect["rank"] = df_effect.groupby("county_fips")[["fdd1"]].rank("first", ascending=False)
        # df_effect["rank"] = df_effect.groupby("county_fips")[[coeff[var_name].name]].rank("first", ascending=False)
        df_effect = df_effect[df_effect["rank"] == rank]
        df_effect["effect"] = df_effect[coeff[var_name].name] * coeff[var_name].value
        df_effect["ci"] = np.abs(df_effect[coeff[var_name].name] * coeff[var_name].SE)
    # if var_name == "gdd3_spring":
    #     print(df_effect[df_effect["county_fips"] == 27001].head(50))
    #
    # if var_name == "gdd3_spring":
    #     print(df_effect[df_effect["county_fips"] == 6059].head(50))  # 6059, 6075

    df_effect = df_effect.reset_index()
    df_effect = df_effect.sort_values(by=['county_fips', ])

    if df_fips is not None:
        df_effect = pd.merge(df_fips, df_effect, on="county_fips", how="left")

    if var_name.startswith("fdd1_sc2"):
        df_effect = df_effect[["county_fips", "effect", "fdd1", "fdd1_sc2_sctf", "fdd1:sc2_sctf", "year", "ci"]]
    elif var_name.startswith("fdd2_sc2"):
        df_effect = df_effect[["county_fips", "effect", "fdd2", "fdd2_sc2_sctf", "year", "ci"]]
    else:
        df_effect = df_effect[["county_fips", "effect", coeff[var_name].name, "year", "ci"]]
    df_effect.to_csv(output_filename, index=False)


def generate_effect_data_rank_gdd(dataframe, var_name, rank, output_filename, df_fips=None):
    df_effect = dataframe.copy()
    # df_effect["gdd_effect"] = \
    #     df_effect["fdd1"] * coeff["fdd1"].value \
    #     + df_effect[coeff["gdd1_spring"].name] * coeff["gdd1_spring"].value \
    #     + df_effect[coeff["gdd1_winter"].name] * coeff["gdd1_winter"].value \
    #     + df_effect[coeff["gdd1_fall"].name] * coeff["gdd1_fall"].value \
    #     + df_effect[coeff["gdd2_spring"].name] * coeff["gdd2_spring"].value \
    #     + df_effect[coeff["gdd2_winter"].name] * coeff["gdd2_winter"].value \
    #     + df_effect[coeff["gdd2_fall"].name] * coeff["gdd2_fall"].value \
    #     + df_effect[coeff["gdd3_spring"].name] * coeff["gdd3_spring"].value \
    #     + df_effect[coeff["gdd3_winter"].name] * coeff["gdd3_winter"].value \
    #     + df_effect[coeff["gdd3_fall"].name] * coeff["gdd3_fall"].value \
    #     + df_effect["prcp_fall"] * coeff["prcp_fall"].value \
    #     + df_effect["prcp_winter"] * coeff["prcp_winter"].value \
    #     + df_effect["prcp_spring"] * coeff["prcp_spring"].value \
    #     + df_effect["fdd1"] * df_effect["fdd1_sc2_sctf"] * coeff["fdd1_sc2_sctf"].value
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
        df_effect["ci"] = np.sqrt(np.square(df_effect["fdd1"] * coeff["fdd1"].SE)\
                              + np.square(df_effect["fdd1"] * df_effect["fdd1_sc2_sctf"] * coeff["fdd1_sc2_sctf"].SE))
    else:
        df_effect["rank"] = df_effect.groupby("county_fips")[["gdd_effect"]].rank("first", ascending=True)
        # df_effect["rank"] = df_effect.groupby("county_fips")[[coeff[var_name].name]].rank("first", ascending=False)
        df_effect = df_effect[df_effect["rank"] == rank]
        df_effect["effect"] = df_effect[coeff[var_name].name] * coeff[var_name].value
        df_effect["ci"] = np.abs(df_effect[coeff[var_name].name] * coeff[var_name].SE)

    # if var_name == "gdd3_spring":
    #     print(df_effect[df_effect["county_fips"] == 27001].head(50))
    #
    # if var_name == "gdd3_spring":
    #     print(df_effect[df_effect["county_fips"] == 6059].head(50))  # 6059, 6075

    df_effect = df_effect.reset_index()
    df_effect = df_effect.sort_values(by=['county_fips', ])

    if df_fips is not None:
        df_effect = pd.merge(df_fips, df_effect, on="county_fips", how="left")

    if var_name.startswith("fdd1_sc2"):
        df_effect = df_effect[["county_fips", "effect", "fdd1", "fdd1_sc2_sctf", "fdd1:sc2_sctf", "year", "ci"]]
    else:
        df_effect = df_effect[["county_fips", "effect", coeff[var_name].name, "year", "ci"]]
    df_effect.to_csv(output_filename, index=False)


def aggregate(file_pattern, output_var, elements):
    if output_var == "effect_all":
        df = None
        for element in elements:
            df_el = pd.read_csv(file_pattern.format(element))[["county_fips", "effect", "ci"]]
            if df is None:
                df = df_el
            else:
                df = df.merge(df_el, on="county_fips")
                # frame.a.fillna(0) + frame.b.fillna(0)
                df["effect"] = df["effect_x"].fillna(0) + df["effect_y"].fillna(0)
                df["ci"] = np.sqrt(np.square(df["ci_x"].fillna(0)) + np.square(df["ci_y"].fillna(0)))
                df = df[["county_fips", "effect", "ci"]].copy()
                if output_var == "effect_all":
                    print(df[df["county_fips"] == 12087])
        df.to_csv(file_pattern.format(output_var), index=False)
    elif output_var == "fdd1_sc2_overlap":
        df = None
        for element in elements:
            df_el = pd.read_csv(file_pattern.format(element))[["county_fips", "fdd1", "fdd1_sc2_sctf", "fdd1:sc2_sctf", "effect", "ci"]]
            if df is None:
                df = df_el
            else:
                df = df.merge(df_el, on="county_fips")
                # frame.a.fillna(0) + frame.b.fillna(0)
                df["effect"] = df["effect_x"].fillna(0) + df["effect_y"].fillna(0)
                df["ci"] = np.sqrt(np.square(df["ci_x"].fillna(0)) + np.square(df["ci_y"].fillna(0)))
                df["fdd1"] = df["fdd1_x"].fillna(0) + df["fdd1_y"].fillna(0)
                df["fdd1_sc2_sctf"] = df["fdd1_sc2_sctf_x"].fillna(0) + df["fdd1_sc2_sctf_y"].fillna(0)
                df["fdd1:sc2_sctf"] = df["fdd1:sc2_sctf_x"].fillna(0) + df["fdd1:sc2_sctf_y"].fillna(0)
                df = df[["county_fips", "effect", "fdd1", "fdd1_sc2_sctf", "fdd1:sc2_sctf", "ci"]].copy()
        df.to_csv(file_pattern.format(output_var), index=False)
    elif output_var.endswith("_all"):
        df = None
        for element in elements:
            df_el = pd.read_csv(file_pattern.format(element))[["county_fips", coeff[element].name, "effect", "ci"]]
            if df is None:
                df = df_el
                df[output_var] = df_el[coeff[element].name]
            else:
                df = df.merge(df_el, on="county_fips")
                # frame.a.fillna(0) + frame.b.fillna(0)
                df["effect"] = df["effect_x"].fillna(0) + df["effect_y"].fillna(0)
                df["ci"] = np.sqrt(np.square(df["ci_x"].fillna(0)) + np.square(df["ci_y"].fillna(0)))
                df[output_var] = df[output_var].fillna(0) + df[coeff[element].name].fillna(0)
                df = df[["county_fips", "effect", output_var, "ci"]].copy()
        df.to_csv(file_pattern.format(output_var), index=False)
    else:
        df = None
        for element in elements:
            df_el = pd.read_csv(file_pattern.format(element))[["county_fips", coeff[element].name, "effect", "ci"]]
            if df is None:
                df = df_el
                df[coeff[element].name] = df_el[coeff[element].name]
            else:
                df = df.merge(df_el, on="county_fips")
                # frame.a.fillna(0) + frame.b.fillna(0)
                df["effect"] = df["effect_x"].fillna(0) + df["effect_y"].fillna(0)
                df["ci"] = np.sqrt(np.square(df["ci_x"].fillna(0)) + np.square(df["ci_y"].fillna(0)))
                df[coeff[element].name] = df[coeff[element].name+"_x"].fillna(0) + df[coeff[element].name+"_y"].fillna(0)
                df = df[["county_fips", "effect", coeff[element].name, "ci"]].copy()
        df.to_csv(file_pattern.format(output_var), index=False)


def aggregate_rank(file_pattern, output_var, elements):
    if output_var == "effect_all":
        df = None
        for element in elements:
            df_el = pd.read_csv(file_pattern.format(element))[["county_fips", "effect", "year", "ci"]]
            if df is None:
                df = df_el
            else:
                df = df.merge(df_el, on="county_fips")
                df["effect"] = df["effect_x"].fillna(0) + df["effect_y"].fillna(0)
                df["ci"] = np.sqrt(np.square(df["ci_x"].fillna(0)) + np.square(df["ci_y"].fillna(0)))
                df["year"] = df["year_x"]
                if (df["year_x"] - df["year_y"]).sum() != 0:
                    print("ERROR: Year in top10p files should have the same year among files.")
                    print(df[df["year_x"] != df["year_y"]])
                # print(df.head())
                df = df[["county_fips", "effect", "year", "ci"]].copy()
        df.to_csv(file_pattern.format(output_var), index=False, columns=["county_fips", "effect", "year", "ci"])
    elif output_var == "fdd1_sc2_overlap":
        df = None
        for element in elements:
            df_el = pd.read_csv(file_pattern.format(element))[["county_fips", "fdd1", "fdd1_sc2_sctf", "fdd1:sc2_sctf", "effect", "year", "ci"]]
            if df is None:
                df = df_el
            else:
                df = df.merge(df_el, on="county_fips")
                df["effect"] = df["effect_x"].fillna(0) + df["effect_y"].fillna(0)
                df["ci"] = np.sqrt(np.square(df["ci_x"].fillna(0)) + np.square(df["ci_y"].fillna(0)))

                df["year"] = df["year_x"]
                if (df["year_x"] - df["year_y"]).sum() != 0:
                    print("ERROR: Year in top10p files should have the same year among files.")
                    print(df[df["year_x"] != df["year_y"]])
                # print(df.head())
                df = df[["county_fips", "effect", "year", "fdd1", "fdd1_sc2_sctf", "fdd1:sc2_sctf", "ci"]].copy()
        df.to_csv(file_pattern.format(output_var), index=False, columns=["county_fips", "effect", "year", "fdd1", "fdd1_sc2_sctf", "fdd1:sc2_sctf", "ci"])
    elif output_var.endswith("_all"):
        df = None
        for element in elements:
            df_el = pd.read_csv(file_pattern.format(element))[["county_fips", coeff[element].name, "effect", "year", "ci"]]
            if df is None:
                df = df_el
                df[output_var] = df[coeff[element].name]
            else:
                df = df.merge(df_el, on="county_fips")
                df["effect"] = df["effect_x"].fillna(0) + df["effect_y"].fillna(0)
                df["ci"] = np.sqrt(np.square(df["ci_x"].fillna(0)) + np.square(df["ci_y"].fillna(0)))
                df[output_var] = df[output_var].fillna(0) + df[coeff[element].name].fillna(0)

                df["year"] = df["year_x"]
                if (df["year_x"] - df["year_y"]).sum() != 0:
                    print("ERROR: Year in top10p files should have the same year among files.")
                    print(df[df["year_x"] != df["year_y"]])
                # print(df.head())
                df = df[["county_fips", "effect", "year", output_var, "ci"]].copy()
        df.to_csv(file_pattern.format(output_var), index=False, columns=["county_fips", "effect", "year", output_var, "ci"])
    else:
        df = None
        for element in elements:
            df_el = pd.read_csv(file_pattern.format(element))[["county_fips", coeff[element].name, "effect", "year", "ci"]]
            if df is None:
                df = df_el
            else:
                df = df.merge(df_el, on="county_fips")
                df["effect"] = df["effect_x"].fillna(0) + df["effect_y"].fillna(0)
                df["ci"] = np.sqrt(np.square(df["ci_x"].fillna(0)) + np.square(df["ci_y"].fillna(0)))
                df[coeff[element].name] = df[coeff[element].name+"_x"].fillna(0) + df[coeff[element].name + "_y"].fillna(0)
                df["year"] = df["year_x"]
                if (df["year_x"] - df["year_y"]).sum() != 0:
                    print("ERROR: Year in top10p files should have the same year among files.")
                    print(df[df["year_x"] != df["year_y"]])
                # print(df.head())
                df = df[["county_fips", "effect", "year", coeff[element].name, "ci"]].copy()
        df.to_csv(file_pattern.format(output_var), index=False, columns=["county_fips", "effect", "year", coeff[element].name, "ci"])


if __name__ == "__main__":
    # base_dir = get_project_root() / "output/gridUS"
    base_dir = get_project_root() / "output/gridUS/2.5_arc_min"
    # output_dir = get_project_root() / "output/county_map"
    # output_dir = get_project_root() / "output/county_map/v20200806-for-Peng-rawdata"  # all states + all missing counties including variables not only effect
    # output_dir = get_project_root() / "output/county_map/v20200814-for-Peng-rawdata"  # fix igdd/ifdd errors
    output_dir = get_project_root() / "output/county_map/v20200927-for-Peng-rawdata"  # add snowfall variables
    output_dir = "N:/WinterWheat/county_map/v20210111-for-Peng-rawdata"  # add more GCMs

    make_dirs(output_dir)

    # df = pd.read_csv(os.path.join(base_dir, "winter_wheat_input_ifdd_igdd_v20200814_gridUS.csv"))
    df = pd.read_csv(os.path.join(base_dir, "winter_wheat_input_ifdd_igdd_v20210107_gridUS_4km.csv"))
    df_fips = pd.read_csv(get_project_root() / "input/county_fips_3107.csv")

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
        generate_effect_data(dataframe=df,
                             var_name=variable,
                             output_filename=os.path.join(output_dir, "gridMET_{}_avg.csv".format(variable)),
                             df_fips=df_fips)

    for variable in variables:
        generate_effect_data_rank(dataframe=df,
                                  var_name=variable,
                                  rank=2,
                                  output_filename=os.path.join(output_dir, "gridMET_{}_top10p.csv".format(variable)),
                                  df_fips=df_fips)

    for variable in variables:
        generate_effect_data_rank_gdd(dataframe=df,
                                      var_name=variable,
                                      rank=2,
                                      output_filename=os.path.join(output_dir,
                                                                   "gridMET_{}_gdd10p.csv".format(variable)),
                                      df_fips=df_fips)

    aggregate(os.path.join(output_dir, "gridMET_{}_avg.csv"), "gdd_all", [x for x in variables if "gdd" in x])
    aggregate(os.path.join(output_dir, "gridMET_{}_avg.csv"), "gdd1_all", [x for x in variables if "gdd1" in x])
    aggregate(os.path.join(output_dir, "gridMET_{}_avg.csv"), "gdd2_all", [x for x in variables if "gdd2" in x])
    aggregate(os.path.join(output_dir, "gridMET_{}_avg.csv"), "gdd3_all", [x for x in variables if "gdd3" in x])
    aggregate(os.path.join(output_dir, "gridMET_{}_avg.csv"), "gdd_fall_all", [x for x in variables if "gdd" in x and "fall" in x])
    aggregate(os.path.join(output_dir, "gridMET_{}_avg.csv"), "gdd_winter_all", [x for x in variables if "gdd" in x and "winter" in x])
    aggregate(os.path.join(output_dir, "gridMET_{}_avg.csv"), "gdd_spring_all", [x for x in variables if "gdd" in x and "spring" in x])
    aggregate(os.path.join(output_dir, "gridMET_{}_avg.csv"), "prcp_all", [x for x in variables if "prcp" in x])
    aggregate(os.path.join(output_dir, "gridMET_{}_avg.csv"), "snowfall_all", [x for x in variables if "snowfall" in x])
    aggregate(os.path.join(output_dir, "gridMET_{}_avg.csv"), "effect_all", [x for x in variables if x != "fdd1_sc2_overlap"])

    aggregate_rank(os.path.join(output_dir, "gridMET_{}_top10p.csv"), "gdd_all", [x for x in variables if "gdd" in x])
    aggregate_rank(os.path.join(output_dir, "gridMET_{}_top10p.csv"), "gdd1_all", [x for x in variables if "gdd1" in x])
    aggregate_rank(os.path.join(output_dir, "gridMET_{}_top10p.csv"), "gdd2_all", [x for x in variables if "gdd2" in x])
    aggregate_rank(os.path.join(output_dir, "gridMET_{}_top10p.csv"), "gdd3_all", [x for x in variables if "gdd3" in x])
    aggregate_rank(os.path.join(output_dir, "gridMET_{}_top10p.csv"), "gdd_fall_all", [x for x in variables if "gdd" in x and "fall" in x])
    aggregate_rank(os.path.join(output_dir, "gridMET_{}_top10p.csv"), "gdd_winter_all", [x for x in variables if "gdd" in x and "winter" in x])
    aggregate_rank(os.path.join(output_dir, "gridMET_{}_top10p.csv"), "gdd_spring_all", [x for x in variables if "gdd" in x and "spring" in x])
    aggregate_rank(os.path.join(output_dir, "gridMET_{}_top10p.csv"), "prcp_all", [x for x in variables if "prcp" in x])
    aggregate_rank(os.path.join(output_dir, "gridMET_{}_top10p.csv"), "snowfall_all", [x for x in variables if "snowfall" in x])
    aggregate_rank(os.path.join(output_dir, "gridMET_{}_top10p.csv"), "effect_all", [x for x in variables if x != "fdd1_sc2_overlap"])

    aggregate_rank(os.path.join(output_dir, "gridMET_{}_gdd10p.csv"), "gdd_all", [x for x in variables if "gdd" in x])
    aggregate_rank(os.path.join(output_dir, "gridMET_{}_gdd10p.csv"), "gdd1_all", [x for x in variables if "gdd1" in x])
    aggregate_rank(os.path.join(output_dir, "gridMET_{}_gdd10p.csv"), "gdd2_all", [x for x in variables if "gdd2" in x])
    aggregate_rank(os.path.join(output_dir, "gridMET_{}_gdd10p.csv"), "gdd3_all", [x for x in variables if "gdd3" in x])
    aggregate_rank(os.path.join(output_dir, "gridMET_{}_gdd10p.csv"), "gdd_fall_all", [x for x in variables if "gdd" in x and "fall" in x])
    aggregate_rank(os.path.join(output_dir, "gridMET_{}_gdd10p.csv"), "gdd_winter_all", [x for x in variables if "gdd" in x and "winter" in x])
    aggregate_rank(os.path.join(output_dir, "gridMET_{}_gdd10p.csv"), "gdd_spring_all", [x for x in variables if "gdd" in x and "spring" in x])
    aggregate_rank(os.path.join(output_dir, "gridMET_{}_gdd10p.csv"), "prcp_all", [x for x in variables if "prcp" in x])
    aggregate_rank(os.path.join(output_dir, "gridMET_{}_gdd10p.csv"), "snowfall_all", [x for x in variables if "snowfall" in x])
    aggregate_rank(os.path.join(output_dir, "gridMET_{}_gdd10p.csv"), "effect_all", [x for x in variables if x != "fdd1_sc2_overlap"])
