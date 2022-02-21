import os

import pandas as pd

from winter_wheat.util import get_project_root, make_dirs


def export(output_filename, gee_export_files):
    base_dir = get_project_root() / "output/US"

    # Load the dataset
    df = pd.read_csv(os.path.join(base_dir, "survey_winter_wheat_yield_all_year.csv"))
    df = pd.melt(df, id_vars=['county_fips'])
    df = df.astype({"county_fips": int, "variable": int, "value": float})
    df = df.rename(columns={"variable": "year", "value": "yield_survey"})
    df = df[df["year"] >= 1999]

    df_prod = pd.read_csv(os.path.join(base_dir, "survey_winter_wheat_production_all_year.csv"))
    df_prod = pd.melt(df_prod, id_vars=['county_fips'])
    df_prod = df_prod.astype({"county_fips": int, "variable": int})
    df_prod["value"] = df_prod["value"].str.replace(",", "").astype(float)
    df_prod = df_prod.rename(columns={"variable": "year", "value": "production_survey"})
    df = pd.merge(df, df_prod, left_on=["county_fips", "year"], right_on=["county_fips", "year"], how="outer")

    df_acre = pd.read_csv(os.path.join(base_dir, "survey_winter_wheat_acre_all_year.csv"))
    df_acre = pd.melt(df_acre, id_vars=['county_fips'])
    df_acre = df_acre.astype({"county_fips": int, "variable": int})
    df_acre["value"] = df_acre["value"].str.replace(",", "").astype(float)
    df_acre = df_acre.rename(columns={"variable": "year", "value": "acre_survey"})
    # print(df_acre.dropna().head(5))
    df = pd.merge(df, df_acre, left_on=["county_fips", "year"], right_on=["county_fips", "year"], how="outer")
    # print(df_acre.dropna().size)

    df_acre_irr = pd.read_csv(os.path.join(base_dir, "survey_winter_wheat_acre_irrigated_all_year.csv"))
    df_acre_irr = pd.melt(df_acre_irr, id_vars=['county_fips'])
    df_acre_irr = df_acre_irr.astype({"county_fips": int, "variable": int})
    df_acre_irr["value"] = df_acre_irr["value"].str.replace(",", "").astype(float)
    # df_prod = df_prod.dropna()
    df_acre_irr = df_acre_irr.rename(columns={"variable": "year", "value": "acre_irrigated_survey"})
    df = pd.merge(df, df_acre_irr, left_on=["county_fips", "year"], right_on=["county_fips", "year"], how="outer")
    df["irr_acre_percent"] = df["acre_irrigated_survey"] / df["acre_survey"]

    df = df[df["year"] >= 1999]

    model_vars = [
        "fdd1", "fdd2", "fdd3",
        "fdd1_sc2", "fdd1_sc5", "fdd1_sc10",
        "fdd2_sc2", "fdd2_sc5", "fdd2_sc10",
        "fdd3_sc2", "fdd3_sc5", "fdd3_sc10",
        "fdd1_day", "fdd2_day", "fdd3_day",
        "fdd1_sc2_sctf", "fdd1_sc5_sctf", "fdd1_sc10_sctf",
        "fdd2_sc2_sctf", "fdd2_sc5_sctf", "fdd2_sc10_sctf",
        "fdd3_sc2_sctf", "fdd3_sc5_sctf", "fdd3_sc10_sctf",
        "gdd_low_fall", "gdd_med_fall", "gdd_high_fall", "prcp_fall", "snowfall_fall",
        "gdd_low_winter", "gdd_med_winter", "gdd_high_winter", "prcp_winter", "snowfall_winter",
        "gdd_low_spring", "gdd_med_spring", "gdd_high_spring", "prcp_spring", "snowfall_spring",
    ]
    gee_export_field = ["GEOID", "index"]
    rename_columns = {
        "GEOID": "county_fips",
        "index": "year",
    }

    for i in range(len(model_vars)):
        gee_export_field.append("b{}".format(i+1))
        rename_columns["b{}".format(i+1)] = model_vars[i]

    df_var = None
    for gee_export_file in gee_export_files:
        df_var_each_file = pd.read_csv(gee_export_file, usecols=gee_export_field, dtype={"GEOID": "int64", "index": "int64",})
        df_var_each_file = df_var_each_file.rename(columns=rename_columns)
        df_var_each_file = df_var_each_file[["county_fips", "year"] + model_vars]
        if df_var is None:
            df_var = df_var_each_file
        else:
            df_var = pd.concat([df_var, df_var_each_file], ignore_index=True)

    df = pd.merge(df, df_var, left_on=["county_fips", "year"], right_on=["county_fips", "year"], how="right")

    df["gdd_low_fall"] = df["gdd_low_fall"] - df["gdd_med_fall"]
    df["gdd_med_fall"] = df["gdd_med_fall"] - df["gdd_high_fall"]
    df["gdd_low_winter"] = df["gdd_low_winter"] - df["gdd_med_winter"]
    df["gdd_med_winter"] = df["gdd_med_winter"] - df["gdd_high_winter"]
    df["gdd_low_spring"] = df["gdd_low_spring"] - df["gdd_med_spring"]
    df["gdd_med_spring"] = df["gdd_med_spring"] - df["gdd_high_spring"]

    df = df.sort_values(by=["county_fips", "year",])

    df.to_csv(output_filename, index=False)
    # field_dfwos = ["GEOID", "year", "b1_mean"]
    # df_dfwos = pd.read_csv("C:/DATA/WinterWheat/county-dfwos-all.csv", usecols=field_dfwos, dtype={"GEOID": "int64", "year": "int64", "b1_mean": "float64"})
    # df_dfwos = df_dfwos.rename(columns={"GEOID": "county_fips", "b1_mean": "dfwos"})


if __name__ == "__main__":
    output_dir = get_project_root() / "output/gridUS"
    make_dirs(output_dir)

    # export(
    #     os.path.join(output_dir, "winter_wheat_input_ifdd_igdd_v20200814_gridUS.csv"),
    #     ["C:/DATA/WinterWheat/gridUS/export_from_gee/county-grid-iGDD-US-v20200814-1999-2019.csv",]
    # )

    export(
        os.path.join(output_dir, "winter_wheat_input_ifdd_igdd_v20200927_gridUS.csv"),
        ["C:/DATA/WinterWheat/gridUS/export_from_gee/county-grid-iGDD-US-v20200927-1999-2019.csv",]
    )
