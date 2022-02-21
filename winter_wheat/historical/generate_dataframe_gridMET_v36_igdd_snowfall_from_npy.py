import os
import sys

import pandas as pd
import numpy as np
import xarray as xr
import tqdm

from winter_wheat.util import get_project_root, make_dirs

base_drive = "N:/gridMET"
base_fdd_drive = "N:/gridMET"

if sys.platform == "linux" or sys.platform == "linux2":  # MSI
    base_drive = "/home/jinzn/taegon/WinterWheat/gridMET"
base_npy_dir = os.path.join(base_drive, "gridUS_annual_npy_20200927")
base_fdd_npy_dir = os.path.join(base_fdd_drive, "gridUS_annual_npy_20210107_sd1520-MSI")


def export_snowfall(output_filename, fdd_filepattern, gdd_filepattern):
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

    df_yield_rainfed = pd.read_csv(os.path.join(base_dir, "survey_winter_wheat_yield_rainfed_all_year.csv"))
    df_yield_rainfed = pd.melt(df_yield_rainfed, id_vars=['county_fips'])
    df_yield_rainfed = df_yield_rainfed.astype({"county_fips": int, "variable": int})
    # df_yield_rainfed["value"] = df_yield_rainfed["value"].str.replace(",", "").astype(float)
    df_yield_rainfed = df_yield_rainfed.rename(columns={"variable": "year", "value": "yield_rainfed_survey"})
    df_yield_rainfed = df_yield_rainfed[df_yield_rainfed["yield_rainfed_survey"] > 0]
    df = pd.merge(df, df_yield_rainfed, left_on=["county_fips", "year"], right_on=["county_fips", "year"], how="outer")

    df_prod_rainfed = pd.read_csv(os.path.join(base_dir, "survey_winter_wheat_production_rainfed_all_year.csv"))
    df_prod_rainfed = pd.melt(df_prod_rainfed, id_vars=['county_fips'])
    df_prod_rainfed = df_prod_rainfed.astype({"county_fips": int, "variable": int})
    # df_prod_rainfed["value"] = df_prod_rainfed["value"].str.replace(",", "").astype(float)
    df_prod_rainfed = df_prod_rainfed.rename(columns={"variable": "year", "value": "production_rainfed_survey"})
    df = pd.merge(df, df_prod_rainfed, left_on=["county_fips", "year"], right_on=["county_fips", "year"], how="outer")

    df_acre_rainfed = pd.read_csv(os.path.join(base_dir, "survey_winter_wheat_acre_rainfed_all_year.csv"))
    df_acre_rainfed = pd.melt(df_acre_rainfed, id_vars=['county_fips'])
    df_acre_rainfed = df_acre_rainfed.astype({"county_fips": int, "variable": int})
    df_acre_rainfed["value"] = df_acre_rainfed["value"].str.replace(",", "").astype(float)
    df_acre_rainfed = df_acre_rainfed.rename(columns={"variable": "year", "value": "acre_rainfed_survey"})
    df = pd.merge(df, df_acre_rainfed, left_on=["county_fips", "year"], right_on=["county_fips", "year"], how="outer")

    df = df[df["year"] >= 1999]

    # for i in range(len(model_vars)):
    #     gee_export_field.append("b{}".format(i+1))
    #     rename_columns["b{}".format(i+1)] = model_vars[i]

    # df_var = None
    # for gee_export_file in gee_export_files:
    #     df_var_each_file = pd.read_csv(gee_export_file, usecols=gee_export_field, dtype={"GEOID": "int64", "index": "int64",})
    #     df_var_each_file = df_var_each_file.rename(columns=rename_columns)
    #     df_var_each_file = df_var_each_file[["county_fips", "year"] + model_vars]
    #     if df_var is None:
    #         df_var = df_var_each_file
    #     else:
    #         df_var = pd.concat([df_var, df_var_each_file], ignore_index=True)
    #
    # df = pd.merge(df, df_var, left_on=["county_fips", "year"], right_on=["county_fips", "year"], how="right")

    fips_array = xr.open_rasterio(os.path.join(get_project_root() / "input/raster", "county_FIPS_gridMET.tif"))
    fips_array = fips_array.to_masked_array()[0]
    fips_array_flatten = fips_array.flatten()
    # print(fips_array.shape)
    # return
    # np.savetxt(os.path.join(get_project_root() / "output/cc_swe_all", "test_fips_us.csv"), fips_array,
    #            delimiter=",")

    df_weather = None

    for year in tqdm.tqdm(range(1999, 2020), desc=output_filename):
    # for year in range(2004, 2005):
        array_fdd = np.load(os.path.join(base_fdd_npy_dir, fdd_filepattern.format(year)))["fdd_sctf"]
        array_gdd = np.load(os.path.join(base_npy_dir, gdd_filepattern.format(year)))["gdd"]
        weather_dataset = np.concatenate((array_fdd, array_gdd), axis=0)
        weather_dataset[weather_dataset == -9999] = np.nan
        weather_dataset[weather_dataset < -9999] = np.nan

        # df_fdd_gdd_raw = pd.DataFrame({
        #     "county_fips": fips_array_flatten,
        #     "fdd1": weather_dataset[0].flatten(),
        #     "fdd2": weather_dataset[1].flatten(),
        #     "fdd3": weather_dataset[2].flatten(),
        #     "fdd1_sc2": weather_dataset[3].flatten(),
        #     "fdd1_sc5": weather_dataset[4].flatten(),
        #     "fdd1_sc10": weather_dataset[5].flatten(),
        #     "fdd2_sc2": weather_dataset[6].flatten(),
        #     "fdd2_sc5": weather_dataset[7].flatten(),
        #     "fdd2_sc10": weather_dataset[8].flatten(),
        #     "fdd3_sc2": weather_dataset[9].flatten(),
        #     "fdd3_sc5": weather_dataset[10].flatten(),
        #     "fdd3_sc10": weather_dataset[11].flatten(),
        #     "fdd1_day": weather_dataset[12].flatten(),
        #     "fdd2_day": weather_dataset[13].flatten(),
        #     "fdd3_day": weather_dataset[14].flatten(),
        #     "fdd1_sc2_sctf": weather_dataset[15].flatten(),
        #     "fdd1_sc5_sctf": weather_dataset[16].flatten(),
        #     "fdd1_sc10_sctf": weather_dataset[17].flatten(),
        #     "fdd2_sc2_sctf": weather_dataset[18].flatten(),
        #     "fdd2_sc5_sctf": weather_dataset[19].flatten(),
        #     "fdd2_sc10_sctf": weather_dataset[20].flatten(),
        #     "fdd3_sc2_sctf": weather_dataset[21].flatten(),
        #     "fdd3_sc5_sctf": weather_dataset[22].flatten(),
        #     "fdd3_sc10_sctf": weather_dataset[23].flatten(),
        #     "gdd_low_fall": weather_dataset[24].flatten(),
        #     "gdd_med_fall": weather_dataset[25].flatten(),
        #     "gdd_high_fall": weather_dataset[26].flatten(),
        #     "prcp_fall": weather_dataset[28].flatten(),
        #     "snowfall_fall": weather_dataset[27].flatten(),
        #     "gdd_low_winter": weather_dataset[29].flatten(),
        #     "gdd_med_winter": weather_dataset[30].flatten(),
        #     "gdd_high_winter": weather_dataset[31].flatten(),
        #     "prcp_winter": weather_dataset[33].flatten(),
        #     "snowfall_winter": weather_dataset[32].flatten(),
        #     "gdd_low_spring": weather_dataset[34].flatten(),
        #     "gdd_med_spring": weather_dataset[35].flatten(),
        #     "gdd_high_spring": weather_dataset[36].flatten(),
        #     "prcp_spring": weather_dataset[38].flatten(),
        #     "snowfall_spring": weather_dataset[37].flatten(),
        # })
        df_fdd_gdd_raw = pd.DataFrame({
            "county_fips": fips_array_flatten,
            "fdd1": weather_dataset[0].flatten(),
            "fdd2": weather_dataset[1].flatten(),
            "fdd3": weather_dataset[2].flatten(),
            "fdd1_sc2": weather_dataset[3].flatten(),
            "fdd1_sc5": weather_dataset[4].flatten(),
            "fdd1_sc10": weather_dataset[5].flatten(),
            "fdd1_sc15": weather_dataset[6].flatten(),
            "fdd1_sc20": weather_dataset[7].flatten(),
            "fdd2_sc2": weather_dataset[8].flatten(),
            "fdd2_sc5": weather_dataset[9].flatten(),
            "fdd2_sc10": weather_dataset[10].flatten(),
            "fdd2_sc15": weather_dataset[11].flatten(),
            "fdd2_sc20": weather_dataset[12].flatten(),
            "fdd3_sc2": weather_dataset[13].flatten(),
            "fdd3_sc5": weather_dataset[14].flatten(),
            "fdd3_sc10": weather_dataset[15].flatten(),
            "fdd3_sc15": weather_dataset[16].flatten(),
            "fdd3_sc20": weather_dataset[17].flatten(),
            "fdd1_day": weather_dataset[18].flatten(),
            "fdd2_day": weather_dataset[19].flatten(),
            "fdd3_day": weather_dataset[20].flatten(),
            "fdd1_sc2_sctf": weather_dataset[21].flatten(),
            "fdd1_sc5_sctf": weather_dataset[22].flatten(),
            "fdd1_sc10_sctf": weather_dataset[23].flatten(),
            "fdd1_sc15_sctf": weather_dataset[24].flatten(),
            "fdd1_sc20_sctf": weather_dataset[25].flatten(),
            "fdd2_sc2_sctf": weather_dataset[26].flatten(),
            "fdd2_sc5_sctf": weather_dataset[27].flatten(),
            "fdd2_sc10_sctf": weather_dataset[28].flatten(),
            "fdd2_sc15_sctf": weather_dataset[29].flatten(),
            "fdd2_sc20_sctf": weather_dataset[30].flatten(),
            "fdd3_sc2_sctf": weather_dataset[31].flatten(),
            "fdd3_sc5_sctf": weather_dataset[32].flatten(),
            "fdd3_sc10_sctf": weather_dataset[33].flatten(),
            "fdd3_sc15_sctf": weather_dataset[34].flatten(),
            "fdd3_sc20_sctf": weather_dataset[35].flatten(),
            "gdd_low_fall": weather_dataset[36].flatten(),
            "gdd_med_fall": weather_dataset[37].flatten(),
            "gdd_high_fall": weather_dataset[38].flatten(),
            "prcp_fall": weather_dataset[40].flatten(),  # Be careful, order is flipped with snowfall
            "snowfall_fall": weather_dataset[39].flatten(),
            "gdd_low_winter": weather_dataset[41].flatten(),
            "gdd_med_winter": weather_dataset[42].flatten(),
            "gdd_high_winter": weather_dataset[43].flatten(),
            "prcp_winter": weather_dataset[45].flatten(),  # Be careful, order is flipped with snowfall
            "snowfall_winter": weather_dataset[44].flatten(),
            "gdd_low_spring": weather_dataset[46].flatten(),
            "gdd_med_spring": weather_dataset[47].flatten(),
            "gdd_high_spring": weather_dataset[48].flatten(),
            "prcp_spring": weather_dataset[50].flatten(),  # Be careful, order is flipped with snowfall
            "snowfall_spring": weather_dataset[49].flatten(),
        })
        df_year = df_fdd_gdd_raw.groupby(["county_fips"]).mean()
        df_year = df_year.reset_index()
        df_year["year"] = year
        df_year = df_year[df_year.county_fips != 65535]

        if df is None:
            df_weather = df_year
        else:
            df_weather = pd.concat([df_weather, df_year], ignore_index=True)

    df = pd.merge(df, df_weather, left_on=["county_fips", "year"], right_on=["county_fips", "year"], how="right")

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
    output_dir = get_project_root() / "output/gridUS/2.5_arc_min"
    make_dirs(output_dir)

    export_snowfall(
        os.path.join(output_dir, "winter_wheat_input_ifdd_igdd_v20210107_gridUS_4km.csv"),
        "fdd_sctf_v36_{}.npz",
        "igdd_{}.npz",
    )
