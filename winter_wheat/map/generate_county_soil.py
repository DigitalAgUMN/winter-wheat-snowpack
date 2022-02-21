import os

import pandas as pd

from winter_wheat.util import get_project_root, make_dirs


def generate_dataset(output_filename: str, dataset_filename: str, df_fips: pd.DataFrame) -> None:
    # Load the dataset
    df_soil_pct = pd.read_csv(dataset_filename)[["FIPS", "sandtotal_r", "silttotal_r", "om_r"]]
    df_soil_pct["claytotal"] = 100 - df_soil_pct["sandtotal_r"] - df_soil_pct["silttotal_r"]
    df_soil_pct = df_soil_pct.rename(columns={"FIPS": "county_fips", })


    # df_acre = pd.melt(df_acre, id_vars=['county_fips'])
    # df_acre = df_acre.astype({"county_fips": int, "variable": int})
    # df_acre["value"] = df_acre["value"].str.replace("(D)", "", regex=False)
    # df_acre["value"] = df_acre["value"].str.replace(",", "")
    # df_acre["value"] = pd.to_numeric(df_acre["value"], errors='coerce')
    # df_acre = df_acre.rename(columns={"variable": "year", "value": "acre_survey"})

    df = pd.merge(df_fips, df_soil_pct, on="county_fips", how="left")
    df = df[["county_fips", "sandtotal_r", "claytotal", "silttotal_r", "om_r"]]

    df.to_csv(output_filename, index=False)


def main():
    output_dir = get_project_root() / "output/US/soil_pct"
    df_fips = pd.read_csv(get_project_root() / "input/county_fips_3107.csv")
    make_dirs(output_dir)

    # This file is generated from ECOSYS-util (ecosys_util.preprocess.winter_wheat.get_avg_mu_each_county_STATSGO2
    soil_filename = "N:/users/taegon/DATA/gSSURGO_20210715/each_county_STATSGO2/county_soil_county_US_final.csv"
    output_filename = os.path.join(output_dir, "soil_pct_by_county_20211018.csv")

    generate_dataset(output_filename, soil_filename, df_fips)


if __name__ == "__main__":
    main()
