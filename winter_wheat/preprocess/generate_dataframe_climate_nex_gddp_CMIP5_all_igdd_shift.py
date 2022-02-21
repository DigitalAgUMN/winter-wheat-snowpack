import os
import multiprocessing

import pandas as pd
import tqdm

from winter_wheat.util import get_project_root, make_dirs


def export(base_dir, base_gdd_dir, model_name, fdd_filename_pattern, gdd_filename_pattern):
    model_vars = [
        "model", "scenario", "county_fips", "year",
        "fdd1",  # "fdd2", "fdd3",
        "fdd1_sc2", # "fdd1_sc5", "fdd1_sc10", "fdd1_sc15", "fdd1_sc20",
        # "fdd2_sc2", "fdd2_sc5", "fdd2_sc10", "fdd2_sc15", "fdd2_sc20",
        # "fdd3_sc2", "fdd3_sc5", "fdd3_sc10", "fdd3_sc15", "fdd3_sc20",
        "fdd1_day",  # "fdd2_day", "fdd3_day",
        "fdd1_sc2_sctf",  # "fdd1_sc5_sctf", "fdd1_sc10_sctf", "fdd1_sc15_sctf", "fdd1_sc20_sctf",
        # "fdd2_sc2_sctf", "fdd2_sc5_sctf", "fdd2_sc10_sctf", "fdd2_sc15_sctf", "fdd2_sc20_sctf",
        # "fdd3_sc2_sctf", "fdd3_sc5_sctf", "fdd3_sc10_sctf", "fdd3_sc15_sctf", "fdd3_sc20_sctf",
        "gdd_low_fall", "gdd_med_fall", "gdd_high_fall", "prcp_fall", "snowfall_fall",
        "gdd_low_winter", "gdd_med_winter", "gdd_high_winter", "prcp_winter", "snowfall_winter",
        "gdd_low_spring", "gdd_med_spring", "gdd_high_spring", "prcp_spring", "snowfall_spring",
    ]

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
    if not os.path.exists(base_dir):
        print("{} is not ready.".format(base_dir))
        return
    if not os.path.exists(base_gdd_dir):
        print("{} is not ready.".format(base_gdd_dir))
        return
    input_fdd_files = set([f for f in os.listdir(base_dir) if f.endswith(".csv")])
    input_gdd_files = set([f for f in os.listdir(base_gdd_dir) if f.endswith(".csv")])
    # print(input_files)


    for scenario in scenarios:
        for year in range(periods[scenario][0], periods[scenario][1] + 1):
            if fdd_filename_pattern.format(model_name, scenario, year) not in input_fdd_files:
                print("This file is missing: {}".format(fdd_filename_pattern.format(model_name, scenario, year)))
                return None
            elif gdd_filename_pattern.format(model_name, scenario, year) not in input_gdd_files:
                print("This file is missing: {}".format(gdd_filename_pattern.format(model_name, scenario, year)))
                return None

    df = None
    for scenario in scenarios:
        for year in range(periods[scenario][0], periods[scenario][1] + 1):
            fdd_filename = os.path.join(base_dir, fdd_filename_pattern.format(model_name, scenario, year))
            gdd_filename = os.path.join(base_gdd_dir, gdd_filename_pattern.format(model_name, scenario, year))
            df_fdd = pd.read_csv(fdd_filename)
            df_gdd = pd.read_csv(gdd_filename)
            df_var = df_fdd.merge(df_gdd, on=["county_fips", "season_year"])
            df_var["year"] = df_var["season_year"]
            df_var["scenario"] = scenario
            df_var["model"] = model_name

            df_var = df_var[model_vars]
            df_var = df_var.sort_values(by=["model", "scenario", "county_fips", "year"])
            if df is None:
                df = df_var
            else:
                df = pd.concat([df, df_var], ignore_index=True)

    df["gdd_low_fall"] = df["gdd_low_fall"] - df["gdd_med_fall"]
    df["gdd_med_fall"] = df["gdd_med_fall"] - df["gdd_high_fall"]
    df["gdd_low_winter"] = df["gdd_low_winter"] - df["gdd_med_winter"]
    df["gdd_med_winter"] = df["gdd_med_winter"] - df["gdd_high_winter"]
    df["gdd_low_spring"] = df["gdd_low_spring"] - df["gdd_med_spring"]
    df["gdd_med_spring"] = df["gdd_med_spring"] - df["gdd_high_spring"]

    return df


def process_bc(model_name):
    base_dir = os.path.join("N:/WinterWheat/NEX-GDDP-annual-swe-corrected-shift", model_name)
    base_gdd_dir = os.path.join("N:/WinterWheat/NEX-GDDP-annual-shift", model_name)

    output_base_dir = "N:/WinterWheat/model-input-swe-bc-shift"
    output_dir = os.path.join(output_base_dir, model_name)
    make_dirs(output_dir)

    output_filename = os.path.join(output_dir, "winter_wheat_input_igdd_sdepth_bc_v20210903_{}_shift-15.csv".format(model_name))

    if os.path.exists(output_filename):
        return

    df = export(base_dir, base_gdd_dir, model_name,
                fdd_filename_pattern="climate_input_data_fdd_swe_corrected_{}_{}_{}_shift-15_20210903.csv",
                gdd_filename_pattern="climate_input_data_igdd_{}_{}_{}_shift-15_20210903.csv",
                )

    if df is not None:
        df.to_csv(output_filename, index=False)

    output_filename = os.path.join(output_dir, "winter_wheat_input_igdd_sdepth_bc_v20210903_{}_shift+15.csv".format(model_name))

    if os.path.exists(output_filename):
        return

    df = export(base_dir, base_gdd_dir, model_name,
                fdd_filename_pattern="climate_input_data_fdd_swe_corrected_{}_{}_{}_shift+15_20210903.csv",
                gdd_filename_pattern="climate_input_data_igdd_{}_{}_{}_shift+15_20210903.csv",
                )

    if df is not None:
        df.to_csv(output_filename, index=False)


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

    # models = [
    #     # Phase 1: Using 3 GCMs in first step
    #     "CSIRO-Mk3-6-0",
    #     ]

    for model in models:
    #     process(model)
        process_bc(model)

    # pool = multiprocessing.Pool(4)
    # results = []
    # # print(models_for_mp)
    # for model in models:
    #     # results.append(pool.apply_async(process, args=(model, )))
    #     results.append(pool.apply_async(process_bc, args=(model, )))
    #
    # for i, result in enumerate(tqdm.tqdm(results)):
    #     result.get()
    #     print("Result: Model (idx={}) is processed.".format(i, ))


if __name__ == "__main__":
    main()
