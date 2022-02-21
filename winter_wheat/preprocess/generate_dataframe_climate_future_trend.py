import os
import multiprocessing

import pandas as pd
import tqdm

from winter_wheat.util import get_project_root, make_dirs


def export(base_dir, model_name, filename_pattern):
    model_vars = [
        "model", "scenario", "county_fips", "year",
        "tasmin_fall", "tasmax_fall", "tasavg_fall", "prcp_fall", "snow_depth_mm_fall",
        "tasmin_winter", "tasmax_winter", "tasavg_winter", "prcp_winter", "snow_depth_mm_winter",
        "tasmin_spring", "tasmax_spring", "tasavg_spring", "prcp_spring", "snow_depth_mm_spring",
        "tasmin", "tasmax", "tasavg", "pr", "snow_depth_mm",
    ]

    scenarios = ["historical", "rcp45", "rcp85"]
    periods = {
        "historical": (1951, 2005),
        "rcp45": (2007, 2100),
        "rcp85": (2007, 2100),
    }
    if not os.path.exists(base_dir):
        print("{} is not ready.".format(base_dir))
        return
    input_files = set([f for f in os.listdir(base_dir) if f.endswith(".csv")])
    # print(input_files)

    for scenario in scenarios:
        for year in range(periods[scenario][0], periods[scenario][1] + 1):
            if filename_pattern.format(model_name, scenario, year) not in input_files:
                print("This file is missing: {}".format(filename_pattern.format(model_name, scenario, year)))
                return None

    df = None
    for scenario in scenarios:
        for year in range(periods[scenario][0], periods[scenario][1] + 1):
            fdd_filename = os.path.join(base_dir, filename_pattern.format(model_name, scenario, year))
            df_var = pd.read_csv(fdd_filename)
            df_var["year"] = df_var["season_year"]
            df_var["scenario"] = scenario
            df_var["model"] = model_name

            df_var = df_var[model_vars]
            df_var = df_var.sort_values(by=["model", "scenario", "county_fips", "year"])
            if df is None:
                df = df_var
            else:
                df = pd.concat([df, df_var], ignore_index=True)

    return df


def process_bc(model_name):
    base_dir = os.path.join("N:/WinterWheat/NEX-GDDP-annual-swe-corrected", model_name)

    output_base_dir = "N:/WinterWheat/model-input-swe-bc"
    output_dir = os.path.join(output_base_dir, model_name)
    make_dirs(output_dir)

    output_filename = os.path.join(output_dir, "winter_wheat_climate_trend_sdepth_bc_v20210129_{}.csv".format(model_name))

    if os.path.exists(output_filename):
        return

    df = export(base_dir, model_name,
                filename_pattern="climate_trend_swe_corrected_{}_{}_{}_20210129.csv",
                )

    if df is not None:
        df.to_csv(output_filename, index=False)


def process(model_name):
    base_dir = os.path.join("N:/WinterWheat/NEX-GDDP-annual", model_name)
    output_base_dir = "N:/WinterWheat/model-input"
    output_dir = os.path.join(output_base_dir, model_name)
    make_dirs(output_dir)

    output_filename = os.path.join(output_dir, "winter_wheat_climate_trend_v20210129_{}.csv".format(model_name))

    if os.path.exists(output_filename):
        return

    df = export(base_dir, model_name,
                filename_pattern="climate_trend_{}_{}_{}_20210129.csv",
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

    # for model in models:
    #     process(model)
    #     process_bc(model)

    pool = multiprocessing.Pool(4)
    results = []
    # print(models_for_mp)
    for model in models:
        results.append(pool.apply_async(process, args=(model, )))
        results.append(pool.apply_async(process_bc, args=(model, )))

    for i, result in enumerate(tqdm.tqdm(results)):
        result.get()
        # print("Result: Model (idx={}) is processed.".format(i, ))


if __name__ == "__main__":
    main()
