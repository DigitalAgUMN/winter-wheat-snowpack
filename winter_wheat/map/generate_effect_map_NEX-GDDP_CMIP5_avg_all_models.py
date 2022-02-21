import os
from typing import NamedTuple
import multiprocessing

import pandas as pd

from winter_wheat.map.generate_effect_map_gridMET import aggregate_rank, aggregate
from winter_wheat.util import get_project_root, make_dirs


def process(base_dir, models, variables, file_pattern):
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
    for var_name in variables:
        for scenario in scenarios:
            for period_name, period_year in periods.items():
                data_filenames = []
                for model_name in models:
                    output_dir = os.path.join(base_dir, model_name)
                    output_dir_each = os.path.join(output_dir, scenario)
                    output_dir_each = os.path.join(output_dir_each, period_name)
                    data_filename = os.path.join(output_dir_each, file_pattern.format(var_name))
                    if not os.path.exists(data_filename):
                        print("{} is not exist.".format(data_filename))
                    else:
                        data_filenames.append(data_filename)
                if len(data_filenames) == 0:
                    print("{} doesn't have complete fileset.".format(var_name))
                    continue
                output_all_dir = os.path.join(base_dir, "all_models_avg_{}".format(len(data_filenames)))
                output_all_dir_each = os.path.join(output_all_dir, scenario)
                output_all_dir_each = os.path.join(output_all_dir_each, period_name)
                make_dirs(output_all_dir_each)
                avg_filename = os.path.join(output_all_dir_each, file_pattern.format(var_name))

                df_all = None
                for data_filename in data_filenames:
                    df_each = pd.read_csv(data_filename)
                    if df_all is None:
                        df_all = df_each
                    else:
                        df_all = pd.concat([df_all, df_each], ignore_index=True)
                if df_all is not None:
                    df_all = df_all.groupby(["county_fips"], as_index=False).mean()
                    df_all.reset_index()
                    df_all.to_csv(avg_filename, index=False)


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
        "gdd_all",
        "prcp_all",
        "effect_all",
        "gdd1_all",
        "gdd2_all",
        "gdd3_all",
        "gdd_fall_all",
        "gdd_winter_all",
        "gdd_spring_all",
        "snowfall_all",
    ]

    base_dir = get_project_root() / "output/county_map/v20200814-for-Peng-rawdata"  # fix igdd/ifdd errors
    base_dir = get_project_root() / "output/county_map/v20200927-for-Peng-rawdata"  # add snowfall variables
    base_dir = "N:/WinterWheat/county_map/v20210111-for-Peng-rawdata"  # add more GCMs


    process(base_dir, models, variables, "gridMET_{}_avg.csv")
    process(base_dir, models, variables, "gridMET_{}_top10p.csv")
    process(base_dir, models, variables, "gridMET_{}_gdd10p.csv")

    variables = [
        "fdd1_fall",
        "fdd1_winter",
        "fdd1_spring",
    ]
    process(base_dir, models, variables, "gridMET_{}_avg.csv")


if __name__ == "__main__":
    main()
