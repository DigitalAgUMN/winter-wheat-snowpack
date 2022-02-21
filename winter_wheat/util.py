from pathlib import Path
import os

import pandas as pd
from pandas import DataFrame


def get_project_root() -> Path:
    """Returns project root folder."""
    # Path(__file__).parent.parent.parent
    return Path(__file__).parents[1]


def make_dirs(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def is_float(str_value):
    try:
        float(str_value)
        return True
    except ValueError:
        return False


def read_clean_data(yield_source: str = "survey",
                    snow_source: str = "CMC",
                    min_prod: float = 100000,
                    min_time_series: int = 4,
                    min_lat: float = None) -> DataFrame:
    YIELD_SOURCES = ("census", "survey")
    SNOW_SOURCES = ("CMC", "NCC", "PNAS-DD", "NCC-CMC")

    output_dir = os.path.join(get_project_root(), "output")

    if yield_source is None or yield_source not in YIELD_SOURCES:
        yield_source = "survey"
    if snow_source is None or snow_source not in SNOW_SOURCES:
        snow_source = "CMC"

    data = pd.read_csv(os.path.join(output_dir, "us_winter_wheat_cmc_dd13_freezing.csv"), low_memory=False)
    data = data[data["production"] > min_prod]
    if min_lat is not None:
        data = data[data["latitude"] > min_lat]
    data["dfwos"] = data['dfwos'].where(data['dfwos'] > 0, 0)
    data["dsc"] = data['dsc'].where(data['dsc'] > 0, 0)
    data["dfg"] = data['dfg'].where(data['dfg'] > 0, 0)
    data["dfwos_masked"] = data['dfwos_masked'].where(data['dfwos_masked'] > 0, 0)
    data["dsc_masked"] = data['dsc_masked'].where(data['dsc_masked'] > 0, 0)

    if yield_source == "survey":
        data["yield"] = data["yield_survey"]
        data["production"] = data["production_survey"]
        data["acre"] = data["acre_survey"]
        data.drop(['yield_census', 'production_census', 'acre_census', 'acre_irrigated_survey', 'irr_acre_percent'], axis=1, inplace=True)
    elif yield_source == "census":
        data["yield"] = data["yield_census"]
        data["production"] = data["production_census"]
        data["acre"] = data["acre_census"]
        data.drop(['yield_survey', 'production_survey', 'acre_survey', 'acre_irrigated_survey', 'irr_acre_percent'], axis=1, inplace=True)
    else:
        # return pd.DataFrame({'county_fips': [], 'year': [], 'yield': []})
        raise TypeError("Not supported type (census or survey): {} - {}".format(yield_source, snow_source))

    if snow_source == "CMC":
        data.drop(['dfwos', 'dsc', 'dfg', 'dfwos_masked', 'dsc_masked',], axis=1, inplace=True)
    elif snow_source == "NCC":  # NCC
        data.drop(['cmc_snow_depth_avg', 'cmc_snow_period', 'cmc_snow_cover_days_over_0',
                   'cmc_snow_cover_days_over_5', 'cmc_snow_cover_days_over_10',
                   'cmc_snow_cover_days_over_15', 'cmc_snow_depth_avg_by_snow_day',
                   'ne5f_sc_daymet', '12f_sc_daymet', '0f_sc_daymet', 'ne10c_sc_daymet'], axis=1, inplace=True)
    elif snow_source == "NCC-CMC":  # NCC
        data.drop(['cmc_snow_depth_avg', 'cmc_snow_period', 'cmc_snow_cover_days_over_0',
                   'cmc_snow_cover_days_over_5', 'cmc_snow_cover_days_over_10',
                   'cmc_snow_cover_days_over_15', 'cmc_snow_depth_avg_by_snow_day',
                   ], axis=1, inplace=True)
    elif snow_source == "PNAS-DD":
        data.drop(['dfwos', 'dsc', 'dfg',], axis=1, inplace=True)
        data.drop(['cmc_snow_depth_avg', 'cmc_snow_period', 'cmc_snow_cover_days_over_0',
                   'cmc_snow_cover_days_over_5', 'cmc_snow_cover_days_over_10',
                   'cmc_snow_cover_days_over_15', 'cmc_snow_depth_avg_by_snow_day',
                   'ne5f_sc_daymet', '12f_sc_daymet', '0f_sc_daymet', 'ne10c_sc_daymet'], axis=1, inplace=True)
    else:
        raise TypeError("Not supported type (CMC or NCC): {} - {}".format(yield_source, snow_source))

    # unit conversion: precipitation: m -> mm
    data["prcp_fall"] = data["prcp_fall"] * 1000
    data["prcp_winter"] = data["prcp_winter"] * 1000
    data["prcp_spring"] = data["prcp_spring"] * 1000

    data = data.dropna()
    counts = data['county_fips'].value_counts()
    data = data[~data['county_fips'].isin(counts[counts < min_time_series].index)]
    data["year_copy"] = data["year"]
    data["fips_copy"] = data["county_fips"]

    return data
