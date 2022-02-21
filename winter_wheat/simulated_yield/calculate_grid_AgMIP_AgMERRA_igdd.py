import multiprocessing
import os
import datetime
import sys

import numpy as np
import pandas as pd
import tqdm

base_drive = "N:/WinterWheat/"

if sys.platform == "linux" or sys.platform == "linux2":  # MSI
    base_drive = "/scratch.global/taegon/WinterWheat/"

base_dir = os.path.join(base_drive, "AgMIP-AgMERRA/weather_daily_npy_0125")
output_dir = os.path.join(base_drive, "simulated-AgMIP-AgMERRA/CMC_AgMERRA_0125_annual_npy_20210726")

hourly_interpolation = np.sin(np.linspace(-np.pi / 2, np.pi / 2, num=24)) + 1


def get_igdd(tmin, tmax, criteria=0):
    tsa_kk = tmin + hourly_interpolation * (tmax - tmin) / 2 - criteria
    return tsa_kk[tsa_kk > 0].sum() / 24


def partition_snow(pr, tavg):
    t_rain = 3.0
    t_snow = -1.0

    pr_fraction = (t_rain - tavg) / (t_rain - t_snow)
    if pr_fraction < 0:
        pr_fraction = 0
    elif pr_fraction > 1:
        pr_fraction = 1
    return pr * pr_fraction


def partition_rain(pr, tavg):
    t_rain = 3.0
    t_snow = -1.0

    pr_fraction = (t_rain - tavg) / (t_rain - t_snow)
    if pr_fraction < 0:
        pr_fraction = 0
    elif pr_fraction > 1:
        pr_fraction = 1
    return pr * (1 - pr_fraction)


vectorized_get_igdd = np.vectorize(get_igdd)
vectorized_partition_snow = np.vectorize(partition_snow)
# vectorized_partition_rain = np.vectorize(partition_rain)


def vectorized_partition_rain(pr, tavg):
    t_rain = 3.0
    t_snow = -1.0

    pr_fraction = (t_rain - tavg) / (t_rain - t_snow)
    pr_fraction = np.where(pr_fraction < 0, 0, pr_fraction)
    pr_fraction = np.where(pr_fraction > 1, 1, pr_fraction)
    return pr * (1 - pr_fraction)


def process(year):
    nodata = -9999
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_vars = [
        "gdd_low_fall", "gdd_med_fall", "gdd_high_fall", "prcp_fall", "snowfall_fall",
        "gdd_low_winter", "gdd_med_winter", "gdd_high_winter", "prcp_winter", "snowfall_winter",
        "gdd_low_spring", "gdd_med_spring", "gdd_high_spring", "prcp_spring", "snowfall_spring",
    ]

    with open(os.path.join(base_dir, "1999/AgMERRA_pr_dly_19990101.npy"), 'rb') as f:
        dataset_first = np.load(f)
    height, width = dataset_first.shape

    # for year in tqdm.tqdm(range(s_year, s_year)):
    # for year in tqdm.tqdm(range(2004, 2005)):
    weather_array = np.empty((len(model_vars), height, width))

    start = datetime.datetime.strptime("{}/09/01".format(year - 1), "%Y/%m/%d")
    end = datetime.datetime.strptime("{}/12/01".format(year - 1), "%Y/%m/%d")
    # print(height, width)
    date_generated = [start + datetime.timedelta(days=x) for x in range(0, (end - start).days)]
    # print(date_generated)
    for i, date in enumerate(date_generated):
        criteria_lmh = [0, 10, 17]
        #         "fall": [0, 10, 17],
        #         "winter": [0, 5, 10],
        #         "spring": [0, 18, 30],

        date_str = date.strftime("%Y%m%d")
        date_year = date_str[:4]
        doy = int(date.strftime('%j'))

        with open(os.path.join(base_dir, "{}/AgMERRA_{}_dly_{}.npy".format(date_year, "tasmin", date_str)), 'rb') as f:
            tmin = np.load(f)
        with open(os.path.join(base_dir, "{}/AgMERRA_{}_dly_{}.npy".format(date_year, "tasmax", date_str)), 'rb') as f:
            tmax = np.load(f)
        with open(os.path.join(base_dir, "{}/AgMERRA_{}_dly_{}.npy".format(date_year, "pr", date_str)), 'rb') as f:
            pr = np.load(f)

        tmin = np.where(tmax == nodata, nodata, tmin)
        tmax = np.where(tmin == nodata, nodata, tmax)
        tavg = (tmin + tmax) / 2

        for i, criteria in enumerate(criteria_lmh):
            # gdd = np.copy(tavg)
            # gdd = gdd - criteria
            gdd = vectorized_get_igdd(tmin, tmax, criteria)
            gdd = np.where(gdd < 0, 0, gdd)
            weather_array[i + 5 * 0, :, :] = weather_array[i + 5 * 0, :, :] + gdd
        pr_partition = vectorized_partition_rain(pr, tavg)
        snow_partition = pr - pr_partition
        # snow_partition = verterized_partition_snow(pr, tavg)
        weather_array[3 + 5 * 0, :, :] = weather_array[3 + 5 * 0, :, :] + pr_partition
        weather_array[4 + 5 * 0, :, :] = weather_array[4 + 5 * 0, :, :] + snow_partition


    start = datetime.datetime.strptime("{}/12/01".format(year - 1), "%Y/%m/%d")
    end = datetime.datetime.strptime("{}/03/01".format(year), "%Y/%m/%d")
    date_generated = [start + datetime.timedelta(days=x) for x in range(0, (end - start).days)]
    for i, date in enumerate(date_generated):
        criteria_lmh = [0, 5, 10]
        #         "fall": [0, 10, 17],
        #         "winter": [0, 5, 10],
        #         "spring": [0, 18, 30],

        date_str = date.strftime("%Y%m%d")
        date_year = date_str[:4]
        doy = int(date.strftime('%j'))

        with open(os.path.join(base_dir, "{}/AgMERRA_{}_dly_{}.npy".format(date_year, "tasmin", date_str)), 'rb') as f:
            tmin = np.load(f)
        with open(os.path.join(base_dir, "{}/AgMERRA_{}_dly_{}.npy".format(date_year, "tasmax", date_str)), 'rb') as f:
            tmax = np.load(f)
        with open(os.path.join(base_dir, "{}/AgMERRA_{}_dly_{}.npy".format(date_year, "pr", date_str)), 'rb') as f:
            pr = np.load(f)

        tmin = np.where(tmax == nodata, nodata, tmin)
        tmax = np.where(tmin == nodata, nodata, tmax)
        tavg = (tmin + tmax) / 2

        for i, criteria in enumerate(criteria_lmh):
            # gdd = np.copy(tavg)
            # gdd = gdd - criteria
            gdd = vectorized_get_igdd(tmin, tmax, criteria)
            gdd = np.where(gdd < 0, 0, gdd)
            weather_array[i + 5 * 1, :, :] = weather_array[i + 5 * 1, :, :] + gdd
        pr_partition = vectorized_partition_rain(pr, tavg)
        snow_partition = pr - pr_partition
        # snow_partition = verterized_partition_snow(pr, tavg)
        weather_array[3 + 5 * 1, :, :] = weather_array[3 + 5 * 1, :, :] + pr_partition
        weather_array[4 + 5 * 1, :, :] = weather_array[4 + 5 * 1, :, :] + snow_partition
        # weather_array[3 + 4 * 1, :, :] = weather_array[3 + 4 * 1, :, :] + pr

    start = datetime.datetime.strptime("{}/03/01".format(year), "%Y/%m/%d")
    end = datetime.datetime.strptime("{}/06/01".format(year), "%Y/%m/%d")
    date_generated = [start + datetime.timedelta(days=x) for x in range(0, (end - start).days)]
    for i, date in enumerate(date_generated):
        criteria_lmh = [0, 18, 30]
        #         "fall": [0, 10, 17],
        #         "winter": [0, 5, 10],
        #         "spring": [0, 18, 30],

        date_str = date.strftime("%Y%m%d")
        date_year = date_str[:4]
        doy = int(date.strftime('%j'))

        with open(os.path.join(base_dir, "{}/AgMERRA_{}_dly_{}.npy".format(date_year, "tasmin", date_str)), 'rb') as f:
            tmin = np.load(f)
        with open(os.path.join(base_dir, "{}/AgMERRA_{}_dly_{}.npy".format(date_year, "tasmax", date_str)), 'rb') as f:
            tmax = np.load(f)
        with open(os.path.join(base_dir, "{}/AgMERRA_{}_dly_{}.npy".format(date_year, "pr", date_str)), 'rb') as f:
            pr = np.load(f)

        tmin = np.where(tmax == nodata, nodata, tmin)
        tmax = np.where(tmin == nodata, nodata, tmax)
        tavg = (tmin + tmax) / 2

        for i, criteria in enumerate(criteria_lmh):
            # gdd = np.copy(tavg)
            # gdd = gdd - criteria
            gdd = vectorized_get_igdd(tmin, tmax, criteria)
            gdd = np.where(gdd < 0, 0, gdd)
            # print(np.max(gdd))
            weather_array[i + 5 * 2, :, :] = weather_array[i + 5 * 2, :, :] + gdd
        pr_partition = vectorized_partition_rain(pr, tavg)
        snow_partition = pr - pr_partition
        # snow_partition = verterized_partition_snow(pr, tavg)
        weather_array[3 + 5 * 2, :, :] = weather_array[3 + 5 * 2, :, :] + pr_partition
        weather_array[4 + 5 * 2, :, :] = weather_array[4 + 5 * 2, :, :] + snow_partition
        # weather_array[3 + 4 * 2, :, :] = weather_array[3 + 4 * 2, :, :] + pr

    for idx in range((len(model_vars))):
        weather_array[idx, :, :] = np.where(dataset_first == nodata, nodata, weather_array[idx, :, :])

    np.savez_compressed(os.path.join(output_dir, "igdd_{}.npz".format(year)), gdd=weather_array)
    # loaded = np.load(os.path.join(output_dir, "gdd_{}.npz".format(year)))
    # print("save is okay? =>", np.array_equal(weather_array, loaded['fdd_sctf']))
    return year


def main():
    if len(sys.argv) >= 2:
        year = int(sys.argv[1])
        process(year, year+1)
    else:
        pool = multiprocessing.Pool(4)
        results = []
        for year in range(1999, 2010 + 1):
            results.append(pool.apply_async(process, args=(year,)))
            # print(res.get())
        for i, result in enumerate(tqdm.tqdm(results)):
            result.get()


if __name__ == "__main__":
    # This script use gridMET tiff (4km resolution)
    main()
