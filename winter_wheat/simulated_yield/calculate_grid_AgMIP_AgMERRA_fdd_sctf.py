import os
import datetime
import sys
import multiprocessing

import numpy as np
import pandas as pd
import tqdm

base_drive = "N:/WinterWheat/"

if sys.platform == "linux" or sys.platform == "linux2":  # MSI
    base_drive = "/scratch.global/taegon/WinterWheat/"

base_dir = os.path.join(base_drive, "AgMIP-AgMERRA/weather_daily_npy_0125")
base_CMC_dir = os.path.join(base_drive, "CMC/CMC_0125_resolution_daily_npy")
output_dir = os.path.join(base_drive, "simulated-AgMIP-AgMERRA/CMC_AgMERRA_0125_annual_npy_20210726")

hourly_interpolation = np.sin(np.linspace(-np.pi / 2, np.pi / 2, num=24)) + 1


def get_fdd(tmin, tmax, criteria=0, nodata=-9999):
    it = np.nditer([tmin, tmax, None])
    with it:
        for tmn, tmx, tsa in it:
            if tmn == nodata or tmx == nodata:
                tsa[...] = nodata
            else:
                tsa_kk = criteria - (tmn + hourly_interpolation * (tmx - tmn) / 2)
                # if tsa_kk[tsa_kk < 0].sum() < 0:
                #     print(tsa_kk)
                #     print("=>", -tsa_kk[tsa_kk < 0].sum())
                tsa[...] = tsa_kk[tsa_kk > 0].sum() / 24

        tsa = it.operands[2]  # same as z
    tsa = np.where(tmin == nodata, nodata, tsa)
    return tsa


def get_fdd_snow_cover(fdd_array, sdepth_array, criteria=2, nodata=-9999):
    it = np.nditer([fdd_array, sdepth_array, None])
    with it:
        for fdd, sdepth, fdd_sc in it:
            if fdd == nodata or sdepth == nodata:
                fdd_sc[...] = nodata
            else:
                if fdd > 0 and sdepth >= criteria:
                    fdd_sc[...] = 1
                else:
                    fdd_sc[...] = 0
        fdd_sc = it.operands[2]  # same as z
    fdd_sc = np.where(fdd_array == nodata, nodata, fdd_sc)
    return fdd_sc


def process(year, size_of_variables, nodata=-9999):
    start = datetime.datetime.strptime("{}/09/01".format(year - 1), "%Y/%m/%d")
    end = datetime.datetime.strptime("{}/06/01".format(year), "%Y/%m/%d")

    with open(os.path.join(base_CMC_dir, "cmc_us_0125_sdepth_dly_19980901.npy"), 'rb') as f:
        dataset_first = np.load(f)
    height, width = dataset_first.shape
    # print(height, width)
    weather_array = np.empty((size_of_variables, height, width))
    date_generated = [start + datetime.timedelta(days=x) for x in range(0, (end - start).days)]
    # print(date_generated)

    # if year == 1999:
    #     for i, date in enumerate(tqdm.tqdm(date_generated, desc="process year={}".format(year))):
    #         if i > 2:
    #             break
    #         date_str = date.strftime("%Y%m%d")
    #         doy = int(date.strftime('%j'))
    #
    #         with open(os.path.join(base_dir, "gridMET_{}_{}.npy".format("tmmn", date_str)), 'rb') as f:
    #             tmin = np.load(f)
    #         with open(os.path.join(base_dir, "gridMET_{}_{}.npy".format("tmmx", date_str)), 'rb') as f:
    #             tmax = np.load(f)
    #         # with open(os.path.join(base_dir, "gridMET_{}_{}.npy".format("pr", date_str)), 'rb') as f:
    #         #     pr = np.load(f)
    #         with open(os.path.join(base_ERA5_agmet_dir, "ERA5-agmet_us_sdepth_dly_{}.npy".format(date_str)), 'rb') as f:
    #             sdepth = np.load(f)
    #         # np.linspace(-np.pi / 2, np.pi / 2, num=24)
    #
    #         tmin = tmin - 273.15
    #         tmin = np.where(tmax == nodata, nodata, tmin)
    #         tmax = tmax - 273.15
    #         tmax = np.where(tmin == nodata, nodata, tmax)
    #         # print(tmin)
    #         # tmin2 = tmin2 + 273.15
    #         # tmin2 = np.where(tmax == -9999, -9999, tmin2)
    #         # print(np.allclose(tmin, tmin2))
    #         df = pd.DataFrame({'tmin': tmin.flatten(), 'tmax': tmax.flatten(), 'sdepth': sdepth.flatten()})
    #
    #         fdd1 = get_fdd(tmin, tmax, criteria=0)
    #         fdd2 = get_fdd(tmin, tmax, criteria=-5)
    #         fdd3 = get_fdd(tmin, tmax, criteria=-10)
    #
    #         df = pd.concat([df, pd.DataFrame({'fdd1': fdd1.flatten(),
    #                                           'fdd2': fdd2.flatten(),
    #                                           'fdd3': fdd3.flatten(), })], axis=1)
    #
    #
    #         for idx, fdd_123 in enumerate([fdd1, fdd2, fdd3]):
    #             fdd_123_sc2 = get_fdd_snow_cover(fdd_123, sdepth, criteria=2)
    #             fdd_123_sc5 = get_fdd_snow_cover(fdd_123, sdepth, criteria=5)
    #             fdd_123_sc10 = get_fdd_snow_cover(fdd_123, sdepth, criteria=10)
    #             fdd_123_sc15 = get_fdd_snow_cover(fdd_123, sdepth, criteria=15)
    #             fdd_123_sc20 = get_fdd_snow_cover(fdd_123, sdepth, criteria=20)
    #
    #             df = pd.concat([df, pd.DataFrame({'fdd{}_sc2'.format(idx + 1): fdd_123_sc2.flatten(),
    #                                               'fdd{}_sc5'.format(idx + 1): fdd_123_sc5.flatten(),
    #                                               'fdd{}_sc10'.format(idx + 1): fdd_123_sc10.flatten(),
    #                                               'fdd{}_sc15'.format(idx + 1): fdd_123_sc15.flatten(),
    #                                               'fdd{}_sc20'.format(idx + 1): fdd_123_sc20.flatten(),
    #                                               })], axis=1)
    #
    #         for idx, fdd_123 in enumerate([fdd1, fdd2, fdd3]):
    #             fdd_123_days = np.where(fdd_123 > 0, 1, fdd_123)
    #             df = pd.concat([df, pd.DataFrame({'fdd{}_days'.format(idx+1): fdd_123_days.flatten(), })], axis=1)
    #         df.to_csv(os.path.join(test_daily_dir, "fdd_sctf_daily_test_{}.csv".format(date_str)))

    # for i, date in enumerate(date_generated):
    for i, date in enumerate(tqdm.tqdm(date_generated)):
        date_str = date.strftime("%Y%m%d")
        date_year = date_str[:4]
        doy = int(date.strftime('%j'))

        with open(os.path.join(base_dir, "{}/AgMERRA_{}_dly_{}.npy".format(date_year, "tasmin", date_str)), 'rb') as f:
            tmin = np.load(f)
        with open(os.path.join(base_dir, "{}/AgMERRA_{}_dly_{}.npy".format(date_year, "tasmax", date_str)), 'rb') as f:
            tmax = np.load(f)
        with open(os.path.join(base_CMC_dir, "cmc_us_0125_sdepth_dly_{}.npy".format(date_str)), 'rb') as f:
            sdepth = np.load(f)
        # np.linspace(-np.pi / 2, np.pi / 2, num=24)

        tmin = np.where(tmax == nodata, nodata, tmin)
        tmax = np.where(tmin == nodata, nodata, tmax)
        # print(tmin)
        # tmin2 = tmin2 + 273.15
        # tmin2 = np.where(tmax == -9999, -9999, tmin2)
        # print(np.allclose(tmin, tmin2))

        fdd1 = get_fdd(tmin, tmax, criteria=0)
        fdd2 = get_fdd(tmin, tmax, criteria=-5)
        fdd3 = get_fdd(tmin, tmax, criteria=-10)
        weather_array[0, :, :] = weather_array[0, :, :] + fdd1
        weather_array[1, :, :] = weather_array[1, :, :] + fdd2
        weather_array[2, :, :] = weather_array[2, :, :] + fdd3

        for idx, fdd_123 in enumerate([fdd1, fdd2, fdd3]):
            fdd_123_sc2 = get_fdd_snow_cover(fdd_123, sdepth, criteria=2)
            fdd_123_sc5 = get_fdd_snow_cover(fdd_123, sdepth, criteria=5)
            fdd_123_sc10 = get_fdd_snow_cover(fdd_123, sdepth, criteria=10)
            fdd_123_sc15 = get_fdd_snow_cover(fdd_123, sdepth, criteria=15)
            fdd_123_sc20 = get_fdd_snow_cover(fdd_123, sdepth, criteria=20)

            weather_array[idx*5 + 3, :, :] = weather_array[idx*5 + 3, :, :] + fdd_123_sc2
            weather_array[idx*5 + 4, :, :] = weather_array[idx*5 + 4, :, :] + fdd_123_sc5
            weather_array[idx*5 + 5, :, :] = weather_array[idx*5 + 5, :, :] + fdd_123_sc10
            weather_array[idx*5 + 6, :, :] = weather_array[idx*5 + 6, :, :] + fdd_123_sc15
            weather_array[idx*5 + 7, :, :] = weather_array[idx*5 + 7, :, :] + fdd_123_sc20

        for idx, fdd_123 in enumerate([fdd1, fdd2, fdd3]):
            fdd_123_days = np.where(fdd_123 > 0, 1, fdd_123)
            # if not np.array_equal(fdd_123_days[(fdd_123_days != 0) & (fdd_123_days != -9999) & (fdd_123_days != 1)], np.array([])):
            #     print(fdd_123_days[(fdd_123_days != 0) & (fdd_123_days != -9999) & (fdd_123_days != 1)])
            weather_array[idx + 18, :, :] = weather_array[idx + 18, :, :] + fdd_123_days

    for fdd_idx in range(3):  # fdd1
        for sc_idx in range(5):  # sc2, sc5, sc10, sc15, sc20
            with np.errstate(divide='ignore', invalid='ignore'):
                weather_array[21 + fdd_idx * 5 + sc_idx, :, :] = np.where(weather_array[fdd_idx + 18, :, :] == 0, -9999,
                                                                          weather_array[fdd_idx * 5 + 3 + sc_idx, :,
                                                                          :] / weather_array[fdd_idx + 18, :, :])

    for idx in range(size_of_variables):
        weather_array[idx, :, :] = np.where(tmin == nodata, nodata, weather_array[idx, :, :])

    np.savez_compressed(os.path.join(output_dir, "fdd_sctf_AgMERRA_v36_{}.npz".format(year)), fdd_sctf=weather_array)
    return year


def main():
    nodata = -9999
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    us_bbox = [-125, 24, -66, 50]
    model_vars = [
        "fdd1", "fdd2", "fdd3",
        "fdd1_sc2", "fdd1_sc5", "fdd1_sc10", "fdd1_sc15", "fdd1_sc20",
        "fdd2_sc2", "fdd2_sc5", "fdd2_sc10", "fdd2_sc15", "fdd2_sc20",
        "fdd3_sc2", "fdd3_sc5", "fdd3_sc10", "fdd3_sc15", "fdd3_sc20",
        "fdd1_day", "fdd2_day", "fdd3_day",
        "fdd1_sc2_sctf", "fdd1_sc5_sctf", "fdd1_sc10_sctf", "fdd1_sc15_sctf", "fdd1_sc20_sctf",
        "fdd2_sc2_sctf", "fdd2_sc5_sctf", "fdd2_sc10_sctf", "fdd2_sc15_sctf", "fdd2_sc20_sctf",
        "fdd3_sc2_sctf", "fdd3_sc5_sctf", "fdd3_sc10_sctf", "fdd3_sc15_sctf", "fdd3_sc20_sctf",
    ]

    if len(sys.argv) >= 2:
        year = int(sys.argv[1])

    process(year, size_of_variables=len(model_vars), nodata=nodata)


def main_multiprocessing():
    nodata = -9999
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_vars = [
        "fdd1", "fdd2", "fdd3",
        "fdd1_sc2", "fdd1_sc5", "fdd1_sc10", "fdd1_sc15", "fdd1_sc20",
        "fdd2_sc2", "fdd2_sc5", "fdd2_sc10", "fdd2_sc15", "fdd2_sc20",
        "fdd3_sc2", "fdd3_sc5", "fdd3_sc10", "fdd3_sc15", "fdd3_sc20",
        "fdd1_day", "fdd2_day", "fdd3_day",
        "fdd1_sc2_sctf", "fdd1_sc5_sctf", "fdd1_sc10_sctf", "fdd1_sc15_sctf", "fdd1_sc20_sctf",
        "fdd2_sc2_sctf", "fdd2_sc5_sctf", "fdd2_sc10_sctf", "fdd2_sc15_sctf", "fdd2_sc20_sctf",
        "fdd3_sc2_sctf", "fdd3_sc5_sctf", "fdd3_sc10_sctf", "fdd3_sc15_sctf", "fdd3_sc20_sctf",
    ]

    # jobs = []
    # for year in range(2006, 2020):
        # p = multiprocessing.Process(target=process, args=(year, len(model_vars), nodata))
        # jobs.append(p)
        # p.start()
    number_of_process = 4
    if len(sys.argv) >= 2:
        number_of_process = int(sys.argv[1])
    if len(sys.argv) >= 4:
        s_year = int(sys.argv[2])
        e_year = int(sys.argv[3])
    else:
        s_year = 1999
        e_year = 2010

    pool = multiprocessing.Pool(number_of_process)
    results = []
    for year in range(s_year, e_year+1):
        results.append(pool.apply_async(process, args=(year, len(model_vars), nodata)))
        # print(res.get())
    for result in tqdm.tqdm(results, desc="processing fdd"):
        processedYear = result.get()

    # for year in range(1999, 2020):
    #     process(year, size_of_variables=len(model_vars), nodata=nodata)


if __name__ == "__main__":
    # main()  # for MSI
    main_multiprocessing()
