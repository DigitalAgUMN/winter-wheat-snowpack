import os
import datetime
import sys
import multiprocessing

import numpy as np
import tqdm

base_drive = "N:/gridMET"
if sys.platform == "linux" or sys.platform == "linux2":  # MSI
    base_drive = "/home/jinzn/taegon/WinterWheat/gridMET"

base_dir = os.path.join(base_drive, "gridMET_raw_daily_npy")
base_cmc_dir = os.path.join(base_drive, "CMC_gridMET_resolution_daily_npy")
output_dir = os.path.join(base_drive, "gridUS_annual_npy_20210107_sd1520")
# test_daily_dir = os.path.join(base_drive, "gridUS_daily_test_20210107_sd1520")

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
    with open(os.path.join(base_cmc_dir, "cmc_us_sdepth_dly_19980901.npy"), 'rb') as f:
        dataset_first = np.load(f)
    height, width = dataset_first.shape
    weather_array = np.empty((size_of_variables, height, width))

    fdd_period = [
        (datetime.datetime.strptime("{}/09/01".format(year - 1), "%Y/%m/%d"),
         datetime.datetime.strptime("{}/06/01".format(year), "%Y/%m/%d")),
        (datetime.datetime.strptime("{}/09/01".format(year - 1), "%Y/%m/%d"),
         datetime.datetime.strptime("{}/12/01".format(year - 1), "%Y/%m/%d")),
        (datetime.datetime.strptime("{}/12/01".format(year - 1), "%Y/%m/%d"),
         datetime.datetime.strptime("{}/03/01".format(year), "%Y/%m/%d")),
        (datetime.datetime.strptime("{}/03/01".format(year), "%Y/%m/%d"),
         datetime.datetime.strptime("{}/06/01".format(year), "%Y/%m/%d")),
    ]

    for pid, period in enumerate(fdd_period):
        start, end = period
        date_generated = [start + datetime.timedelta(days=x) for x in range(0, (end - start).days)]

        for date in tqdm.tqdm(date_generated, desc="process year={}".format(year)):
            date_str = date.strftime("%Y%m%d")
            doy = int(date.strftime('%j'))

            with open(os.path.join(base_dir, "gridMET_{}_{}.npy".format("tmmn", date_str)), 'rb') as f:
                tmin = np.load(f)
            with open(os.path.join(base_dir, "gridMET_{}_{}.npy".format("tmmx", date_str)), 'rb') as f:
                tmax = np.load(f)

            tmin = tmin - 273.15
            tmin = np.where(tmax == nodata, nodata, tmin)
            tmax = tmax - 273.15
            tmax = np.where(tmin == nodata, nodata, tmax)


            fdd1 = get_fdd(tmin, tmax, criteria=0)
            weather_array[pid, :, :] = weather_array[pid, :, :] + fdd1

    for idx in range(size_of_variables):
        weather_array[idx, :, :] = np.where(tmin == nodata, nodata, weather_array[idx, :, :])

    np.savez_compressed(os.path.join(output_dir, "fdd_season_{}.npz".format(year)), fdd_sctf=weather_array)
    return year


def main_multiprocessing():
    nodata = -9999
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_vars = [
        "fdd1",
        "fdd1_fall",
        "fdd1_winter",
        "fdd1_spring",
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
        e_year = 2019

    pool = multiprocessing.Pool(number_of_process)
    results = []
    for year in range(s_year, e_year+1):
        results.append(pool.apply_async(process, args=(year, len(model_vars), nodata)))
        # print(res.get())
    for result in results:
        processedYear = result.get()
        print("Result: Year=%d is processed." % (processedYear,))

    # for year in range(1999, 2020):
    #     process(year, size_of_variables=len(model_vars), nodata=nodata)


if __name__ == "__main__":
    # main()
    main_multiprocessing()
