import os
import datetime
import re
import multiprocessing

from osgeo import gdal, osr, gdal_array
import numpy as np
import xarray as xr
import tqdm
import rasterio

from winter_wheat.preprocess.generate_snow_depth_grid import create_geotiff
from winter_wheat.util import make_dirs


def get_netcdf_info(fname, BAND_VARS):
    """
    This dataset has the same null value for all bands. (:_FillValue = -999.0f)
    """
    dataset = {}
    for var in BAND_VARS:
        # print('Creating GDAL datastructures')
        subset = 'NETCDF:"' + fname + '":' + var
        sub_ds = gdal.Open(subset)
        nodata = sub_ds.GetRasterBand(1).GetNoDataValue()
        xsize = sub_ds.RasterXSize
        ysize = sub_ds.RasterYSize
        # print(sub_ds.GetGeoTransform())
        # geot = sub_ds.GetGeoTransform()
        geot = (-20, 0.05, 0, 38, 0, -0.05)
        proj = osr.SpatialReference()
        # proj.SetWellKnownGeogCS('NAD27')
        proj.SetWellKnownGeogCS("epsg:4326")
        data = xr.open_dataset(fname)[var]
        data = np.ma.masked_array(data, mask=np.isnan(data), fill_value=-9999)
        data = np.ma.masked_array(data, mask=(data == nodata), fill_value=-9999)
        print(nodata)
        print(data)
        print(data.mask)
        data = data[:,:,:].filled()
        xsize = data.shape[1]
        ysize = data.shape[0]
        print(data.shape)
        # np.savetxt(os.path.join(output_dir, "test_data.csv"), data[0,:,:], delimiter=",")
        # np.savetxt("../../output/MODIS_test_{}.csv".format(var), data[::-1,:,0], delimiter=',')
        dataset[var] = data

    return nodata, xsize, ysize, geot, proj, dataset


def export_one_day(output_dir):
    # get_netcdf_info("N:/CMIP5/snw/snw_day_inmcm4_historical_r1i1p1_20000101-20051231.nc", ["snw"])
    filename = "N:/CMIP5/snw/snw_day_inmcm4_historical_r1i1p1_20000101-20051231.nc"
    var_name = "snw"
    ds = xr.open_dataset(filename)[var_name].isel(time=0)
    # ds = xr.tutorial.open_dataset("air_temperature").isel(time=0)
    print(ds.lon)
    print(ds.lat)

    new_lon = np.linspace(0.125, 360-0.125, 360 * 4)
    new_lat = np.linspace(0.125, 90-0.125, 90 * 4)
    print(new_lon)
    print(new_lat)
    dsi = ds.interp(lat=new_lat, lon=new_lon, method="nearest")
    np.savetxt(os.path.join(output_dir, "test_snw_origin.csv"), ds,
               delimiter=",")
    np.savetxt(os.path.join(output_dir, "test_snw_interploated.csv"), dsi,
               delimiter=",")


def export_us_first(output_dir):
    us_bbox = [-125 + 360, 24, -66 + 360, 50]
    height = abs(us_bbox[3] - us_bbox[1]) * 4
    width = abs(us_bbox[2] - us_bbox[0]) * 4
    cell_size = .25
    # -125 + .125 = -124.875 => -124.875 + 124.7 = -0.175 / 0.04166666 = -4
    x_index = np.array([us_bbox[0] + cell_size / 2 + x*cell_size for x in range(width)])
    y_index = np.array([us_bbox[3] - cell_size / 2 - x * cell_size for x in range(height)])
    new_lon = np.linspace(us_bbox[0] + cell_size / 2, us_bbox[2] + cell_size / 2, width, endpoint=False)
    new_lat = np.linspace(us_bbox[3] - cell_size / 2, us_bbox[1] - cell_size / 2, height, endpoint=False)
    print(width, height)
    print(np.equal(x_index, new_lon))
    print(np.equal(y_index, new_lat))
    print(new_lon)
    print(new_lat)
    filename = "N:/CMIP5/snw/snw_day_inmcm4_historical_r1i1p1_20000101-20051231.nc"
    var_name = "snw"
    ds = xr.open_dataset(filename)[var_name]
    # print(ds.encoding)
    ds = ds.isel(time=0)
    dsi = ds.interp(lat=new_lat, lon=new_lon, method="nearest")
    np.savetxt(os.path.join(output_dir, "test_snw_interploated_us.csv"), dsi,
                   delimiter=",")


def read_nc_data(year, var_name="tmmn"):
    gridMET_var_names = {
        "tmmn": "air_temperature",
        "tmmx": "air_temperature",
        "pr": "precipitation_amount",
    }
    fname = "N:/gridMET/{}_{}.nc".format(var_name, year)
    subset = 'NETCDF:"' + fname + '":' + gridMET_var_names[var_name]
    sub_ds = gdal.Open(subset)
    nodata = sub_ds.GetRasterBand(1).GetNoDataValue()
    data = xr.open_dataset(fname)[gridMET_var_names[var_name]]
    data = np.ma.masked_array(data, mask=np.isnan(data), fill_value=-9999)
    data = np.ma.masked_array(data, mask=(data == nodata), fill_value=-9999)
    data = data[:, :, :].filled()
    return data


def is_leap_year(year) -> bool:
    return year % 4 == 0 and year % 100 != 0 or year % 400 == 0


def main(base_dir, output_base_npy_dir, model_name):
    output_npy_dir = os.path.join(output_base_npy_dir, model_name)
    if not os.path.exists(output_npy_dir):
        make_dirs(output_npy_dir)
    var_name = "snw"
    us_bbox = [-125 + 360, 24, -66 + 360, 50]

    # cell size = .125, each deg has 8 cells
    height = abs(us_bbox[3] - us_bbox[1]) * 8
    width = abs(us_bbox[2] - us_bbox[0]) * 8
    cell_size = .125
    new_lon = np.array([us_bbox[0] + cell_size / 2 + x*cell_size for x in range(width)])
    new_lat = np.array([us_bbox[3] - cell_size / 2 - x * cell_size for x in range(height)])
    # print(x_index)
    # print(y_index)

    scenarios = ["historical", "rcp45", "rcp85"]
    for scenario in scenarios:
        files = [f for f in os.listdir(base_dir) if re.match(r'snw_day_{}_{}.*\.nc'.format(model_name, scenario), f)]

        check_ensemble = {}
        for file in files:
            m = re.search(r'snw_day_{}_{}_([A-Za-z0-9]+)_([0-9]+)-([0-9]+).*\.nc'.format(model_name, scenario), file)
            ensemble = m.group(1)
            start_year = int(m.group(2))
            end_year = int(m.group(3))
            # print(ensemble, start_year, end_year)
            if ensemble not in check_ensemble:
                check_ensemble[ensemble] = []
            if start_year < 21000101:
                check_ensemble[ensemble].append(file)

        if len(check_ensemble) > 1:
            # print(sorted(list(check_ensemble.keys()))[0])
            files = check_ensemble[sorted(list(check_ensemble.keys()))[0]]
            if model_name == "GFDL-CM3" and scenario == "rcp45":
                files = check_ensemble[sorted(list(check_ensemble.keys()))[-1]]  # select GFDL-CM3_rcp45_r5i1p1 (2036-2040 is missing in original dataset) if this is comment out, it means r1i1p1 (2031-2100)

        for file in tqdm.tqdm(files, desc="Processing {} - {}".format(model_name, scenario)):
            m = re.search(r'snw_day_{}_{}_([A-Za-z0-9]+)_([0-9]+)-([0-9]+).*\.nc'.format(model_name, scenario), file)
            ensemble = m.group(1)
            start_yeardate = int(m.group(2))
            end_yeardate = int(m.group(3))
            if start_yeardate >= 21010101:
                continue
            elif end_yeardate < 19500101:
                continue

            start_year = start_yeardate // 10000
            end_year = end_yeardate // 10000
            if end_yeardate % 10000 != 1231:
                print("end date is not 12/31, check filename")
            filename = os.path.join(base_dir, file)
            ds = xr.open_dataset(filename)[var_name]

            leap_mode = False
            if model_name in ["CNRM-CM5", "MPI-ESM-MR", "MIROC-ESM", "MRI-CGCM3"]:
                start = datetime.datetime.strptime(str(start_yeardate), "%Y%m%d")
                end = datetime.datetime.strptime(str(end_yeardate), "%Y%m%d")
                assert ((end - start).days + 1) == ds.shape[0]
                leap_mode = True

            elif ds.shape[0] != (end_year - start_year + 1) * 365:
                print("Model: {}".format(model_name))
                print("number of data is not match year. This data might include leap. This program assume that 365-day calendar")
                return

            for yidx, y in enumerate(range(start_year, end_year + 1)):
                if y < 1950:
                    continue
                if os.path.exists(os.path.join(output_npy_dir, "snw_{}_{}_{}.npy".format(model_name, scenario, y))):
                    continue
                start_eachyear = datetime.datetime.strptime("{}0101".format(y), "%Y%m%d")

                days_in_year = 365
                if leap_mode and is_leap_year(y):
                    days_in_year = 366

                snw_annual = np.zeros((days_in_year, height, width))

                for d in range(days_in_year):
                    if leap_mode:
                        today = start_eachyear + datetime.timedelta(days=d)
                        ds_day = ds.isel(time=(today - start).days)
                    else:
                        ds_day = ds.isel(time=yidx * 365 + d)
                    dsi = ds_day.interp(lat=new_lat, lon=new_lon, method="nearest")
                    dsi = np.ma.masked_array(dsi, mask=np.isnan(dsi), fill_value=-9999)
                    # dsi = np.ma.masked_array(dsi, mask=(dsi == nodata), fill_value=-9999)
                    dsi = dsi[:, :].filled()
                    snw_annual[d, :, :] = dsi[:, :]
                np.save(os.path.join(output_npy_dir, "snw_{}_{}_{}.npy".format(model_name, scenario, y)), snw_annual)
                if y == 1952:
                    np.savetxt(os.path.join(output_npy_dir, "snw_{}_{}_{}.csv".format(model_name, scenario, y)), snw_annual[0],
                               delimiter=",")


if __name__ == "__main__":
    base_dir = "N:/CMIP5/snw"
    output_dir = "C:/Data/WinterWheat/CMIP5/snw_daily"
    output_dir = "C:/Data/WinterWheat/CMIP5/snw_daily_test_20200728"
    # output_npy_dir = "C:/Data/WinterWheat/CMIP5/snw_daily_npy"
    output_npy_dir = "N:/CMIP5/snw_daily_npy_0125"

    assert is_leap_year(2000) is True
    assert is_leap_year(1900) is False
    assert is_leap_year(2004) is True
    assert is_leap_year(2003) is False

    make_dirs(output_dir)
    make_dirs(output_npy_dir)

    models = [
        # Phase 1: Using 3 GCMs in first step
        "CanESM2",
        "CSIRO-Mk3-6-0",
        "inmcm4",

        # Phase 2: Add remained GCMs
        "CNRM-CM5",
        "GFDL-CM3",
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

    # export_one_day(output_dir)
    # export_us_first(output_dir)
    # main(base_dir, output_npy_dir, models[1])

    pool = multiprocessing.Pool(4)
    results = []
    # print(models_for_mp)
    for model in models:
        results.append(pool.apply_async(main, args=(base_dir, output_npy_dir, model)))
    for i, result in enumerate(results):
        result.get()
        print("Result: Model (idx={}) is processed.".format(i, ))
