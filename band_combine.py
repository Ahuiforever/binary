# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/8/31 14:40
# @Author  : Ahuiforever
# @File    : band_combine.py
# @Software: PyCharm

import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
from osgeo import gdal

band1_path = r'D:\Work\try\450'
band2_path = r'D:\Work\try\540'
band1 = '450'
band2 = '540'
save_path = r'D:\Work\try9\450-540'

band1_file = glob(f'{band1_path}/*.tif')
band2_file = glob(f'{band2_path}/*.tif')

# if len(band1_file) != len(band2_file):
#     diff = len(band2_file) - len(band1_file)
#     if diff > 0:
#         band1_file.append()

i = 0
for file in tqdm(band1_file):
    if file.replace(band1, band2) in band2_file:
        i += 1
        img1 = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(file.replace(band1, band2), cv2.IMREAD_GRAYSCALE)
        img = np.stack((img1, img2, img2), axis=-1)
        # ds1 = gdal.Open(file)
        # width = ds1.RasterXSize
        # height = ds1.RasterYSize
        # num: int = ds1.RasterCount
        # projection = ds1.GetProjection()
        # geo_transform = ds1.GetGeoTransform()
        #
        # ds2 = gdal.Open(file.replace(band1, band2))
        #
        # driver = gdal.GetDriverByName("GTiff")
        # dataset = driver.Create(save_path, width, height, 2, gdal.GDT_UInt16)
        # dataset.SetProjection(projection)
        # dataset.SetGeoTransform(geo_transform)
        #
        # dataset.GetRasterBand(1).WriteArray(ds1)
        # dataset.FlushCache()

        cv2.imwrite(file.replace(band1, band2).replace(band2_path, save_path).replace('tif', 'jpg'), img)
        # print(file.replace(band1, band2).replace(band2_path, save_path))
print(f'Combined {i}/{band1_file.__len__()} images.')

# band_file = list(set(band1_file) & set(band2_file))
# for file in tqdm(band_file):
#     i += 1
#     img1 = cv2.imread(file)
#     img2 = cv2.imread(file.replace(band1, band2))
#     img = np.stack((img1, img2), axis=-1)
#     # cv2.imwrite(file.replace(band1, band2).replace(band2_path, save_path), img)
# print(f'Combined {i}/{band1_file.__sizeof__()} images.')
