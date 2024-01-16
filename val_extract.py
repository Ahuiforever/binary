# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/10/14 17:05
# @Author  : Ahuiforever
# @File    : val_extract.py
# @Software: PyCharm

import shutil
from glob import glob
from wjh.utils import LogWriter
import os


class Filer:
    def __init__(self,):
        pass

    def __call__(self, root_path: str, file_type: str) -> list:
        file_list = glob(os.path.join(root_path, f'*.{file_type}'))
        # >>> print(files) = "D:\\Work\\test\\450\\450_D_2022_06_15_14-54-43.tif"
        return file_list
    
    
class Extractor:
    def __init__(self, band_name: str, src_path: str, dst_path: str, img_files: list):
        self.band = band_name
        self.src_path = src_path
        self.dst_path = dst_path
        self.image_files = img_files

    def _single(self, _val_sample: tuple):
        # _val_sample = 'D_2022_06_15_14-54-43'
        file_name = f'{self.band}_{_val_sample}'
        try:
            shutil.move(src=os.path.join(self.src_path, file_name), dst=os.path.join(self.dst_path, file_name))
            log('Moved ', os.path.join(self.src_path, file_name))
        except FileNotFoundError:
            print(f'{os.path.join(self.src_path, file_name)} missed.')
                
    def single(self, *val_samples: any, is_list: bool = False):
        if is_list:
            for val_sample in val_samples:
                self._single(val_sample)
        else:
            self._single(val_samples)

    def multiple(self, *start_ends: tuple):
        for start_end in start_ends:
            start_with, end_with = start_end
            try:
                start_index = self.image_files.index(start_with)
                end_index = self.image_files.index(end_with)
                val_samples = self.image_files[start_index: end_index + 1]
                self.single(val_samples, is_list=True)
            except ValueError:
                print(f'{start_with} or {end_with} are not found in img_files. Go to the log file for the details.')
                log(self.image_files)


if __name__ == '__main__':
    band = '450'
    filer = Filer()
    log = LogWriter('val_extract_log.txt')
    ext = Extractor(band, 
                    r'D:\Work\test\train', 
                    r'D:\Work\test\val', 
                    filer(r'D:\Work\test', 'tif'))
    ext.single('D_2022_06_15_14-54-43.tif')
    ext.multiple(('D_2022_06_15_14-54-43.tif', 'D_2022_06_15_14-54-43.tif'))
