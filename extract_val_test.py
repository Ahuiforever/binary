# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/8 17:46
# @Author  : Ahuiforever
# @File    : extract_val_test.py
# @Software: PyCharm

"""
Extract validation set and test set from corresponding band folders according to extracted files in 450.
"""
import glob
import shutil
import os

from wjh.utils import LogWriter


class Extractor:
    def __init__(self, ref_dir: str, ref_band: int, label_dir: str) -> None:
        self.ref_dir = ref_dir
        # >>> E:\Work\try00
        self.ref_band = ref_band
        # >>> 450
        self.label_dir = label_dir
        # >>> E:\Work\labels
        self._band = None
        self.positive_label = {}
        self.total_label = {}

    def _extracted_img_list(self, val_test: str) -> list:
        # >>> E:\Work\try00\450\images\val\*.tif
        return glob.glob(
            self.ref_dir
            + os.sep
            + str(self.ref_band)
            + os.sep
            + rf"images\{val_test}"
            + os.sep
            + "*.tif"
        )

    def _label_read(self):
        label_path = self.label_dir + os.sep + f"{self._band}-label"
        # >>> E:\Work\labels\540-label
        label_path = (
            label_path
            + os.sep
            + self.src.split(os.sep)[-1]
            .replace(".tif", ".txt")
            .replace(str(self.ref_band), str(self._band))
        )
        # >>> E:\Work\labels\540-label\*.txt
        try:
            with open(label_path, "r") as f:
                try:
                    f.read(1)
                    label = 1.0
                except ValueError:
                    label = 0.0
        except FileNotFoundError:
            label = 0.0
        return label

    def _move(self, val_test: str):
        key = f"{self._band}-{val_test}"
        self.positive_label[key] = 0.0
        self.total_label[key] = 0.0
        for img_path in self._extracted_img_list(val_test):
            self.src = img_path.replace(
                rf"{self.ref_band}\images\{val_test}", rf"{self._band}\images"
            )
            # >>> E:\Work\try00\450\images\val\*.tif -> E:\Work\try00\540\images\*.tif
            dst = img_path.replace(r"images", rf"images\{val_test}")
            # >>> E:\Work\try00\540\images\val\*.tif
            if self.check:
                if not os.path.exists(self.src):
                    log(self.src, val_test + 'Missing', printf=True)
                else:
                    self.total_label[key] += 1.0
            else:
                try:
                    shutil.move(self.src, dst)
                except FileNotFoundError:
                    log(self.src, f"{val_test} not exists", printf=True)
            self.positive_label[key] += self._label_read()

    def __call__(self, _band: int, check: bool = False) -> None:
        """
        Move the corresponding images from ./images to ./val and ./test.
        """
        self._band = _band
        self.check = check
        self._move("val")
        self._move("test")


if __name__ == "__main__":
    log = LogWriter("extract_val_test.txt")
    extractor = Extractor(r"E:\Work\try00", 450, r"E:\Work\labels")
    bands = [540, 750, 900, 950]
    for band in bands:
        extractor(band, check=True)
    log(extractor.positive_label, extractor.total_label, printf=True)
    # // todo 1: Count the number of positive and negative samples and print or save。
    # // todo 2: How to deal with missing data?
