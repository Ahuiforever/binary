# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/11/2 11:07
# @Author  : Ahuiforever
# @File    : export_tensorboard_data.py
# @Software: PyCharm
import glob
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator

bands = [450, 540, 750, 900, 950]
items = ['accuracy_train', 'accuracy_val', 'loss_train', 'loss_val',
         'Val Index_Precision', 'Val Index_Recall',
         'Val Index_F1', 'Val Index_False Alarm']
tensorboard_path = r'D:\OneDrive - USTC\Data\Pycharm\BinaryClassify\resnet binary\result01_vgg16_bn'
df = pd.DataFrame(columns=items)
xlsx_writer = pd.ExcelWriter('result01.xlsx')

for band in bands:
    for idx, item in enumerate(items):
        for events_file in glob.glob(fr'{tensorboard_path}/{band}_tensorboard/{item}/*events.out.tfevents.*'):
            ea = event_accumulator.EventAccumulator(events_file)
            ea.Reload()
            key = ea.scalars.Keys()[0]
            df[item] = pd.DataFrame(ea.Scalars(key)).value
    df.to_excel(xlsx_writer, sheet_name=str(band), index=False)
xlsx_writer.close()
