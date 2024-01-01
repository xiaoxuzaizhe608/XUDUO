import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import csv
import datetime
from PIL import Image
import math


class Config:
    trainDir = './data/train'
    testDir = 'E:/test_dataset/test_dataset'
    figDir = './figs'
    savePath = 'result.csv'
    trajectoryGridNum = 100
    trajectoryGridStage = 24
    trajectoryGridStep = 1.0 / trajectoryGridStage


config = Config()


def readData(path, model):
    files = os.listdir(path)
    files.sort(key=lambda x: int(x[:-4]))
    data = []

    for i in tqdm(range(len(files)), desc='Reading ' + model + ' data', leave=True, unit='csv', unit_scale=True):
        filePath = os.path.join(path, files[i])
        with open(filePath, encoding='utf-8') as file:
            next(file)
            reader = csv.reader(file)
            data.extend(reader)

    dataFrame = pd.DataFrame(data)
    if model == 'train':
        dataFrame.columns = ['ID', 'lat', 'lon', 'speed', 'dir', 'time', 'type']
    else:
        dataFrame['type'] = 'unknown'
        dataFrame.columns = ['ID', 'lat', 'lon', 'speed', 'dir', 'time', 'type']

    dataFrame = dataFrame.astype({'ID': int, 'lat': float, 'lon': float, 'speed': float, 'dir': float})
    dataFrame['time'] = pd.to_datetime(dataFrame['time'], format='%Y-%m-%d %H:%M:%S')

    return dataFrame


def featureExtraction(dataFrame):
    dataFrame['lat_diff'] = dataFrame.groupby('ID')['lat'].diff(1).fillna(0)
    dataFrame['lon_diff'] = dataFrame.groupby('ID')['lon'].diff(1).fillna(0)
    dataFrame['dir_diff'] = dataFrame.groupby('ID')['dir'].diff(1).fillna(0)

    dataFrame['anchor'] = (dataFrame['speed'] == 0).astype(int)

    for id, group_t in tqdm(dataFrame.groupby('ID'), desc='Extracting features', leave=True, unit='group'):
        group = group_t.reset_index()
        lat_diff_max = group['lat_diff'].max()
        lat_diff_min = group['lat_diff'].min()
        lat_diff_t = lat_diff_max - lat_diff_min
        lon_diff_max = group['lon_diff'].max()
        lon_diff_min = group['lon_diff'].min()
        lon_diff_t = lon_diff_max - lon_diff_min

        group['position_diff_x'] = (config.trajectoryGridNum / 2 * group['lat_diff'] / lat_diff_t).round().astype(int)
        group['position_diff_y'] = (config.trajectoryGridNum / 2 * group['lon_diff'] / lon_diff_t).round().astype(int)

        array = np.ones((config.trajectoryGridNum, config.trajectoryGridNum))

        for _, row in group.iterrows():
            x = int(row['position_diff_x'] + config.trajectoryGridNum / 2 - 1)
            y = int(row['position_diff_y'] + config.trajectoryGridNum / 2 - 1)
            array[x][y] -= config.trajectoryGridStep

            if row['anchor'] == 1 or math.fabs(row['dir_diff']) >= 180:
                array[x][y] -= 4 * config.trajectoryGridStep

        path = os.path.join(config.figDir, group['type'][0].lower(), f'{id}.bmp')
        outputImg = Image.fromarray(array * 255.0)
        outputImg = outputImg.convert('L')
        outputImg.save(path)


if __name__ == "__main__":
    for dir_name in ['weiwang', 'ciwang', 'tuowang', 'test']:
        if not os.path.exists(os.path.join(config.figDir, dir_name)):
            os.makedirs(os.path.join(config.figDir, dir_name))

    train = readData(config.testDir, 'test')
    featureExtraction(train)
