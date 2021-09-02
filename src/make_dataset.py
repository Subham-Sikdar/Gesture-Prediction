# -*- coding: utf-8 -*-
import os
import datetime
import pandas as pd
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
from config.config import read_params


def download_and_unzip():
    """
    Download and extract data from given URL
    """
    config = read_params()
    print("Downloading dataset")
    http_response = urlopen(config['data']['url'])
    zipfile = ZipFile(BytesIO(http_response.read()))
    zipfile.extractall(path=config['data']['raw'])
    print("Downloading dataset complete")


def make_dir():
    """
    Check and create directory to store data if does not exist
    """
    config = read_params()
    if not os.path.isdir(os.path.join(config['data']['raw'], "EMG_data_for_gestures-master")):
        download_and_unzip()
    if not os.path.isdir(config['data']['external']):
        os.mkdir(config['data']['external'])
    if not os.path.isdir(config['data']['interim']):
        os.mkdir(config['data']['interim'])
    if not os.path.isdir(config['data']['processed']):
        os.mkdir(config['data']['processed'])


def create_csv():
    """
    Create CSV file out of text files stored in various folder
    """
    config = read_params()
    make_dir()
    print("Creating dataframe")
    sub_folders = list(
        os.walk(os.path.join(config['data']['raw'], "EMG_data_for_gestures-master")))[1:]
    file_paths = []
    for sub_folder in sub_folders:
        for file in sub_folder[2]:
            file_paths.append(sub_folder[0] + "\\" + file)
    dataframes = []
    for file_path in file_paths:
        df = pd.read_csv(file_path, delimiter='\t')
        dataframes.append(df)
    final_df = pd.concat(dataframes)
    datetime_ = datetime.datetime.now().strftime("%d-%m-%y-%H-%M-%S")
    file_name = f"{config['data']['interim']}\\gesture-prediction-{datetime_}.csv"
    final_df.to_csv(file_name, index=False)
    print(f"Dataframe created on {file_name}")

    return final_df


if __name__ == '__main__':
    create_csv()
