#split the raw data 
# save it in data/processed folder

import os
import argparse
import pandas as pd
from data_preprocessing import *
from sklearn.model_selection import train_test_split
from get_data import read_params

import logging

log_dir='logs'
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
f = logging.Formatter("[%(asctime)s: - %(levelname)s: %(lineno)d:] - %(filename)s - %(message)s",datefmt='%d-%m-%Y %I:%M:%S %p')#- %(pathname)s: ,f = logging.Formatter("[%(asctime)s: - %(name)s: - %(levelname)s: - %(pathname)s - %(module)s:] - %(filename)s - %(message)s")#
filename = '{}.log'.format(os.path.basename(__file__).split('.py')[0])

os.makedirs(log_dir,exist_ok=True) 
fh = logging.FileHandler(filename=os.path.join(log_dir,filename),mode="a")
fh.setFormatter(f)
logger.addHandler(fh)


def split_and_save_data(config_path):
    config = read_params(config_path)
    test_data_path = config["split_data"]["test_path"]
    train_data_path = config["split_data"]["train_path"]
    raw_data_path = config["Feature_extraction"]["x_indept_scaled_var"]
    split_ratio = config["split_data"]["test_size"]
    random_state = config["base"]["random_state"]

    X,y=Feature_extraction(config_path)
    logger.info("after data preprocessing data is been extracted..")
    df = X.merge(pd.DataFrame(y), how='left', left_index=True, right_index=True)

    train,test = train_test_split(df,test_size=split_ratio,random_state=random_state)
    logger.info("Spliting of data into train and test is been completed..")
    train.to_csv(train_data_path,sep=",",index=False)
    test.to_csv(test_data_path,sep=",",index=False)
    logger.info("train and test is been stored in folder")


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config',default='params.yaml')
    parsed_args = args.parse_args()
    split_and_save_data(config_path=parsed_args.config)