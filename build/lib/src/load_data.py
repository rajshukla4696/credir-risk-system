#read the data from data source 
#save it into data/raw for further process
import sys
import os
import yaml
import argparse
from get_data import *
import pandas as pd
from custom_functions import *
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
def load_and_save(config_path):
    #logg = get_logger(__file__)
    config = read_params(config_path)
    df = get_data(config_path)
    raw_data_path = config["load_data"]["raw_dataset_csv"]
    df.to_csv(raw_data_path,sep=",",index=False)
    logger.info("Load data from remote sources and then saving it in data/raw folder")
    return df



if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config',default='params.yaml')
    parsed_args = args.parse_args()
    #ConnectDB().casandra_to_local_get_data(config_path=parsed_args.config)
    load_and_save(config_path=parsed_args.config)

