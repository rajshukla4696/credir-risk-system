## read params
## process 
## return data frame
import sys
import os
import yaml
import pandas as pd
import argparse
#import connect_database
#from logger import get_logger
from custom_function import *
import logging


log_dir='logs'
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
f = logging.Formatter("[%(asctime)s: - %(levelname)s: %(lineno)d:] - %(filename)s - %(message)s",datefmt='%d-%m-%Y %I:%M:%S %p')#- %(pathname)s: ,f = logging.Formatter("[%(asctime)s: - %(name)s: - %(levelname)s: - %(pathname)s - %(module)s:] - %(filename)s - %(message)s")#
filename = '{}.log'.format(os.path.basename(__file__).split('.py')[0])
os.makedirs(log_dir,exist_ok=True) 
fh = logging.FileHandler(filename=os.path.join(log_dir,filename),mode="a")#"get_data.log"
fh.setFormatter(f)
fh.setFormatter(f)
logger.addHandler(fh)

def get_data(config_path):
    
    config = read_params(config_path)
    logger.info("Started getting data from given folder...")
    data_path = config["data_sources"]["cassandra_to_local_path"]
    df = pd.read_csv(data_path,sep=",",encoding='utf-8',low_memory=False)
    logger.info("Data has been successfully readed from the given folder.")
    return df
    

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config",default="params.yaml")
    parsed_args = args.parse_args()
    get_data(config_path= parsed_args.config)
