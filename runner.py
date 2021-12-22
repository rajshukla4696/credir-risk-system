import sys
import os
import pandas as pd
import yaml
import argparse
from connect_database import ConnectDB
from get_data import *
from load_data import *
from split_data import *
import argparse

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config',default='params.yaml')
    parsed_args = args.parse_args()
    #ConnectDB().casandra_to_local_get_data(config_path=parsed_args.config)
    #get_data(config_path= parsed_args.config)
    load_and_save(config_path=parsed_args.config)
    split_and_save_data(config_path=parsed_args.config)