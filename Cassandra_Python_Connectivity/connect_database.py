
import sys
import os
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
import pandas as pd
import yaml
import argparse
from datetime import datetime
from custom_functions import *


import logging
log_dir='logs'
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
f = logging.Formatter("[%(asctime)s: - %(levelname)s: %(lineno)d:] - %(filename)s - %(message)s",datefmt='%d-%m-%Y %I:%M:%S %p')#- %(pathname)s: ,f = logging.Formatter("[%(asctime)s: - %(name)s: - %(levelname)s: - %(pathname)s - %(module)s:] - %(filename)s - %(message)s")#
#filename = '{}.log'.format(os.path.basename('filename').split('.py')[0])

os.makedirs(log_dir,exist_ok=True) 
fh = logging.FileHandler(filename=os.path.join(log_dir,"connect_database.log"),mode="w")
fh.setFormatter(f)
logger.addHandler(fh)
class ConnectDB:

    def __init__(self):
        #self.log = get_logger(__file__)
        pass
    
    def casandra_to_local_get_data(self,config_path):
        config = read_params(config_path)
        
        logger.info("Accessing cassandra connectivity...")
        secure_connect_bundles = config["cassandra_connectivity"]["secure_connect_bundle"]
        cloud_config= {'secure_connect_bundle': secure_connect_bundles}
        secure_connect_bundles = config["cassandra_connectivity"]["secure_connect_bundle"]
        ASTRA_CLIENT_ID = config["cassandra_connectivity"]["ASTRA_CLIENT_ID"]
        ASTRA_CLIENT_SECRET = config["cassandra_connectivity"]["ASTRA_CLIENT_SECRET"]
        auth_provider = PlainTextAuthProvider(ASTRA_CLIENT_ID,ASTRA_CLIENT_SECRET)
        cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
        logger.info( "Connecting cassandra db is done...")
        cassandra_DB_Name = config["data_sources"]["cassandra_DB_Name"]
        session = cluster.connect(cassandra_DB_Name)
        cassandra_Table_Name = config["data_sources"]["cassandra_Table_Name"]
        names=cassandra_Table_Name
        query = config["data_sources"]["query"]
        querys=query.format(names)
        session.row_factory = pandas_factory
        session.default_fetch_size = None
        rows = session.execute(querys)
        df = rows._current_rows
        logger.info( "Accessing rows from cassandra db is done...")
        casandra_to_local_path = config["data_sources"]["cassandra_to_local_path"]
        
        df.to_csv(casandra_to_local_path,index=False)
        logger.info( "Data has been successfully saved to given folder.")
        #self.log.disabled = True
        return df


# if __name__ == '__main__':
#     args = argparse.ArgumentParser()
#     args.add_argument("--config",default="params.yaml")
#     parsed_args = args.parse_args()
#     ConnectDB().casandra_to_local_get_data(config_path=parsed_args.config)
