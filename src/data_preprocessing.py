
import os
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_curve, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
import pickle
import os
import warnings
import matplotlib.pyplot as plt
import json
import joblib
import argparse
#from custom_function import *
from load_data import *
from scipy.stats import chi2_contingency
from scipy.stats import f_oneway
from sklearn.preprocessing import StandardScaler, MinMaxScaler

log_dir='logs'
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
f = logging.Formatter("[%(asctime)s: - %(lineno)d  - %(message)s ] - %(filename)s",datefmt='%d-%m-%Y %I:%M:%S %p')#- %(pathname)s: ,f = logging.Formatter("[%(asctime)s: - %(name)s: - %(levelname)s: - %(pathname)s - %(module)s:] - %(filename)s - %(message)s")#
filename = '{}.log'.format(os.path.basename(__file__).split('.py')[0])

os.makedirs(log_dir,exist_ok=True) 
fh = logging.FileHandler(filename=os.path.join(log_dir,filename),mode="a")
fh.setFormatter(f)
logger.addHandler(fh)


def FunctionAnova(inpData, TargetVariable, ContinuousPredictorList):
    logger.info('If the ANOVA P-Value is <0.05, that means we reject H0')
    # Creating an empty list of final selected predictors
    SelectedPredictors=[]
    logger.info('##### ANOVA Results are shown below :- ##### \n')
    for predictor in ContinuousPredictorList:
        CategoryGroupLists=inpData.groupby(TargetVariable)[predictor].apply(list)
        AnovaResults = f_oneway(*CategoryGroupLists)
        if (AnovaResults[1] < 0.05):
            logger.info(f"{predictor}, 'is correlated with', {TargetVariable}, '| P-Value:', {AnovaResults[1]}")
            SelectedPredictors.append(predictor)
        else:
            logger.info(f"{predictor}, 'is NOT correlated with', {TargetVariable}, '| P-Value:', {AnovaResults[1]}")
    return(SelectedPredictors)



def FunctionChisq(inpData, TargetVariable, CategoricalVariablesList):
    # Creating an empty list of final selected predictors
    SelectedPredictors=[]
    logger.info('If the ChiSq P-Value is <0.05, that means we reject H0')
    for predictor in CategoricalVariablesList:
        CrossTabResult=pd.crosstab(index=inpData[TargetVariable], columns=inpData[predictor])
        ChiSqResult = chi2_contingency(CrossTabResult)
        if (ChiSqResult[1] < 0.05):
            logger.info(f"{predictor}, 'is correlated with', {TargetVariable}, '| P-Value:', {ChiSqResult[1]}")
            SelectedPredictors.append(predictor)
        else:
            logger.info(f"{predictor}, 'is NOT correlated with', {TargetVariable}, '| P-Value:', {ChiSqResult[1]}")        
    return(SelectedPredictors)

def Feature_extraction(config_path):
    config = read_params(config_path)
    logger.info("Started reading data for Preprocessing.")
    data_path = config["load_data"]["raw_dataset_csv"]
    df = pd.read_csv(data_path,sep=",",encoding='utf-8',low_memory=False)
    logger.info("Data has been read successfully.")
    logger.info(f"Checking rows and columns in data {df.shape}")
    logger.info(f"Checking null values in data {df.isnull().sum()}")
    logger.info(f"Checking column data type \n{df.dtypes}")
    cols=['status','credit_history','purpose','credit_risk','employment_duration', 'foreign_worker','housing', 'installment_rate','job', 'number_credits', 'other_debtors','other_installment_plans','people_liable','personal_status_sex', 'present_residence','property', 'savings', 'telephone']
    df[cols] = df[cols].astype('category')
    logger.info(f"Checking {df.dtypes}")
    target_variable=config['base']['target_col']
    #target_variable = 'credit_risk'
    logger.info(f"Our dependent variable i.e {target_variable}")
    logger.info("checking which categorical variables are correlated with target variable.")
    categoricalVariables=df.select_dtypes(include='category').columns
    categoricalVariables=categoricalVariables.drop('credit_risk')
    selectedPrdictor=FunctionChisq(inpData=df, TargetVariable=target_variable,CategoricalVariablesList= categoricalVariables)
    logger.info("Now checking which continous variables are correlated with target variable.")
    ContinuousVariables=df.select_dtypes(include='int64').columns
    select_cont_column=FunctionAnova(inpData=df, TargetVariable=target_variable, ContinuousPredictorList=ContinuousVariables)
    logger.info("Selecting final columns for model")
    joinedlist = selectedPrdictor + select_cont_column
    X=df[joinedlist]
    y=df[target_variable]#.values
    # logger.info(f"This are the final columns used for model before one-hot-encoding:= {Data_for_model.columns}")
    logger.info("Treating all nominal variables at once using One-hot-Encoding..")
    nominaldata1=['status','credit_history','purpose','savings','personal_status_sex','other_debtors','other_installment_plans','housing','foreign_worker']
    ohenc = OneHotEncoder(sparse=False, dtype=int).fit(X[nominaldata1])
    encoded_data = ohenc.transform(X[nominaldata1])
    encoded_df = pd.DataFrame(encoded_data, columns = [ f'OHE{i}' for i in range(1, encoded_data.shape[1] + 1)])
    drop_cols = ['employment_duration', 'property', 'id']
    X = X.drop(drop_cols + nominaldata1, axis=1).merge(encoded_df, how='left', left_index=True,right_index=True)
    ohenc_file = config["one_hot_encoding"]["ohencs"]
    with open(ohenc_file, 'wb') as pkl_file:
        pickle.dump(ohenc, pkl_file) 
    logger.info("Exporting Final Trained Model in pickle file.")
    logger.info("Adding Target Variable to the data.")
    x_indept_scaled_var=config["Feature_extraction"]["x_indept_scaled_var"]
    y_dept_var=config["Feature_extraction"]["y_dept_var"]
    y.to_csv(y_dept_var,sep=",",index=False)
    ### Sandardization of data ###
    PredictorScaler = MinMaxScaler().fit(X)
    X_scaled = PredictorScaler.transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    X_scaled.to_csv(x_indept_scaled_var,sep=",",index=False)
    minmaxscaler = config["one_hot_encoding"]["minmaxscaler"]
    with open(minmaxscaler, 'wb') as pkl_file:
        pickle.dump(PredictorScaler, pkl_file)
    logger.info("Data Preprocessing has been completed successfully...")
    return X_scaled, y

def preprocess(_data,config_path):
    config = read_params(config_path)
    ohenc_file = config["one_hot_encoding"]["ohencs"]
    with open(ohenc_file, 'rb') as pkl_file:
        ohenc = pickle.load(pkl_file)
    nominaldata1=['status','credit_history','purpose','savings','personal_status_sex','other_debtors','other_installment_plans','housing','foreign_worker']
    encoded_data = ohenc.transform(_data[nominaldata1])
    encoded_df = pd.DataFrame(encoded_data, columns = [ f'OHE{i}' for i in range(1, encoded_data.shape[1] + 1)])
    Data_for_model_Numeric = _data.reset_index(drop=True).drop(nominaldata1, axis=1).merge(encoded_df, how='left', left_index=True,right_index=True)
    minmaxscaler = config["one_hot_encoding"]["minmaxscaler"]
    with open(minmaxscaler, 'rb') as pkl_file:
        minmaxscaler = pickle.load(pkl_file)
    Data_for_model_Numeric=minmaxscaler.transform(Data_for_model_Numeric)
    return Data_for_model_Numeric


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config',default='params.yaml')
    parsed_args = args.parse_args()
    #ConnectDB().casandra_to_local_get_data(config_path=parsed_args.config)
    x,y=Feature_extraction(config_path=parsed_args.config)
    # x_preprocess=preprocess(x,config_path=parsed_args.config)