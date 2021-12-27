#load the train and test data
#train algo
#save the metrics and params

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
import logging
from custom_function import *
#from get_data import read_params
log_dir='logs'
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
f = logging.Formatter("[%(asctime)s: - %(levelname)s: %(lineno)d:] - %(filename)s - %(message)s",datefmt='%d-%m-%Y %I:%M:%S %p')#- %(pathname)s: ,f = logging.Formatter("[%(asctime)s: - %(name)s: - %(levelname)s: - %(pathname)s - %(module)s:] - %(filename)s - %(message)s")#
filename = '{}.log'.format(os.path.basename(__file__).split('.py')[0])

os.makedirs(log_dir,exist_ok=True) 
fh = logging.FileHandler(filename=os.path.join(log_dir,filename),mode="a")
fh.setFormatter(f)
logger.addHandler(fh)

def predict_diff_thresh(pred_probs, thresh):
        return np.where(pred_probs > thresh, 1, 0)

def train_and_evaluate(config_path):
    config = read_params(config_path)
    test_data_path = config["split_data"]["test_path"]
    train_data_path = config["split_data"]["train_path"]
    random_state = config["base"]["random_state"]
    model_dir = config["model_dir"]
    print(model_dir)
    target = [config["base"]["target_col"]]
    train = pd.read_csv(train_data_path, sep=",")
    test = pd.read_csv(test_data_path, sep=",")
    logger.info("Reading of train and test data is completed")
    train_y = train[target]
    test_y = test[target]

    train_x = train.drop(target, axis=1)
    test_x = test.drop(target, axis=1)
    logger.info("Number of trees in random ")
    n_estimators = [int(x) for x in np.linspace(start = 10, stop = 50, num = 5)]
    logger.info(f"{n_estimators}")
    max_features = ['auto', 'sqrt']
    logger.info(f"Number of features to consider at every split {max_features}")
    max_depth = [2,4]
    logger.info(f"Maximum number of levels in tree : {max_depth}")
    min_samples_split = [2, 5]
    logger.info(f"Minimum number of samples required to split a node {min_samples_split}")
    min_samples_leaf = [1, 2]
    logger.info(f"Minimum number of samples required at each leaf node {min_samples_leaf}")
    bootstrap = [True, False]
    logger.info(f"Method of selecting samples for training each tree {bootstrap}")
    logger.info("Create the param grid")
    param_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}
    logger.info(f"{param_grid}")
    rf_Model = RandomForestClassifier()
    logger.info(f"Applying random forest..")
    from sklearn.model_selection import GridSearchCV
    cv=config["estimators"]["GridSearch_rf"]["params"]["cv"]
    verbose=config["estimators"]["GridSearch_rf"]["params"]["verbose"]
    n_jobs=config["estimators"]["GridSearch_rf"]["params"]["n_jobs"]
    refit=config["estimators"]["GridSearch_rf"]["params"]["refit"]
    rf_Grid = GridSearchCV(estimator = rf_Model, param_grid = param_grid, cv = cv, verbose=verbose, n_jobs = n_jobs,refit = refit)
    rf_Grid.fit(train_x, train_y)
    logger.info("Model fitting is completed..")
    logger.info(f"\n These are best parameters: \n{rf_Grid.best_params_}")
    params_file = config["report"]["params"]
    with open(params_file, "w") as f:
        json.dump(rf_Grid.best_params_, f, indent=4)

    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "rf_Grid.joblib")
    joblib.dump(rf_Grid, model_path)
    logger.info("Model is dumped successfully..")
    #params_file = config["reports"]["params"]
    prediction_train=rf_Grid.predict(train_x)
    prediction_test=rf_Grid.predict(test_x)
    
    logger.info (f'\n Train Accuracy - : {rf_Grid.score(train_x,train_y):.3f}')
    logger.info(f'\n{metrics.confusion_matrix(train_y,prediction_train)}')
    logger.info (f'\n Test Accuracy - : {rf_Grid.score(test_x,test_y):.3f}')
    logger.info(f'\n{metrics.confusion_matrix(test_y,prediction_test)}')
    

    ## Code for Getting Different threshold

    fpr, tpr, thresholds = roc_curve(test_y, rf_Grid.predict_proba(test_x)[:, 1])
    # print(thresholds)

    accuracy_ls = []
    for thres in thresholds:
        data_dict = dict()
        data_dict['threshold'] = thres
        y_pred = np.where(prediction_test > thres,1,0)
        data_dict['f1_score'] = f1_score(test_y, y_pred)
        accuracy_ls.append(data_dict)
        
    accuracy_ls = pd.DataFrame(accuracy_ls)
    accuracy_ls = accuracy_ls.sort_values('f1_score', ascending=False).reset_index(drop=True)
    logger.info (f'\n Best Threshold values for our model :- \n{accuracy_ls.head()}')
    prediction_train_thres=rf_Grid.predict_proba(train_x)
    prediction_test_thres=rf_Grid.predict_proba(test_x)
    prediction_train_thres1=predict_diff_thresh(prediction_train_thres[:, 1],accuracy_ls['threshold'][0])
    prediction_test_thres1=predict_diff_thresh(prediction_test_thres[:, 1],accuracy_ls['threshold'][0])
    logger.info (f'\n Train Accuracy - : {rf_Grid.score(train_x,train_y):.3f}')
    logger.info(f'\n{metrics.confusion_matrix(train_y,prediction_train_thres1)}')
    logger.info(f'\n{metrics.classification_report(train_y, prediction_train_thres1)}')
    logger.info (f'Test Accuracy - : {rf_Grid.score(test_x,test_y):.3f}')
    logger.info(f'\n{metrics.confusion_matrix(test_y,prediction_test_thres1)}')
    logger.info(f'\n{metrics.classification_report(test_y, prediction_test_thres1)}')    

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config',default='params.yaml')
    parsed_args = args.parse_args()
    #ConnectDB().casandra_to_local_get_data(config_path=parsed_args.config)
    train_and_evaluate(config_path=parsed_args.config)