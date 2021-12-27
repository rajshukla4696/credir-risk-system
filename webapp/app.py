import os
from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import pickle
import joblib
import json

app = Flask(__name__)


## Global Variables
# Directories
pre_dir = os.path.dirname(os.getcwd())
data_dir = os.path.join(pre_dir, 'data_given')
pickle_dir = os.path.join(pre_dir, 'saved_models')

# Input Files
data_fp = os.path.join(data_dir, 'SouthGermanCredit_cassandra.csv')
minmax_fn = os.path.join('static', 'data', 'minmax.json')

threshold = 0.645379
credit_risk_dict = { 1: 'Good', 0: 'Bad' }


if not os.path.isfile(minmax_fn):
    print('JSON not present')
    print('Generating Min Max JSON')
    data = pd.read_csv(data_fp)
    minmaxcols = ['duration', 'amount', 'age']
    with open(minmax_fn, 'w') as json_file:
        data_dict = dict()
        for col in minmaxcols:
            _min = data[col].min()
            _max = data[col].max()
            data_dict[col] = { 'min': int(_min), 'max': int(_max) }
        json.dump(data_dict, json_file)

with open(os.path.join(pickle_dir, 'MinMaxScaler.pkl'), 'rb') as pkl_file:
    MinMaxScaler = pickle.load(pkl_file)

with open(os.path.join(pickle_dir, 'rf_Grid.joblib'), 'rb') as joblib_file:
    model = joblib.load(joblib_file)

with open(os.path.join(pickle_dir, 'ohenc.pkl'), 'rb') as pkl_file:
    ohenc = pickle.load(pkl_file)

def preprocess_features(_data):
    # One Hot Encoding
    nominaldata1=['status','credit_history','purpose','savings','personal_status_sex',
    'other_debtors','other_installment_plans','housing','foreign_worker']
    encoded_data = ohenc.transform(_data[nominaldata1])
    encoded_df = pd.DataFrame(encoded_data, columns = [f'OHE_{i}' for i in range(1, encoded_data.shape[1] + 1)])
    non_nominal_df = _data.drop(nominaldata1, axis=1).reset_index(drop=True)
    _data = non_nominal_df.merge(encoded_df, how='left', left_index=True, right_index=True)
    _data=MinMaxScaler.transform(_data)

    return _data

def predict_diff_thresh(pred_probs, thresh):
    return np.where(pred_probs > thresh, 1, 0)


@app.route("/")
def index():
    with open(minmax_fn, 'r') as json_file:
        minmax_dict = json.load(json_file)
    context = minmax_dict
    return render_template("index.html", **context); 

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    data = request.form.to_dict()
    del data['form_submit']
    data = { k: int(v) for k, v in data.items() }
    
    data_in = preprocess_features(pd.DataFrame([data]))

    probability = model.predict_proba(data_in)[:, 1]
    prediction = predict_diff_thresh(probability, threshold)
    return jsonify({ 'credit_risk': credit_risk_dict[prediction.tolist()[0]] })


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8000, debug=True)