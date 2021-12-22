from fastapi import FastAPI
import uvicorn
import pickle
import numpy as np
import pandas as pd
from pydantic import BaseModel
import pandas as pd
app = FastAPI()

threshold = 0.645379
credit_risk_dict = { 1: 'Good', 0: 'Bad' }

with open('notebooks/MinMaxScaler.pkl', 'rb') as pkl_file:
    MinMaxScaler = pickle.load(pkl_file)

with open('notebooks/rf_Grid.pkl', 'rb') as pkl_file:
    model = pickle.load(pkl_file)

with open('notebooks/ohenc.pkl', 'rb') as pkl_file:
    ohenc = pickle.load(pkl_file)

def preprocess_features(_data):
    # One Hot Encoding
    nominaldata1=['status','credit_history','purpose','savings','personal_status_sex','other_debtors','other_installment_plans','housing','foreign_worker']
    encoded_data = ohenc.transform(_data[nominaldata1])
    encoded_df = pd.DataFrame(encoded_data, columns = [f'OHE_{i}' for i in range(1, encoded_data.shape[1] + 1)])
    non_nominal_df = _data.drop(nominaldata1, axis=1).reset_index(drop=True)
    _data = non_nominal_df.merge(encoded_df, how='left', left_index=True, right_index=True)
    _data=MinMaxScaler.transform(_data)

    return _data

def predict_diff_thresh(pred_probs, thresh):
    return np.where(pred_probs > thresh, 1, 0)

class Request_Data_format(BaseModel):
    status: int
    duration: int
    credit_history: int
    purpose: int
    amount: int
    savings: int
    employment_duration: int
    personal_status_sex: int
    other_debtors: int
    property: int
    age: int
    other_installment_plans: int
    housing: int
    foreign_worker: int




@app.post('/predict')
def predict_species(request_obj: Request_Data_format):
    data = request_obj.dict()
    
    data_in = preprocess_features(pd.DataFrame([data]))

    probability = model.predict_proba(data_in)[:, 1]
    prediction = predict_diff_thresh(probability, threshold)
    return { 'credit_risk': credit_risk_dict[prediction.tolist()[0]] }


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=4000, debug=True)