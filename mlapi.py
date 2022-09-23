# Kütüphaneler
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle



app = FastAPI()

class ScroingItem(BaseModel):
        gender:str 
        SeniorCitizen:int
        Partner:str
        Dependents:str
        tenure:int
        PhoneService:str
        MultipleLines:str
        InternetService:str
        OnlineSecurity:str
        OnlineBackup:str
        DeviceProtection:str
        TechSupport:str
        StreamingTV:str
        StreamingMovies:str
        Contract:str
        PaperlessBilling:str
        PaymentMethod:str
        MonthlyCharges:float
        TotalCharges:str

with open('xgboost_pickle.pkl','rb') as f:
    model = pickle.load(f)

@app.post('/')
async def scoring_endpoint(item:ScroingItem):
    df = pd.DataFrame([item.dict().values()], columns=item.dict().keys())
    yhat = model.predict(df)
    return {"prediction":yhat}