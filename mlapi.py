# Kütüphaneler
from pickletools import uint8
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle



app = FastAPI()

class ScroingItem(BaseModel):
    gender:int
    Partner:int
    Dependents:int
    tenure:float
    PaperlessBilling:int
    MonthlyCharges:float
    PhoneService:str
    TotalCharges:str
   NEW_TotalServices
   NEW_AVG_Charges
   NEW_AVG_Service_Fee
   MultipleLines_No phone service
   MultipleLines_Yes
   InternetService_Fiber optic
   InternetService_No
   OnlineSecurity_No internet service
   OnlineSecurity_Yes
   OnlineBackup_No internet service
   OnlineBackup_Yes
   DeviceProtection_No internet service
   DeviceProtection_Yes
   TechSupport_No internet service
   TechSupport_Yes
   StreamingTV_No internet service
   StreamingTV_Yes
   StreamingMovies_No internet service
   StreamingMovies_Yes
   Contract_One year
   Contract_Two year
   PaymentMethod_Credit card (automatic)
   PaymentMethod_Electronic check
   PaymentMethod_Mailed check
   TENURE_CAT_NEW_Very_short
   TENURE_CAT_NEW_long1
   TENURE_CAT_NEW_long2
   TENURE_CAT_NEW_medium1
   TENURE_CAT_NEW_medium2
   TENURE_CAT_NEW_medium3
   MONTHLY_CHARGES_NEW_economy
   MONTHLY_CHARGES_NEW_platinium
   MONTHLY_CHARGES_NEW_premium
   MONTHLY_CHARGES_NEW_upper_eco
   MONTHLY_CHARGES_NEW_x_platinum
   AgeGender_Senior_Male
   AgeGender_Young_Female
   AgeGender_Young_Male
   TotalCharges_New_premium
   Dependent_contracts_No_Two year
   TotalCharges_New_prex2
   TotalCharges_New_prex3
   Dependent_contracts_No_One year
   Dependent_contracts_Yes_Month-to-month
   Dependent_contracts_Yes_One year
   Dependent_contracts_Yes_Two year
   SeniorCitizen_1
   Contract_Extra_Services_1
   Contract_Extra_Services_2
   Contract_Extra_Services_3
   Contract_Extra_Services_4
   Contract_Extra_Services_5
   Contract_Extra_Services_6
   Contract_Extra_Services_7
   NEW_Engaged_1
   NEW_noProt_1
   NEW_Young_Not_Engaged_1                   
   NEW_FLAG_ANY_STREAMING_1
   NEW_FLAG_AutoPayment_1:
   Internet_Service_1.0:

with open('xgboost_pickle.pkl','rb') as f:
    model = pickle.load(f)

@app.post('/')
async def scoring_endpoint(item:ScroingItem):
    df = pd.DataFrame([item.dict().values()], columns=item.dict().keys())
    yhat = model.predict(df)
    return {"prediction":yhat}