from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

# Initialize FastAPI app
app = FastAPI()

# Load trained model, scaler, and label encoder
model = joblib.load("predictive_maintenance_model.pkl")   # Your trained classification model
scaler = joblib.load("scaler.pkl")  # StandardScaler for numerical features
label_encoder_type = joblib.load("label_encoder_type.pkl")  # LabelEncoder for 'Type'

# Define input data format
class MaintenanceInput(BaseModel):
    Type: str
    Air_temperature: float
    Process_temperature: float
    Rotational_speed: int
    Torque: float
    Tool_wear: int
    
@app.post("/predict/")
def predict_maintenance(data: MaintenanceInput):
    # Convert input to DataFrame
    input_data = pd.DataFrame([data.dict()])

    # Ensure column names match those used during training
    column_mapping = {
        "Air_temperature": "Air temperature [K]",
        "Process_temperature": "Process temperature [K]",
        "Rotational_speed": "Rotational speed [rpm]",
        "Torque": "Torque [Nm]",
        "Tool_wear": "Tool wear [min]",
    }
    input_data.rename(columns=column_mapping, inplace=True)

    # Encode 'Type' using the saved LabelEncoder
    input_data["Type"] = label_encoder_type.transform([input_data["Type"].values[0]])[0]

    # Ensure feature order matches training
    feature_columns = [
        "Type", "Air temperature [K]", "Process temperature [K]", 
        "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"
    ]

    # Scale the input
    input_scaled = scaler.transform(input_data[feature_columns])

    # Make prediction
    prediction = model.predict(input_scaled)

    # Return result
    return {"prediction": int(prediction[0])}
