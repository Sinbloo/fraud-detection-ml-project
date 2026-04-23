from fastapi import FastAPI
import pickle
import pandas as pd
import os

app = FastAPI(title="Fraud Detection API")

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# load model
MODEL_PATH = os.path.join(BASE_DIR, "model", "fraud_detection_model.pkl")
model = pickle.load(open(MODEL_PATH, "rb"))

# load feature names
FEATURE_PATH = os.path.join(BASE_DIR, "model", "feature_names.pkl")
feature_names = pickle.load(open(FEATURE_PATH, "rb"))

@app.get("/")
def home():
    return {"message": "API is running 🚀"}

@app.post("/predict")
def predict(data: dict):

    # convert dict → DataFrame بنفس ترتيب التدريب
    df = pd.DataFrame([data])
    df = df[feature_names]

    prediction = model.predict(df)[0]

    return {
        "prediction": int(prediction),
        "result": "FRAUD 🔴" if prediction == 1 else "NOT FRAUD 🟢"
    }