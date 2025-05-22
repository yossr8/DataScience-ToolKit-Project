from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load model
model = joblib.load("models/logistic_regression.pkl")

app = FastAPI(title="Titanic Survival Predictor")

# Define input schema using Pydantic
class Passenger(BaseModel):
    PassengerId: int
    Pclass: int
    Name: str
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Ticket: str
    Fare: float
    Cabin: str
    Embarked: str
@app.get("/")
def read_root():
    return {"Hi": "Alf Mabrook Ya arwaya !!!!!!!!! ;) "}
@app.post("/predict")
def predict_survival(passenger: Passenger):
    # Convert input to DataFrame
    input_data = pd.DataFrame([passenger.dict()])
    
    # Predict using the loaded model
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0].tolist()
    
    return {
        "prediction": int(prediction),
        "probability": {
            "Not Survived": probability[0],
            "Survived": probability[1]
        }
    }
