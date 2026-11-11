
# ===== 1. Import the libraries =====
from typing import Literal
import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# ===== 2. Create the application =====
app = FastAPI(title="Churn API")

# ===== 3. Load the champion ONLY ONCE, at startup =====
mlflow.set_tracking_uri("sqlite:///mlflow.db")
model = mlflow.sklearn.load_model("models:/churn-model@champion")

# ===== 4. Describe the expected data =====
class Customer(BaseModel):
    age: int
    salaire: float
    abonnement: Literal["Basique", "Standard", "Premium"]
    anciennete: float
    consommation: float

# ===== 5. Route /health: check that the API is alive =====
@app.get("/health")
def health():
    return {"status": "ok"}

# ===== 6. Route /predict: receive a customer, return the prediction =====
@app.post("/predict")
def predict(customer: Customer):

    # a) Turn the received customer into a one-row table
    row = pd.DataFrame([customer.model_dump()])

    # b) Turn the text "abonnement" into 0/1 columns
    row = pd.get_dummies(row)

    # c) Keep the model's columns, in the right order; missing ones are 0
    row = row.reindex(columns=model.feature_names_in_, fill_value=0)

    # d) Predict: 1 = leaves, 0 = stays
    result = int(model.predict(row)[0])

    return {
        "churn": result,
        "message": "The customer will leave" if result == 1 else "The customer stays"
    }
