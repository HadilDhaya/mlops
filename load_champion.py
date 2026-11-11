
import mlflow
import mlflow.sklearn
import pandas as pd

# Same database as Lab 3
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# 1. Load the champion model using its alias
model = mlflow.sklearn.load_model("models:/churn-model@champion")

print("Model loaded :", type(model).__name__)
print("Expected variables :", list(model.feature_names_in_))

# 2. Create a new customer
customer = pd.DataFrame([{
    "age": 35,
    "salaire": 2500,
    "anciennete": 1,
    "consommation": 10,
    "abonnement_Premium": False,
    "abonnement_Standard": True
}])

# 3. Keep only the columns known to the model, in the right order
customer = customer[list(model.feature_names_in_)]

# 4. Predict: 1 = customer leaves, 0 = customer stays
prediction = model.predict(customer)[0]

print("Will the customer leave? ->", "YES" if prediction == 1 else "NO")
