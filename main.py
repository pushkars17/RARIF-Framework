import pandas as pd
from data_generator import generate_data
from model import train_model
from evaluate import evaluate
from explain import explain
data = pd.read_csv("data/sample_data.csv")

# Step 1: Generate Data
data = generate_data()

# Step 2: Train Model
model, X_test, y_test = train_model(data)

# Step 3: Evaluate
evaluate(model, X_test, y_test)

# Step 4: Explain
explain(model, X_test)