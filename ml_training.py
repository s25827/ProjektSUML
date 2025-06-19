import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import pickle, shap

RENT = True # True for training the rent model, False for the sale model

# Reading the dataset
base_data = pd.read_csv(f"apartments_{'rent_' if RENT else ''}pl_2024_06.csv")

# Removing unnecessary columns
# "id", "latitude" and "longitude" are useless for general price prediction
data = base_data.drop(columns=["id","latitude","longitude"])

#We don't need to remove na values, as xgboost can handle them
# Encoding categorical variables
for column in ["city", "type", "ownership", "buildingMaterial", "condition", "hasParkingSpace", "hasBalcony", "hasElevator", "hasSecurity", "hasStorageRoom"]:
    data[column] = data[column].astype("category")

# We will use RandomForestRegressor since we need to predict a continuous value (price)
x_train, x_test, y_train, y_test = train_test_split(data.drop(columns=["price"]), data["price"], test_size = 0.2, random_state = 0)

# Training the model
regressor = XGBRegressor(enable_categorical=True)
regressor.fit(x_train, y_train)
print(f"\nTraining score: {regressor.score(x_train, y_train)}\nTest score: {regressor.score(x_test, y_test)}")

# Saving the model
pickle.dump(regressor, open(f"{'rent_' if RENT else ''}model_xgb_original.sv",'wb'))
pickle.dump(data.drop(columns=["price"]), open(f"explainer_data.shap",'wb'))

category_mappings = {}
for col in ["city", "type", "ownership", "buildingMaterial", "condition", "hasParkingSpace", "hasBalcony", "hasElevator", "hasSecurity", "hasStorageRoom"]:
    category_mappings[col] = data[col].cat.categories.tolist()

# Save this dict somewhere, e.g. as a JSON or pickle file
import json
with open("category_mappings.json", "w") as f:
    json.dump(category_mappings, f)