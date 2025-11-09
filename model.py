import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import joblib
import json
import warnings
import os
warnings.filterwarnings('ignore')

# Get the absolute path of the project directory
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

print("Loading and processing all datasets...")

# Load all datasets
crop_recommendation_df = pd.read_csv(os.path.join(project_dir, 'next_crop_prediction.csv'))
fertilizer_df = pd.read_csv(os.path.join(project_dir, 'fertilizer_and_pesticide_recommend.csv'))
price_risk_df = pd.read_csv(os.path.join(project_dir, 'price_risk_coaching_dataset_50000.csv'))
production_df = pd.read_csv(os.path.join(project_dir, 'production_estimation.csv'))

print(f"Crop recommendation dataset: {crop_recommendation_df.shape}")
print(f"Fertilizer dataset: {fertilizer_df.shape}")
print(f"Price risk dataset: {price_risk_df.shape}")
print(f"Production dataset: {production_df.shape}")

# 1. CROP RECOMMENDATION MODEL
print("\n=== CROP RECOMMENDATION MODEL ===")
X_crop = crop_recommendation_df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y_crop = crop_recommendation_df['label']

X_train_crop, X_test_crop, y_train_crop, y_test_crop = train_test_split(X_crop, y_crop, test_size=0.2, random_state=42)

crop_model = RandomForestClassifier(n_estimators=100, random_state=42)
crop_model.fit(X_train_crop, y_train_crop)

y_pred_crop = crop_model.predict(X_test_crop)
crop_accuracy = accuracy_score(y_test_crop, y_pred_crop)
print(f"Crop recommendation accuracy: {crop_accuracy:.2f}")

# 2. FERTILIZER RECOMMENDATION MODEL
print("\n=== FERTILIZER RECOMMENDATION MODEL ===")
# Prepare fertilizer data
fertilizer_features = ['Crop', 'State', 'Area', 'Annual_Rainfall', 'Yield']
fertilizer_target = 'Fertilizer'

# Encode categorical variables
le_crop = LabelEncoder()
le_state = LabelEncoder()

fertilizer_df_clean = fertilizer_df.dropna()
fertilizer_df_clean['Crop_encoded'] = le_crop.fit_transform(fertilizer_df_clean['Crop'])
fertilizer_df_clean['State_encoded'] = le_state.fit_transform(fertilizer_df_clean['State'])

X_fert = fertilizer_df_clean[['Crop_encoded', 'State_encoded', 'Area', 'Annual_Rainfall', 'Yield']]
y_fert = fertilizer_df_clean['Fertilizer']

X_train_fert, X_test_fert, y_train_fert, y_test_fert = train_test_split(X_fert, y_fert, test_size=0.2, random_state=42)

fertilizer_model = RandomForestRegressor(n_estimators=100, random_state=42)
fertilizer_model.fit(X_train_fert, y_train_fert)

y_pred_fert = fertilizer_model.predict(X_test_fert)
fertilizer_r2 = r2_score(y_test_fert, y_pred_fert)
print(f"Fertilizer recommendation R² score: {fertilizer_r2:.2f}")

# 3. PRICE PREDICTION MODEL
print("\n=== PRICE PREDICTION MODEL ===")
price_features = ['Current_Price', 'Storage_Cost_per_Day', 'Daily_Loss_%', 'Interest_Rate_Monthly_%', 'Quantity_Qtl']
price_target = 'Predicted_Price_15D'

X_price = price_risk_df[price_features]
y_price = price_risk_df[price_target]

X_train_price, X_test_price, y_train_price, y_test_price = train_test_split(X_price, y_price, test_size=0.2, random_state=42)

price_model = RandomForestRegressor(n_estimators=100, random_state=42)
price_model.fit(X_train_price, y_train_price)

y_pred_price = price_model.predict(X_test_price)
price_r2 = r2_score(y_test_price, y_pred_price)
print(f"Price prediction R² score: {price_r2:.2f}")

# 4. PRODUCTION ESTIMATION MODEL
print("\n=== PRODUCTION ESTIMATION MODEL ===")
prod_features = ['Area_ha', 'N_req_kg_per_ha', 'P_req_kg_per_ha', 'K_req_kg_per_ha', 'Temperature_C', 'Humidity_%', 'pH', 'Rainfall_mm']
prod_target = 'Yield_kg_per_ha'

production_df_clean = production_df.dropna()
X_prod = production_df_clean[prod_features]
y_prod = production_df_clean[prod_target]

X_train_prod, X_test_prod, y_train_prod, y_test_prod = train_test_split(X_prod, y_prod, test_size=0.2, random_state=42)

production_model = RandomForestRegressor(n_estimators=100, random_state=42)
production_model.fit(X_train_prod, y_train_prod)

y_pred_prod = production_model.predict(X_test_prod)
prod_r2 = r2_score(y_test_prod, y_pred_prod)
print(f"Production estimation R² score: {prod_r2:.2f}")

# Save models and encoders
print("\n=== SAVING MODELS ===")
import pickle
with open('crop_recommendation_model.pkl', 'wb') as f:
    pickle.dump(crop_model, f)
with open('fertilizer_model.pkl', 'wb') as f:
    pickle.dump(fertilizer_model, f)
with open('price_model.pkl', 'wb') as f:
    pickle.dump(price_model, f)
with open('production_model.pkl', 'wb') as f:
    pickle.dump(production_model, f)
with open('label_encoder_crop.pkl', 'wb') as f:
    pickle.dump(le_crop, f)
with open('label_encoder_state.pkl', 'wb') as f:
    pickle.dump(le_state, f)
with open('crop_features.pkl', 'wb') as f:
    pickle.dump(X_crop.columns.tolist(), f)
with open('fertilizer_features.pkl', 'wb') as f:
    pickle.dump(fertilizer_features, f)
with open('price_features.pkl', 'wb') as f:
    pickle.dump(price_features, f)
with open('production_features.pkl', 'wb') as f:
    pickle.dump(prod_features, f)

# Save feature information for API
feature_info = {
    'crop_features': ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'],
    'fertilizer_features': ['Crop', 'State', 'Area', 'Annual_Rainfall', 'Yield'],
    'price_features': ['Current_Price', 'Storage_Cost_per_Day', 'Daily_Loss_%', 'Interest_Rate_Monthly_%', 'Quantity_Qtl'],
    'production_features': ['Area_ha', 'N_req_kg_per_ha', 'P_req_kg_per_ha', 'K_req_kg_per_ha', 'Temperature_C', 'Humidity_%', 'pH', 'Rainfall_mm'],
    'crop_labels': le_crop.classes_.tolist(),
    'state_labels': le_state.classes_.tolist()
}

with open('feature_info.json', 'w') as f:
    json.dump(feature_info, f)

print("All models saved successfully!")
print(f"Crop recommendation model accuracy: {crop_accuracy:.2f}")
print(f"Fertilizer model R²: {fertilizer_r2:.2f}")
print(f"Price model R²: {price_r2:.2f}")
print(f"Production model R²: {prod_r2:.2f}")