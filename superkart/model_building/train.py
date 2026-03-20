# for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
from sklear.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error)
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError

api = HfApi()

Xtrain_path = "hf://datasets/celesqa/Superkart_MLOPS/Xtrain.csv"
Xtest_path = "hf://datasets/celesqa/Superkart_MLOPS/Xtest.csv"
ytrain_path = "hf://datasets/celesqa/Superkart_MLOPS/ytrain.csv"
ytest_path = "hf://datasets/celesqa/Superkart_MLOPS/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)


# One-hot encode 'Type' and scale numeric features
numeric_features = [
    'Product_Weight',
    'Product_Allocated_Area'
    'Product_MRP',
    'Store_Establishment_Year',
    ]
categorical_features = [
    'Product_Sugar_Content',
    'Product_Type',
    'Store_Size',
    'Store_Location_City_Type',
    'Store_Type'
    ]

# Preprocessing pipeline
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# Define XGBoost model
rf_model = RandomForestRegressor(random_state=42)

# Define hyperparameter grid
param_grid = {
    'randomforestregressor__max_depth':[3, 4, 5, 6],
    'randomforestregressor__max_features': ['sqrt','log2',None],
    'randomforestregressor__n_estimators': [50, 75, 100, 125, 150]
}

# Create pipeline
model_pipeline = make_pipeline(preprocessor, rf_model)

# Grid search with cross-validation
grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring='r2_score', n_jobs=-1)
grid_search.fit(Xtrain, ytrain)

# Best model
best_model = grid_search.best_estimator_
print("Best Params:\n", grid_search.best_params_)

# Predict on training set
y_pred_train = best_model.predict(Xtrain)

# Predict on test set
y_pred_test = best_model.predict(Xtest)

# Save best model
joblib.dump(best_model, "superkart_v1.joblib")

# Upload to Hugging Face
repo_id = "celesqa/Superkart_MLOPS"
repo_type = "model"

api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Model Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Model Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Model Space '{repo_id}' created.")

# create_repo("best_machine_failure_model", repo_type="model", private=False)
api.upload_file(
    path_or_fileobj="superkart_v1.joblib",
    path_in_repo="superkart_v1.joblib",
    repo_id=repo_id,
    repo_type=repo_type,
)
