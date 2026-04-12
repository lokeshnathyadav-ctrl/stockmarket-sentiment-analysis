import os
import requests
import sklearn
import torch
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
from sklearn.metrics import(
    confusion_matrix, 
    classification_report, 
    recall_score, 
    precision_score, 
    accuracy_score, 
    f1_score)
from huggingface_hub import login,HfApi
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline

#if "GITHUB_WORKSPACE" in os.environ:
#    base_path = os.environ["GITHUB_WORKSPACE"]
#else:
#    base_path = os.getcwd()

#mlflow.set_tracking_uri(f"file:{os.path.join(base_path,'mlruns')}")
#mlflow.set_experiment("P666-training-experiment")

api = HfApi(token=os.getenv("hf-api-key"))

Xtrain_path = "hf://datasets/Lokeshnathy/Stock-Market-News-Data/Xtrain.csv"
Xtest_path = "hf://datasets/Lokeshnathy/Stock-Market-News-Data/Xtest.csv"
ytrain_path = "hf://datasets/Lokeshnathy/Stock-Market-News-Data/ytrain.csv"
ytest_path = "hf://datasets/Lokeshnathy/Stock-Market-News-Data/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)

dependent_variable = 'Label'
independent_variables= Xtrain.columns

preprocessor = make_column_transformer(StandardScaler(),independent_variables)

gb_model = GradientBoostingClassifier(random_state=42)

param_grid = {
    'gradientboostingclassifier__n_estimators':[90,120,150],
    'gradientboostingclassifier__min_samples_leaf':[0.5,0.6,0.7,0.8],
    'gradientboostingclassifier__max_features':[0.8,1,'sqrt','log2'],
    'gradientboostingclassifier__max_depth':[3,5,7,10],
    'gradientboostingclassifier__learning_rate':[0.001,0.05,0.1],
    'gradientboostingclassifier__min_samples_split':[0.5,0.8,1.0],
    'gradientboostingclassifier__n_iter_no_change':[100,150]}
model_pipeline = make_pipeline(preprocessor,gb_model)
grid_search = RandomizedSearchCV(model_pipeline, param_grid,cv=5,n_jobs=-1)
grid_search.fit(Xtrain,ytrain)
grid_search.best_params_
best_model = grid_search.best_estimator_
best_model
classification_threshold=0.45
y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)
y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)
print(classification_report(ytrain,y_pred_train))
print(classfication_report(ytest,y_pred_test))
joblib.dump(best_model, "stk_mrkt_ns_anlzr_v1.joblib")
# Upload to Hugging face
repo_id = "Lokeshnathy/Stock-market-news-Analyzer"
repo_type = "model"                                 

# To Check if space exists
try:
    api.repo_info(repo_id=repo_id,repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id,repo_type=repo_type,private=False)
    print(f"Space '{repo_id}' created.")

# Uploading serialized model to HF Hub
api.upload_file(
    path_or_fileobj="stk_mrkt_ns_anlzr_v1.joblib",
    path_in_repo="stk_mrkt_ns_anlzr_v1.joblib",
    repo_id = repo_id,
    repo_type=repo_type)
