import os
import requests
import sklearn
import torch
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn import metrics
from sklearn.metrics import(
    confusion_matrix, 
    classification_report, 
    recall_score, 
    precision_score, 
    accuracy_score, 
    f1_score)
from huggingface_hub import login,HfApi
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.model_selection import RandomizedSearchCV
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow
import datasets
from datetime import datetime

# Setting the tracking URL for MLflow & defining name of the experiment
#mlflow.set_tracking_uri("https://localhost:5000")
if "GITHUB_WORKSPACE" in os.environ:
    base_path = os.environ["GITHUB_WORKSPACE"]
else:
    base_path = os.getcwd()

mlflow.set_tracking_uri(f"file:{os.path.join(base_path,'mlruns')}")
mlflow.set_experiment("NLP-Experiment-B31")

api = HfApi(token=os.getenv("HF_TOKEN"))

# defining the path to access the splitted datasets
Xtrain_path = "hf://datasets/Lokeshnathy/Stock-Market-News-Data/Xtrain.csv"
Xtest_path = "hf://datasets/Lokeshnathy/Stock-Market-News-Data/Xtest.csv"
ytrain_path = "hf://datasets/Lokeshnathy/Stock-Market-News-Data/ytrain.csv"
ytest_path = "hf://datasets/Lokeshnathy/Stock-Market-News-Data/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)

gb_transformer = GradientBoostingClassifier(random_state=42)
param_grid = {
    'gradientboostingclassifier__n_estimators':[50,100],
    'gradientboostingclassifier__max_depth':[9,10,11],
    'gradientboostingclassifier__min_samples_leaf':[8,9,10],
    'gradientboostingclassifier__max_features':[0.9,1],
    'gradientboostingclassifier__learning_rate':[0.05,0.1],
}
model_pipeline = make_pipeline(preprocessor,gb_transformer)
with mlflow.start_run():
    random_search = RandomizedSearchCV(model_pipeline,
                                     param_grid,
                                     scoring='recall',
                                     cv=5,
                                     n_jobs=-1)
    random_search.fit(Xtrain,ytrain)
    results = random_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]
        std_score = results['std_test_score'][i]
        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_test_score",mean_score)
            mlflow.log_metric("std_test_score",std_score)
    mlflow.log_params(random_search.best_params_)
    best_model = random_search.best_estimator_
    classification_threshold = 0.45
    y_pred_train_proba = best_model.predict_proba(Xtrain)[:,1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)
    y_pred_test_proba = best_model.predict_proba(Xtest)[:,1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)  
    test_accuracy = accuracy_score(ytest,y_pred_test)
    test_f1 = f1_score(ytest,y_pred_test,average='weighted')
    test_recall = recall_score(ytest,y_pred_test,average ='weighted')
    test_report = classification_report(ytest,y_pred_test,output_dict=True,zero_division=1.0)
    mlflow.log_metrics({
        "accuracy":test_accuracy,
        "f1":test_f1,
        "recall":test_recall,
        "test_precision_label_I": test_report['1']['precision'],
        "test_precision_label_O": test_report['0']['precision'],
        "test_recall_label_I":test_report['1']['recall'],
        "test_recall_label_O":test_report['0']['recall'],
    })
    model_path = "best_model_for_stock_news_analyze_v1.joblib"
    joblib.dump(best_model,model_path)
    mlflow.log_artifact(model_path,artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")
    # Uploading best model to Hugging Face
    repo_id = "Lokeshnathy/Stock-market-news-Analyzer"
    repo_type = "model"
    try:
        api.repo_info(repo_id=repo_id,repo_type=repo_type)
        print(f"Space '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Space '{repo_id}' not found. Creating new space...")
        create_repo(repo_id=repo_id,repo_type=repo_type,private=False)
        print(f"Space '{repo_id}' created.")

    # Uploading serialized model to HF Hub
    api.upload_file(
        path_or_fileobj="best_model_for_stock_news_analyze_v1.joblib",
        path_in_repo="best_model_for_stock_news_analyze_v1.joblib",
        repo_id = repo_id,
        repo_type=repo_type)
