# To import necessary libraries
import os
import torch
import sklearn
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from huggingface_hub import login,HfApi
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
# Loading the dataset
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/Lokeshnathy/Stock-Market-News-Data/stock_news.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")
# converting news column into embedding matrix
transformer_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedding_matrix = transformer_model.encode(df['News'],device=device,show_progress_bar=False)
embedding_df = pd.DataFrame(embedding_matrix)
df.join(embedding_df,how='right')
# To split data into train test splits
X = df.drop(['News','Date','Label'],axis=1)
y = df['Label']
X.columns = X.columns.astype(str)
Xtrain,Xtest,ytrain,ytest = train_test_split(
    X,y,
    test_size=0.15,
    random_state = 42)
# Converting the splits into split datasets
Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)
dataset_related = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]
# Uploading split datasets into HF repository
for file_path in dataset_related:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo = file_path.split("/")[-1],
        repo_id="Lokeshnathy/Stock-Market-News-Data",
        repo_type="dataset")
