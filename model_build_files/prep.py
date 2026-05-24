
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

api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/Lokeshnathy/Stock-Market-News-Data/stock_news.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

transformer_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedding_matrix = transformer_model.encode(df['News'],device=device,show_progress_bar=False)

X = embedding_matrix
y = df['Label']

Xtrain,Xtest,ytrain,ytest = train_test_split(
    X,y,
    test_size=0.15,
    random_state = 42)


np.save('Xtrain.npy',Xtrain)
np.save('Xtest.npy',Xtest)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)

dataset_related = ["Xtrain.npy","Xtest.npy","ytrain.csv","ytest.csv"]

for file_path in dataset_related:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo = file_path.split("/")[-1],
        repo_id="Lokeshnathy/Stock-Market-News-Data",
        repo_type="dataset")
