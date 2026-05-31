
import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib
import torch
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer

model_path = hf_hub_download(repo_id="Lokeshnathy/Stock-market-news-Analyzer",filename="best_model_for_stock_news_analyze_v1.joblib")       
model = joblib.load(model_path)

st.title("Stock Market News - Sentiment Finder")
st.subheader("""
This application predicts stock market volatility and analyzes sentiment extracted from relevant news headlines. It is intended for internal use within the investment firm.
""")

# Collected app data
Date = st.date_input("Select a date",value=(datetime.today()))
Open = st.number_input("Beginning stock rate ($)",min_value=1.000000,max_value=100.000000,value=66.817497)
High = st.number_input("Maximum stock rate ($)",min_value=1.000000,max_value=100.000000,value=67.062500)
Low = st.number_input("Lowest stock rate ($)",min_value=1.000000,max_value=100.000000,value=65.862503)
Close = st.number_input("Closing stock rate ($)",min_value=1.000000,max_value=100.000000,value=64.805229)
Volume = st.number_input("Shares traded today",min_value=10000000.0,max_value=1000000000.0,value=100000000.0)
News = st.text_area("Headline",placeholder="Type/ copy & paste the news headline here...")

input_data = pd.DataFrame([{
    'Open': Open,
    'High': High,
    'Low': Low,
    'Close': Close,
    'Volume': Volume}])
transformer_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
news_embedding = transformer_model.encode(News,device=device,show_progress_bar=False)
embeddings_list = news_embedding.tolist()

# creating a new dataframe for embeddings feature names
embedding_df = pd.DataFrame([embeddings_list],columns = [f'{i}' for i in range(len(embeddings_list))])
input_data = pd.concat([input_data, embedding_df], axis=1)
input_data.columns = input_data.columns.astype(str)
classification_threshold=0.45

if st.button("Analyze"):
    prediction_proba=model.predict_proba(input_data)[-1,0,1]
    prediction=(prediction_proba>=classification_threshold).astype(int)
    result = "Negative" if prediction == -1 else "Neutral" if prediction == 0 else "Positive"
    st.suheader("Analysis is completed, find result below:")
    st.succes(f"The provided news headline projecting a **{result}** sentiment.")
