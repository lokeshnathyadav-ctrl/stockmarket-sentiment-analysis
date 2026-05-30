
import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

model_path = hf_hub_download(repo_id="Lokeshnathy/Stock-market-news-Analyzer",filename="stk_mrkt_ns_anlzr_v1.joblib")
model = joblib.load(model_path)

st.title("Stock Market News - Sentiment Finder")
st.write("""
This application predicts stock market volatility and analyzes sentiment extracted from relevant news headlines. It is intended for internal use within the investment firm.
""")

# News Headline
Date = st.date_input("Select a date",value=("today",datetime.date),max_value="today")
Open = st.number_input("Beginning stock rate ($)",min_value=1.000000,max_value=100.000000,value=66.817497)
High = st.number_input("Maximum stock rate ($)",min_value=1.000000,max_value=100.000000,value=67.062500)
Low = st.number_input("Lowest stock rate ($)",min_value=1.000000,max_value=100.000000,value=65.862503)
Close = st.number_input("Closing stock rate ($)",min_value=1.000000,max_value=100.000000,value=64.805229)
Volume = st.number_input("Shares traded today",min_value=10000000.0,max_value=1000000000.0,value=100000000.0)
News = st.text_area("Headline",placeholder="Type/ copy & paste the news headline here...")

input_data = pd.DataFrame([{
    'News':News,
    'Date': Date,
    'Open': Open,
    'High': High,
    'Low': Low,
    'Close': Close,
    'Volumne': Volumne}])
classification_threshold=0.45

if st.button("Analyze"):
    prediction_proba=model.predict_proba(input_data)[-1,0,1]
    prediction=(prediction_proba>=classification_threshold).astype(int)
    result = "Negative" if prediction == -1 else result = "Neutral" if prediction == 0 else result = "Neutral"
    st.suheader("Analysis is completed, find result below:")
    st.succes(f"The provided news headline projecting a **{result}** sentiment.")
