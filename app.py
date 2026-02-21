import streamlit as st
import pandas as pd
import numpy as np
import joblib


## Load the saved machine learning objects
#kmeans = joblib.load("kmeans_model.pkl")
#scaler = joblib.load("scaler.pkl")
#pca = joblib.load("pca.pkl")

pipeline = joblib.load("customer_segmentation.pkl")

# 2. Set up the App Interface
st.title("Customer Segmentation App")
st.write("Enter customer details to predict the segment.")

# 3. Create input fields for user data
# Use st.number_input for numerical data
age = st.number_input("Age", min_value=18, max_value=100, value=50)
income = st.number_input("Income", min_value=0,max_value=200000, value=10000)
total_spending = st.number_input("Total Spending", min_value=0,max_value=5000, value=2500)
num_web_purchases = st.number_input("Number of Web Purchases", min_value=0,max_value=100, value=50)
num_store_purchases = st.number_input("Number of Store Purchases", min_value=0,max_value=100, value=50)
num_web_visits = st.number_input("Number of Web Visits", min_value=0,max_value=50, value=25)
recency = st.number_input("Recency (days since last purchase)", min_value=0,max_value=365, value=150)


# 4. Create DataFrame for processing
input_data = pd.DataFrame({
    "Age": [age],
    "Income": [income],
    "Total_Spending": [total_spending],
    "NumWebPurchases": [num_web_purchases],
    "NumStorePurchases": [num_store_purchases],
    "NumWebVisitsMonth": [num_web_visits],
    "Recency": [recency]
})


if st.button("Predict Segment"):
    st.balloons()
    
    
    ## Scaling input before prediction
    #input_scaled = scaler.transform(input_data)
    ## pca.transform not pca.fit_transform because 
    #input_pca = pca.transform(input_data)
    #cluster = kmeans.predict(input_pca )[0]
    
    cluster = pipeline.predict(input_data)[0]
    st.success(f"Predicted Segment: Cluster {cluster}")