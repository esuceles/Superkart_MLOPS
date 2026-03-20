import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib
from datetime import datetime

# Download and load the model
model_path = hf_hub_download(repo_id="celesqa/Superkart_MLOPS", filename="superkart_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Machine Failure Prediction
st.title("Superkart SalesForecast App")
st.write("""
This application predicts the revenue.
Please enter the data below to get the sales forecast.
""")

# User input
ProductID=st.text_input("Product ID", value=1, max_chars=6, placeholder='Enter the products unique ID')
ProductWeight=st.number_input("Product Weight", min_value=1, max_value=30)
ProductSugarContent=st.selectbox("Product Sugar Content", ["Low Sugar", "No Sugar", "Regular"])
ProductAllocatedArea=st.number_input("Product Allocated Area", value=0.0, min_value=0.0, max_value=0.5 format="%.3f")
ProductType=st.selectbox("Product Type",["Baking Goods","Breads","Breakfast","Canned","Dairy","Frozen Foods","Fruits and Vegetables","Hard Drinks","Health and Hygiene","Household","Meat","Others","Seafood","Snack Foods","Soft Drinks","Starchy Foods"])
ProductMRP=st.number_input("Product MRP",  value=0.0, min_value=1.0, max_value=300,format="%.2f")
StoreId=st.text_input("Store ID", value=1, max_chars=6, placeholder='Enter the stores unique ID')
current_year = datetime.now().year
StoreEstablishmentYear=st.number_input("Store Establishment Year", range(current_year, 1979, -1))
StoreSize=st.selectbox("Store Size", ["High", "Medium", "Low"])
StoreLocationCityType=st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])
StoreType=st.selectbox("Store Type",["Departmental Store","Food Mart","Supermarket Type1","Supermarket Type2"])

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    Product_Id='ProductID',
    Product_Weight='ProductWeight',
    Product_Sugar_Content='ProductSugarContent'
    Product_Allocated_Area='ProductAllocatedArea'
    Product_Type='ProductType'
    Product_MRP='ProductMRP'
    Store_ID='StoreId'
    Store_Establishment_Year='StoreEstablishmentYear'
    Store_Size='StoreSize'
    Store_Location_City_Type='StoreLocationCityType'
    Store_Type='StoreType'
}])


if st.button("Forecast Sales"):
    prediction = model.predict(input_data)
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{prediction[0]:.2f}**")
