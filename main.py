import streamlit as st 
from st_files_connection import FilesConnection
import pandas as pd 
import numpy as np
# import scipy as sp
import pickle
import datetime as dt
import json
from xgboost import plot_importance
from xgboost.sklearn import XGBRegressor as XGBR
import boto3
import feature_engine
from io import BytesIO
# from collections import Counter

conn = st.connection('s3', type=FilesConnection)

st.title("Traffic Forecasting for the San Francisco International Airport")
st.header("This application predicts the total passengers of user-specified combinations", 
         divider='rainbow')
st.subheader("At this time, the forecasting model was updated with before December 2023 and works for a year ahead")
st.write("Specify input conditions (parameters)")

# df = pd.read_csv("/Users/owner/myvirtenv/time_series_forecasting/before_encoding.csv")
df = conn.read("airtraffic/before_encoding.csv", input_format="csv", ttl=600)
               
order = ['count_by_airline', 'count_by_price_category',
       'count_by_geo', 'count_by_region', 'count_by_activity_code',
       'count_by_operating_code', 'count_by_terminal',
       'count_by_boarding_area', 'year', 'month',
       'yrs_since_2020', 'airline_yrs', 'pricing',
       'passengers_per_operating_code', 'operating_code_ratio',
       'passengers_per_activity_code', 'activity_code_ratio',
       'passengers_per_geo', 'geo_ratio', 'passengers_per_region',
       'region_ratio', 'passengers_per_airline', 'airline_ratio',
       'passengers_per_terminal', 'terminal_ratio',
       'passengers_per_boarding_area', 'boarding_area_ratio',
       'passengers_per_pricing', 'pricing_ratio']

# define user input parameters
def user_inputs():
    yymm = st.text_input("Select a Date in 2024 (Format: YYYYMM)", "202312")
    airline = st.selectbox("Select Airline", df.airline.unique())
    pricing = st.selectbox("Select Price Category", df.price_category.unique())
    geo = st.selectbox("Select Geography", df.geo.unique())
    region = st.selectbox("Select Region", df.region.unique())
    activity_code = st.selectbox("Select Activity Code", df.activity_code.unique())
    operating_code = st.selectbox("Select Operating Code", df.operating_code.unique())
    terminal = st.selectbox("Select Terminal", df.terminal.unique())
    boarding_area = st.selectbox("Select Boarding Area", df.boarding_area.unique()),
    data = {
        "yymm": yymm,
        "airline": airline,
        "pricing": pricing,
        "geo": geo,
        "region": region,
        "activity_code": activity_code,
        "operating_code": operating_code,
        "terminal": terminal,
        "boarding_area": boarding_area
    }
    x_input = pd.DataFrame(data, index=[0])
    return x_input

s3 = boto3.resource('s3')

with BytesIO() as data:
    s3.Bucket("airtraffic").download_fileobj("submit_model.pkl", data)
    data.seek(0)    # move back to the beginning after writing
    model = pickle.load(data)

# model = conn.read("airtraffic/submit_model.pkl", input_format="pkl")
count_dict = conn.read("airtraffic/count_dict.json", input_format="json", ttl=600)
mean_encode_dict = conn.read("airtraffic/mean_encoding_dict.json", input_format="json", ttl=600)
prob_ratio_dict = conn.read("airtraffic/prob_ratio.json", input_format="json", ttl=600)

x_input = user_inputs()
st.write('You selected:')
st.dataframe(x_input)

# transform the user input to fit model feature engineering design
def count_transform(x_input, count_dict, count_cols):
    for c in count_cols:
        if c == "pricing":
            subdict = count_dict["price_category"]
            x_input[f"count_by_price_category"] = x_input[c].map(subdict)
        else:
            subdict = count_dict[c]
            x_input[f"count_by_{c}"] = x_input[c].map(subdict)
    return x_input

def prob_ratio_transform(x_input, mean_encode_dict, encode_cols):
    # Ensure year and month are strings and month is zero-padded
    x_input['yy'] = (x_input['year'].astype(int) - 1).astype(str)
    x_input['mm'] = x_input['month'].astype(str).str.zfill(2)
    
    # Function to apply row-wise
    def apply_ratio_encoding(row, col):
        # Construct the key for this row
        key = f"{row['yy']}{row['mm']}_prob"
        # Access the sub-dictionary for the column
        subdict = mean_encode_dict[col]
        # Get the ratio value using the key and the column value, with a default if key or value is missing
        return subdict.get(key, {}).get(row[col], float('nan'))
    
    # Apply mean encoding to each specified column
    for c in encode_cols:
        # Create a new column for the mean-encoded values
        x_input[f"{c}_ratio"] = x_input.apply(apply_ratio_encoding, col=c, axis=1)
    
    # Clean up temporary columns
    x_input.drop(columns=['yy', 'mm'], inplace=True)
    
    return x_input

def mean_encode_transform(x_input, mean_encode_dict, encode_cols):
    # Ensure year and month are strings and month is zero-padded
    x_input['yy'] = (x_input['year'].astype(int) - 1).astype(str)
    x_input['mm'] = x_input['month'].astype(str).str.zfill(2)
    
    # Function to apply row-wise
    def apply_mean_encoding(row, col):
        # Construct the key for this row
        key = f"{row['yy']}{row['mm']}_mean"
        # Access the sub-dictionary for the column
        subdict = mean_encode_dict[col]
        # Get the mean value using the key and the column value, with a default if key or value is missing
        return subdict.get(key, {}).get(row[col], float('nan'))
    
    # Apply mean encoding to each specified column
    for c in encode_cols:
        # Create a new column for the mean-encoded values
        x_input[f"passengers_per_{c}"] = x_input.apply(apply_mean_encoding, col=c, axis=1)
    
    # Clean up temporary columns
    x_input.drop(columns=['yy', 'mm'], inplace=True)
    
    return x_input

def transform(x_input):

    # time variables
    x_input["month"] = x_input["yymm"][4:].astype(int)
    x_input["year"] = x_input["yymm"][0:4].astype(int)
    # x_input['start_date'] = pd.to_datetime(x_input['yymm'], format='%Y%m').dt.strftime('%Y/%m/%01')

    # investigate into 2020, which was an anomaly
    x_input["yrs_since_2020"] = x_input['year'] - 2020

    # get airlines' years of service 
    minimum = df.groupby(["airline"])["year"].min().to_dict()
    maximum = df.groupby(["airline"])["year"].max().to_dict()
    operation_years = {key: maximum[key] - minimum.get(key, 0) + 1 for key in minimum.keys()}
    x_input["airline_yrs"] = x_input.airline.map(operation_years)

    # define count and encode columns
    count_cols = ["operating_code", "activity_code", "geo", "region", 
               "airline", "terminal", "boarding_area", "pricing"]
    encode_cols = ["operating_code", "activity_code", "geo", "region", 
               "airline", "terminal", "boarding_area", "pricing"]

    x_input = count_transform(x_input, count_dict, count_cols)
    x_input = mean_encode_transform(x_input, mean_encode_dict, encode_cols)
    x_input = prob_ratio_transform(x_input, prob_ratio_dict, encode_cols)

    # change pricing
    x_input["pricing"] = np.where(x_input.pricing == "Other", 1, 0)

    to_keep = [c for c in x_input.columns if c in order]
    x_input = x_input[to_keep]

    # rearrange columns 
    x_input = x_input.loc[:, order]
    return x_input

# Predict with the model 
def predict(model, x_input):
    output = np.exp(model.predict(x_input))-1
    return output

# design user interface
if st.button("Predict"):
    x = transform(x_input)
    prediction = predict(model, x)
    st.subheader("Prediction based on your inputs:")
    st.write(np.ceil(prediction))
