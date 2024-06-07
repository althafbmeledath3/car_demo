import streamlit as st
import pickle
import numpy as np
import sklearn
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd

st.title("Car Price Predictor")
df = pickle.load(open("df.pkl","rb"))
model = pickle.load(open("model.pkl","rb"))

year = st.selectbox("year",df['year'].unique())
km = st.number_input("Ënter the km driven")
mileage = st.number_input("Ënter the Mileage")
engine = st.number_input("Ënter the engine power")
max_power = st.number_input("Enter the max Power")
seats = st.selectbox("seats",df['seats'].unique())


val = np.array([year,km,mileage,engine,max_power,seats]).reshape(1,6)

if st.button("Predict Price"):
    out = model.predict(val)
    st.write(out)



