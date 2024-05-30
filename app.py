import streamlit as st
import joblib
import pandas as pd

# loading the trained model
model = joblib.load('model.joblib')

# Create new data
new_data = st.number_input("GrLivArea", min_value=0.0, max_value=10000.0, value=200.0)

# get prediction
X_columns = ['GrLivArea']
X = pd.DataFrame([[new_data]], columns=X_columns)

# make predictions
result = model.predict(X)

# show the result in your Web App
st.text(f"This is your prediction {result}")