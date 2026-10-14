import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

#Load the trained model
model = tf.keras.models.load_model('model.h5')

with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)

with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl','rb') as file:
    onehot_encoder_geo = pickle.load(file)



## Streamlit app
st.title("Customer Churn Prediction")

## User input
geography = st.selectbox('select the geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('select the gender', label_encoder_gender.classes_)
age = st.slider('enter the age', min_value=18, max_value=120)
balance = st.number_input('enter the balance', min_value=0.0, max_value=250000.0)
credit_score = st.number_input('enter the credit score', min_value=300, max_value=850)
estimated_salary = st.number_input('enter the estimated salary', min_value=0.0, max_value=200000.0)
tenure = st.slider('enter the tenure', min_value=0, max_value=10)
num_of_products = st.slider('enter the number of products', min_value=1, max_value=4)
has_cr_card = st.selectbox('does the customer have a credit card?', [0,1])
is_active_member = st.selectbox('is the customer an active member?', [0,1])


## Prepare the input data for prediction

input_data = pd.DataFrame({
    #'Geography': [geography],
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
    
    
})

geo_encoded = onehot_encoder_geo.transform(
    pd.DataFrame({'Geography': [geography]})
)

geo_encoded_df = pd.DataFrame(geo_encoded,columns = onehot_encoder_geo.get_feature_names_out(['Geography']))
geo_encoded_df

input_data = pd.concat(
    [input_data.reset_index(drop=True), geo_encoded_df.reset_index(drop=True)],
    axis=1
)

#input_data = input_data.drop('Geography', axis=1)

input_data_scaled = scaler.transform(input_data)

prediction = model.predict(input_data_scaled)
prediction_prob = prediction[0][0]

if prediction_prob > 0.5:
    st.write(f"The customer is likely to churn with a probability of {prediction_prob:.2f}")
else:
    st.write(f"The customer is not likely to churn with a probability of {prediction_prob:.2f}")