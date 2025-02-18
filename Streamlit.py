import streamlit as st
import pickle
import numpy as np

diabetes_model=pickle.load(open('diabetes_model.pkl','rb'))
scaler=pickle.load(open('scaler.pkl','rb'))
st.header('Diabetes Prediction')
#Pregnancies	Glucose	BloodPressure	SkinThickness	Insulin	BMI	DiabetesPedigreeFunction	Age	Outcome
pregrnancies=st.number_input('Number of pregrnacies')
Glucose=st.number_input('Level of Glucose')
BloodPressure=st.number_input('Blood Pressure')
SkinThickness=st.number_input('Skin Thickness')
Insulin=st.number_input('level of Insulin')
BMI=st.number_input('Body mass index')
DiabetesPedigreeFunction=st.number_input('Diabetes Ratio')
Age=st.number_input('Age')
# code for prediction
diabetes_diagnosois=''
#prediction
if st.button('Predict'):
    input_data=np.array[[pregrnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]]
    input_data_scaled=scaler.transform(input_data)
    diab_pred=diabetes_model.predict(input_data_scaled)
    if diab_pred[0]==1:
        diabetes_diagnosois='You have diabetes'
        st.error(diabetes_diagnosois)
    else:
        diabetes_diagnosois='You do not have diabetes'
        st.success(diabetes_diagnosois)
    

