import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt


st.write("""
## Predicting whether or not somebody is having heart disease.
""")
st.write("""
1. **age**: age in years
2. **sex**: Male or Female
3. **cp**: chest pain type`0 = typical angina`; `1 = atypical angina`; `2 = non-anginal pain`; `3 = asymptomatic`
4. **trestbps**: resting blood pressure (in mm Hg on admission to the hospital)
5. **chol**: serum cholestoral in mg/dl.
6. **fbs**: (fasting blood sugar > 120 mg/dl) `1 = true`; `0 = false`
7. **restecg**: resting electrocardiographic results `0 = normal`; `1 = having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)`; `2 = showing probable or definite left ventricular hypertrophy by Estes' criteria`
8. **thalach**: maximum heart rate achieved
9. **exang**: exercise induced angina `1 = yes`; `0 = no`
10. **oldpeak**: ST depression induced by exercise relative to rest
11. **slope**: the slope of the peak exercise ST segment `0 = upsloping`; `1 = flat`; `2 = downsloping`)
12. **ca**: number of major vessels (0-3) colored by flourosopy
13. **thal**: `3 = normal`; `6 = fixed defect`; `7 = reversable defect`

""")

st.sidebar.header('Your Input Parameters')

def user_input_features():
    sex_select = st.sidebar.selectbox('Gender',('Male', 'Female'))
    sex = None
    age = st.sidebar.number_input('Age',18,75,28)
    cp = st.sidebar.slider('Chest Pain',0,3,2)
    trestbps = st.sidebar.slider('Blood Pressure',94,200,125)
    chol = st.sidebar.slider('Cholestoral',126,420,318)
    fbs = st.sidebar.slider('Blood Sugar',0,1,1)
    restecg = st.sidebar.slider('Electrocardiographic',0,2,1)
    thalach = st.sidebar.slider('Heart Rate',80,200,145)
    exang = st.sidebar.slider('Induced Angina',0,1,0)
    oldpeak = st.sidebar.slider('ST depression', 0.0,6.2,3.2)
    slope = st.sidebar.slider('Slope',0,2,1)
    ca = st.sidebar.slider('Major Vessels',0,4,2)
    thal = st.sidebar.slider('Thalassemia', 0,3,2)
    if sex_select == 'Male':
        sex = 1
    else:
        sex = 0
    data = {'age':age,
            'sex':sex,
            'cp':cp,
            'trestbps':trestbps,
            'chol':chol,
            'fbs':fbs,
            'restecg':restecg,
            'thalach':thalach,
            'exang':exang,
            'oldpeak':oldpeak,
            'slope':slope,
            'ca':ca,
            'thal':thal}
    features = pd.DataFrame(data,index=[0])
    return features

pure_data = pd.read_csv("data/heart-disease.csv")

df = user_input_features()

fig = plt.figure(figsize=(8, 4))

plt.scatter(pure_data.age[pure_data.target==1],
           pure_data.thalach[pure_data.target==1],
           c="salmon")

plt.scatter(pure_data.age[pure_data.target==0],
           pure_data.thalach[pure_data.target==0],
           c="lightblue");

plt.title("Heart Disease in function of Age and Max Heart Rate")
plt.xlabel("Age")
plt.ylabel("Max Heart Rate")
plt.legend(["Disease", "No Disease"]);
st.pyplot(fig)

# Create a plot of crosstab
fig2 = pd.crosstab(pure_data.target, pure_data.sex).plot(kind="bar",
                                   figsize=(8, 4),
                                   color=["salmon", "lightblue"]);
plt.title("Heart Disease Frequency for Sex")
plt.xlabel("0 = No Disease, 1 = Disease")
plt.ylabel("Amount")
plt.legend(["Female", "Male"])
plt.xticks(rotation = 0);
st.pyplot()
#Creating another plot of crosstab
pd.crosstab(pure_data.cp, pure_data.target).plot(kind="bar",
                                  figsize=(8, 4),
                                  color=["salmon", "lightblue"])
plt.title("Heart Disease Frequency for Chest Pain")
plt.xlabel("Chest Pain Type")
plt.ylabel("Amount")
plt.legend(["No Disease", "Disease"])
plt.xticks(rotation = 0);
st.pyplot()

st.subheader('User Input Parameters')
st.write(df)


filename_lr = 'models/log_reg.sav'
filename_rf = 'models/rs_rf.sav'
filename_knn = 'models/knn.sav'
knn_model = pickle.load(open(filename_knn, 'rb'))
rf_model = pickle.load(open(filename_rf, 'rb'))
lr_model = pickle.load(open(filename_lr, 'rb'))




picker = st.selectbox('Select classifier',
    ('Logistic Regression', 'Random Forest', 'KNNeighbors'))
if st.button('Predict'):
    if picker == 'Logistic Regression':
        prediction_proba = lr_model.predict_proba(df)
        predictions = lr_model.predict(df)
        if predictions == 1:
            st.write('You have heart disease')
            st.write('And here is your probabilities')
            st.write(prediction_proba)
        else:
            st.write('You dont have any heart disease')
            st.write('And here is your probabilities')
            st.write(prediction_proba)
    elif picker == 'Random Forest':
        predictions = rf_model.predict(df)
        prediction_proba = rf_model.predict_proba(df)
        if predictions == 1:
            st.write('You have heart disease')
            st.write('And here is your probabilities')
            st.write(prediction_proba)
        else:
            st.write('You dont have any heart disease')
            st.write('And here is your probabilities')
            st.write(prediction_proba)
    elif picker == 'KNNeighbors':
        predictions = knn_model.predict(df)
        prediction_proba = knn_model.predict_proba(df)
        if predictions == 1:
            st.write('You have heart disease')
            st.write('And here is your probabilities')
            st.write(prediction_proba)
        else:
            st.write('You dont have any heart disease')
            st.write('And here is your probabilities')
            st.write(prediction_proba)
