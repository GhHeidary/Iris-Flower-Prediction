from tracemalloc import Snapshot
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
st.write("""
# Simple Iris Flower Prediction App
This app predicts the **Iris flower** type!
""")

st.sidebar.header('User Input Parameters')
st.sidebar.header("user input parameter")
def user_input_feature():
    sepal_lenght=st.sidebar.slider("sepal_lenght",4.3,7.9,5.4)
    sepal_width=st.sidebar.slider("sepal_width",2.0,4.4,3.4)
    petal_lenght=st.sidebar.slider("petal_lenght",1.0,6.9,1.3)
    petal_width=st.sidebar.slider("petal_width",0.1,2.5,0.2)
    data={"sepal_lenght":sepal_lenght,
          "sepal_width":sepal_width,
          "petal_lenght":petal_lenght,
          "petal_width":petal_width}
    features=pd.DataFrame(data,index=[0])
    return features

df=user_input_feature()
st.subheader('User input parameters')
st.write(df)

iris=datasets.load_iris()
X=iris.data
Y=iris.target

clf=RandomForestClassifier()
clf.fit(X,Y)

prediction=clf.predict(df)
prediction_proba=clf.predict_proba(df)
st.subheader('Class labls and their corresponding index number')
st.write(iris.target_names)

st.subheader('prediction')
st.write(iris.target_names[prediction])

st.subheader('prediction probability')
st.write(prediction_proba)
st.balloons()

import time
my_bar = st.progress(0)
for percent_complete in range(100):
    time.sleep(0.1)
    my_bar.progress(percent_complete + 1)
