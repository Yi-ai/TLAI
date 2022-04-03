import streamlit as st
import numpy as np
import pandas as pd
import io

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

st.header("Alzheimer's Disease Prediction")
st.subheader("Predicts the diagnosis of Alzheimer's disease based on the patient's data.")
st.write("This application uses KNN")

data=pd.read_csv("alzheimer.csv")


data.dropna(inplace=True)

# replacing values
data['Group'].replace(['Nondemented', 'Demented'],[0, 1], inplace=True)
data['M/F'].replace(['M', 'F'],[0, 1], inplace=True)
data=data.drop(data[data["Group"]=="Converted"].index)

x_input = st.slider("Choose X input", min_value=0.0, max_value=1.0,key='x')
y_input = st.slider("Choose Y input", min_value=0.0, max_value=1.0,key='y')
k = st.slider("Choose value of K", min_value=1, max_value=10,key='k')

input = (x_input,y_input)

if st.button('Data info'):
    st.header("Data info")
    buffer = io.StringIO()
    data.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

x = data.drop(['Group'], axis = 1)
y = data["Group"]

## split train / test
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.20, random_state=101)

scaler = StandardScaler()
scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=7)
classifier.fit(x_train, y_train)



corr = data.corr()

if st.button('Show fearures correlation matrix'):
    c1=plt.figure(figsize=(14,8))
    sns.heatmap(corr,cmap="Blues",annot=True,xticklabels=corr.columns,yticklabels=corr.columns)
    st.header("Heatmap")
    st.pyplot(c1)


if st.button('Run model'):
    # model=RandomForestClassifier()
    # model.fit(x_train,y_train)
    y_pred = classifier.predict(x_test)


    score=accuracy_score(y_test, y_pred)
    mat=confusion_matrix(y_test, y_pred)
    

    st.header("Accuracy score")
    st.text(score)
    st.header("Confusion matrix")
    st.write(mat)
    st.write(classification_report(y_test, y_pred))
