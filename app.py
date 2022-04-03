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

# age_input = st.slider("Choose Age input", min_value=60, max_value=100)
# educ_input = st.slider("Choose years of education input", min_value=6, max_value=23)
# gender_input = st.selectbox('What is your gender? (0 = M, 1 = F)', (0,1))

k = st.slider("Choose value of K", min_value=1, max_value=10,key='k')

# input = (age_input, educ_input, gender_input)

if st.button('Data info'):
    st.header("Data info")
    buffer = io.StringIO()
    data.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

x = data.drop(['Group'], axis = 1)
y = data["Group"]
y = y.astype('int')

## split train / test
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.20, random_state=101)

scaler = StandardScaler()
scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=k)
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
    st.text(classification_report(y_test, y_pred))

