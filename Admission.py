import pandas as pd
import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression , Ridge
from sklearn.tree import DecisionTreeRegressor
df=pd.read_csv(r"C:\Users\k\Desktop\Admission_Predict.csv")
df=df.drop("Serial No." , axis=1 )
x=df.drop("admit" , axis=1)
y=df["admit"]

x_train , x_test , y_train , y_test=train_test_split(x,y , test_size=0.2 , random_state=45)

lr=LinearRegression()
rid=Ridge()
dtr=DecisionTreeRegressor()
rid.fit(x_train , y_train)
lr.fit(x_train , y_train)
dtr.fit(x_train , y_train)
st.title("Chances of Admission For Graduation in College")

gre=st.number_input("GRE Score(Graduate Record Exam)")
toefl=st.number_input("TOEFL Score")
university=st.selectbox("Unversity Rating" , [0,1,2,3,4,5])
sop=st.selectbox("Statement of purpose" ,[0,1,2,3,4 ,5] )
lor=st.number_input("lor")
cpga=st.number_input("enter cgpa")
research=st.selectbox("Resarch" , ["yes" , "no" ])

if st.button("Predict"):
    if research=="yes":
       re=1
    else:
       re=0
    prediction=dtr.predict([[gre , toefl , university , sop , lor , cpga , re]])
    st.header("Chances of selecting in college is "+str(int(np.round(prediction*100))) +"%")