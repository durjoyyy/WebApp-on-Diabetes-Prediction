#This program detects for existence of Diabetes in a patient using Machine Learning
#The conecept used here is Logistic Regression 
'''durjoy12091999@gmail.com-Contact me here'''

#importing libraries
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st


#create a title and a sub-title
st.write("""
# Diabetes Detection
Detect if a person has diabeters with the help of given parametes. The accuracy of the model is given below as well
""")

#Adding an Image
image=Image.open("db.jpg")
st.image(image,caption="Diabetes",use_column_width=True)

#Getting the data
df=pd.read_csv('diabetes.csv')

#Set a sub-header
st.subheader('Data Information:')

#show the data as a table
st.dataframe(df)

#shows stats of data
st.write(df.describe())
chart=st.bar_chart(df)

#split the data into x and y independently 
X=df.iloc[:,0:8].values
Y=df.iloc[:,-1].values

#split the dataset into 75% train and 25% test
X_train, X_test, Y_train,Y_test= train_test_split(X,Y,test_size=0.25,random_state=0)

#Get the feature input from the user
def get_user_input():	
	pregnancies=st.sidebar.slider('pregnancies',0,17,3)
	glucose=st.sidebar.slider('glucose',0,199,117)
	blood_pressure=st.sidebar.slider('blood_pressure',0,120,80)
	skin_thickness=st.sidebar.slider('skin_thickness',0,99,23)
	insulin=st.sidebar.slider('insulin',0.0,846.0,30.5)
	BMI=st.sidebar.slider('BMI',0.0,67.1,32.0)
	DPF=st.sidebar.slider('DPF',0.078,2.42,0.3725)
	age=st.sidebar.slider('age',21,100,30)

	#store a dict into a variable
	user_data={
'pregnancies':pregnancies, 
'glucose':glucose,'blood_pressure':blood_pressure, 
'skin_thickness':skin_thickness,
'inslun':insulin, 
'BMI':BMI,
'DPF':DPF,
'age':age
}
	
	features=pd.DataFrame(user_data, index=[0])
	return features

#Taking the user input
user_input = get_user_input()

#Display user input
st.subheader('User Input:')
st.write(user_input)


#create and train the model
RandomForestClassifier=RandomForestClassifier()
RandomForestClassifier.fit(X_train, Y_train)


#Show the model's metrics
st.subheader('Model Test Accuracy Score:')
st.write(str(accuracy_score(Y_test, RandomForestClassifier.predict(X_test)) *100)+'%')


#Store the model's predictions
prediction=RandomForestClassifier.predict(user_input )


#Display the classification
st.subheader('Classifiction')
st.write(prediction)
	

	

