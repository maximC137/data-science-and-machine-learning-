#Installing the pandas library to work with csv files
import pandas as pd
#Install the machine learning library we are going to use to create a model
from sklearn.tree import DecisionTreeClassifier
#This part is going to split the dataset to a training set and a testing set
from sklearn.model_selections import train_test_split
#To test the accuracy we need this part of the sklearn library with a function
from sklearn.metrics import accuracy_score

# Persisting Models :saving the predictions in a file for a fast responding
from sklearn.externals import joblib

music_data = pd.read_csv('music.csv')

#Now we are going to split the data into two parts

#We used the .drop function to exclude the genre column and we get the first part :The input set
x = music_data.drop(columns =['genre'])
#Type x below this line and run the programm to see the difference we have made

#y here represnts the second part of our data set :The output set
y =music_data['genre']
#Type y below this line and run the programm to see the difference we have made

#Calling the functions required for calculating the accuracy of our machine learning model

#We give the functions the x ,y data sets plus the size of our data that we want to test
x_train ,x_test ,y_train ,y_test =train_test_split(x ,y ,test_size =0.2)

#Defining the model and train it to predict by taking both of input and output set 
model =DecisionTreeClassifier()
#Asking the model to predict
#First we pass the two sets
model.fit(x ,y)
#We pass two arrays to train this model
# [21 ,1] means : A 21 years old ,0 :Man .[22 ,0] means : A 22 years old woman 
#predictions = model.predict([ [21 ,1] ,[22 ,0] ])

#Calling the joblib 
joblib.dump(model ,'music_recommender.joblib')


#To test the accuracy and actually train our model
predictions =model.predict(x_test)
#Then we compare the result with our output set (y_test)

#Calling the accuracy function and it gives a score between 0 and 1
score = accuracy_score =(y_test ,predictions)
#Please note that every time we give a different test_size it gives us a different score 
#When the data set is big enough the accuracy score it is in his best result