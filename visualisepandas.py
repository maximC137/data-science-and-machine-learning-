#In this file  we are going to visuallize the desicion tree that our model did it
import pandas as pd 
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

music_data =pd.read_csv('music.csv')
x = music_data.drop(columns =['genre'])
y =music_data['genre']

model =DecisionTreeClassifier()
model.fit(x ,y)

tree.export_graphviz(model ,out_file ='music_recommender.dot'
			,feature_names=['age' ,'gender'] 
			,class_names =sorted(y.unique())
			,label ='all'
			,rounded =True
			,filled =True)
#we run this code
#The dot formate for the file (music_recommender.dot) is text language to describe graphs
#We open the file in visual studio code and we install dot from the extensions
#finally we click : Open Preview to the side and we will get the graph