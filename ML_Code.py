import pandas as pd
import numpy as np

df=pd.read_excel("GameDataCleaned.xlsx")    #read file.
X1=df[["Year_of_Release", "Publisher", "Critic_Score", "Critic_Count", "User_Score", "User_Count", "Published_in_NA",
       "Published_in_EU", "Published_in_JP", "Published_in_Other"]] #Sets up the dataframe 

#Checking if the dataframe contains empty cell values.
df.isnull().values.any() #Cleans out any null values in the dataframe 
#Output label
y=df['Global_Sales']
#print(X1)

#Normalizing the features using MinMaxScaler
#https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html 
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X = scaler.fit_transform(X1)
#print(X)

#Importing the linear regression model, most of this code came from lab 1
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

#sample code
lin_reg = LinearRegression()
np.random.seed(40)  # to make this code example reproducible

kfold = KFold(n_splits=10, shuffle=True, random_state=42) #--->here k=3 with 3-fold cross validation.
#random_state controls randomness of each fold.
scores = []
for i, (train, test) in enumerate(kfold.split(X, y)):#--->splitting into train and test set.
#Enter your code here. You can refer lab 1 and assignment 1 document for the implementation of
#fit and score.
    lin_reg.fit(X[train], y[train])
    score = lin_reg.score(X[test], y[test])
    scores.append(score)

count = 0
avg = 0
for i in scores:
    avg += i
    count += 1

#print(scores)
#print(count)
avg = avg/count
print(avg)