#Creating functions for each of the indices we would like to use: Gini index, Misclassificaiton error, entropy. For use in information gain.
#iloc idea found from https://thispointer.com/pandas-select-last-column-of-dataframe-in-python/


import pandas as pd
import numpy as np
import math



#creating a simple table of data for testing the functions (Mitchel example).

tennis = {'Outlook':['s', 's', 'o', 'r', 'r', 'r', 'o', 's', 's', 'r', 's', 'o', 'o', 'r'], 'Temp':['h', 'h', 'h', 'm', 'c', 'c', 'c', 'm', 'c', 'm', 'm', 'm', 'h', 'm'] ,'Hum':['h', 'h', 'h', 'h', 'n', 'n', 'n', 'h', 'n', 'n', 'n', 'h', 'n', 'h'], 'Wind':['w', 's', 'w', 'w', 'w', 's', 's', 'w', 'w', 'w', 's', 's', 'w', 's'], 'PlayTennis':['n', 'm', 'y', 'y', 'y', 'n', 'y', 'n', 'y', 'y', 'y', 'y', 'y', 'n'] }

#This is our example splits from class (2.1 lecture), used to check if the error functions are being coded correctly.

practice = {'split1':['l', 'l', 'l', 'r', 'l', 'r', 'r', 'r'], 'split2':['l','l', 'r', 'r', 'l', 'l', 'l', 'l'], 'C': [1,1,1,1,0,0,0,0] }

practice=pd.DataFrame(data=practice)
tennis=pd.DataFrame(data=tennis)

practice = practice[(practice["split1"] == 'l')]
tennis = practice[(practice["split2"] == 'l')]

#finding the correct attribute table

#Useful inside of the function because this calls for the class values for the attribute of interest. When filtered, these are exactly the conditional probabilities of the filter e.g P(conditions|filtered attribute feature): Class_Counts  = tennis.iloc[:,-1].value_counts()

#att_dataframe is the attribute we are interesting in finding the misclass error for
#I have it is a DF with (attributes,target). Here, we can restrict to reduction of the matrix by feature of interest, and the last column will contain those values needed for any error function.

#Misclassification error
def misclassError(Att_dataframe):
	probs = []
	Sum = 0
	Class_counts = Att_dataframe.iloc[:,-1].value_counts()
	
	for x in list(range(0,len(Class_counts))):
		Sum = Sum + Class_counts[x]
	
	for x in list(range(0,len(Class_counts))):
		probs.append(Class_counts[x]/Sum)

	error = 1-max(probs)
	# only if you want to see this
	#print(error)
	

#Entropy
def entropy(Att_dataframe):
	probs = []
	Sum = 0
	Class_counts = Att_dataframe.iloc[:,-1].value_counts()	

	for x in list(range(0,len(Class_counts))):
                Sum = Sum + Class_counts[x]

	for x in list(range(0,len(Class_counts))):
		probs.append(Class_counts[x]/Sum*math.log(Class_counts[x]/Sum,2)) 
	

	error = -sum(probs)
	#print(error)
	

#Gini index
def Gini(Att_dataframe):
        probs = []
        Sum = 0
        Class_counts = Att_dataframe.iloc[:,-1].value_counts()

        for x in list(range(0,len(Class_counts))):
                Sum = Sum + Class_counts[x]

        for x in list(range(0,len(Class_counts))):
                probs.append((Class_counts[x]/Sum)**2)


        error = 1-sum(probs)
        print(error)


Gini(practice)
Gini(tennis)
