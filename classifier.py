#Decision Tree Classifier.

#load in packages and other written functions

import pandas as pd
import numpy as np
import math
from info_gain import infoGain, sumSplit
from functions import entropy, Gini, misclassError
from chi_2 import chi_2

#reads an already made excel file by Ken Plackowski using the INVCHI(DOF, Percentage) excel function. This can be updated if we realize that we need more degrees of freedom. I've left in the alpha=0.1 and 0.05, as these are the only two we need. For chi squared 0% confidence level we always split. This is alpha = 1.

chi=pd.read_excel("Chi2_values.xlsx")
train_data = pd.read_csv("train.csv", names = ['Col1', 'Col2', 'Col3'])
column_names = []
#relabel the column with the 60 string DNA sequence as new labels -29 to 30. Assuming they are aligned at the middle of the sequence where the middle is the identifying factor, i would like the first split to register at 0. Checking this, it is true according to all three measures of uncertainty.
for x in list(range(0,len(train_data.iat[1,1]))):
    column_names.append(str(x-29))

#this turns the train.csv data into a more organized dataframe separated by position of base. Ordered from the first position (-29) to the last position (30), centered at 0. The last column is titled RESULT. Which is our target attribute.
Att_dataframe = pd.DataFrame(columns = column_names)
list_of_string_lists = []
for x in list(range(0,train_data.shape[0])):
    list_of_string_lists.append(list(train_data.iat[x,1]))

#This att_dataframe is now the required format to run our code. Each position number is a string from -30 to 29 and the final column is labeled as "Result"
Att_dataframe = pd.DataFrame(list_of_string_lists, columns = column_names)
Att_dataframe.insert(len(list_of_string_lists[1]), "Result", train_data['Col3'], True)


#testing dataframes. Tennis from mitchell, practice from class.
tennis = {'Outlook':['s', 's', 'o', 'r', 'r', 'r', 'o', 's', 's', 'r', 's', 'o', 'o', 'r'], 'Temp':['h', 'h', 'h', 'm', 'c', 'c', 'c', 'm', 'c', 'm', 'm', 'm', 'h', 'm'] ,'Hum':['h', 'h', 'h', 'h', 'n', 'n', 'n', 'h', 'n', 'n', 'n', 'h', 'n', 'h'], 'Wind':['w', 's', 'w', 'w', 'w', 's', 's', 'w', 'w', 'w', 's', 's', 'w', 's'], 'PlayTennis':['n', 'n', 'y', 'y', 'y', 'n', 'y', 'n', 'y', 'y', 'y', 'y', 'y', 'n'] }

practice = {'split1':['l', 'l', 'l', 'r', 'l', 'r', 'r', 'r'], 'split2':['l','l', 'r', 'r', 'l', 'l', 'l', 'l'], 'C': ['1','1','1','1','0','0','0','0'] }

#Use these if we want to test out practice or tennis.
#Att_dataframe=pd.DataFrame(data=practice)
#Att_dataframe=pd.DataFrame(data=tennis)


#Step 1, choose first root node by finding highest information gain.

#this gives all sets of labels and keys and calculates information gain accross all possible roots. With a return on the key of the greatest Information gain.

def maxIG(Att_dataframe, impurity_measure):
    IG_values = []
    name_values = []
    for x in Att_dataframe.columns:
        #only allows the attributes (not target) to be chosen
        if len(IG_values) < Att_dataframe.shape[1]-1:
            IG_values.append(infoGain(Att_dataframe, impurity_measure, x, Att_dataframe[x].unique()))
            name_values.append(x)
    max_IG_att = name_values[IG_values.index(max(IG_values))]
    #Finally, choose the attribute to split by and separate into subsets of split
    return max_IG_att


split_choice = maxIG(Att_dataframe,"Gini")

#can leave this in here to check on what we are keeping as the split. Can use split_choice to keep track of splits if we want to build a tree using the labels.
print("split this node by attribute", split_choice)

#tests chi_squared for the split choice. Can comment out the observed and expected dataframes in chi_2 function file. I keep it to check numbers for sanity.

print(chi_2(Att_dataframe, 0.05, split_choice, Att_dataframe.iloc[:,-1].name))
#make a set of sub dataframes set up by split chosen while removing the split column.
for x in Att_dataframe[split_choice].unique():
    split_x = Att_dataframe[(Att_dataframe[split_choice] == x)]
    new_sub_frame = split_x.drop(columns = split_choice, axis = 1)
    #print("subframe for", x, "is")
    print(new_sub_frame)
    print(str(x))

