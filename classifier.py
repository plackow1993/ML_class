#Decision Tree Classifier.

#load in test dataframe

import pandas as pd
import numpy as np
import math
from info_gain import infoGain, sumSplit
from functions import entropy, Gini, misclassError

tennis = {'Outlook':['s', 's', 'o', 'r', 'r', 'r', 'o', 's', 's', 'r', 's', 'o', 'o', 'r'], 'Temp':['h', 'h', 'h', 'm', 'c', 'c', 'c', 'm', 'c', 'm', 'm', 'm', 'h', 'm'] ,'Hum':['h', 'h', 'h', 'h', 'n', 'n', 'n', 'h', 'n', 'n', 'n', 'h', 'n', 'h'], 'Wind':['w', 's', 'w', 'w', 'w', 's', 's', 'w', 'w', 'w', 's', 's', 'w', 's'], 'PlayTennis':['n', 'n', 'y', 'y', 'y', 'n', 'y', 'n', 'y', 'y', 'y', 'y', 'y', 'n'] }

practice = {'split1':['l', 'l', 'l', 'r', 'l', 'r', 'r', 'r'], 'split2':['l','l', 'r', 'r', 'l', 'l', 'l', 'l'], 'C': ['1','1','1','1','0','0','0','0'] }

Att_dataframe=pd.DataFrame(data=practice)
#Att_dataframe=pd.DataFrame(data=tennis)


#Step 1, choose first root node by finding highest information gain.

#print(tennis.iloc[:,-1].unique())
#this gives all sets of labels and keys and calculates information gain accross all possible roots. With a return on the key of the greatest Information gain.

def maxIG(Att_dataframe, impurity_measure):
    IG_values = []
    name_values = []
    for x in Att_dataframe.columns:
        #print(x)
        #print(Att_dataframe[x].unique())
        #only allows the attributes (not target) to be chosen
        if len(IG_values) < Att_dataframe.shape[1]-1:
            IG_values.append(infoGain(Att_dataframe, impurity_measure, x, Att_dataframe[x].unique()))
            name_values.append(x)

    max_IG_att = name_values[IG_values.index(max(IG_values))]
    #Finally, choose the attribute to split by and separate into subsets of split
    return max_IG_att

    
split_choice = maxIG(Att_dataframe,"entropy")
print("split this node by", split_choice)
#make a set of sub dataframes set up by split chosen
for x in Att_dataframe[split_choice].unique():
    split_x = Att_dataframe[(Att_dataframe[split_choice] == x)]
    new_sub_frame = split_x.drop(split_choice, 1)
    #print("split_"+str(x))
    print("subframe for", x, "is")
    print(new_sub_frame)
    #print(sumSplit(new_sub_frame))


#print(infoGain(practice, "entropy", "split2", ['l','r']))

