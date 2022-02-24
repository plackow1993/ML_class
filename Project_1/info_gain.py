#information gain using functions.py

import pandas as pd
import numpy as np
import math
from functions import *
#same examples as used in funtions.py
tennis = {'Outlook':['s', 's', 'o', 'r', 'r', 'r', 'o', 's', 's', 'r', 's', 'o', 'o', 'r'], 'Temp':['h', 'h', 'h', 'm', 'c', 'c', 'c', 'm', 'c', 'm', 'm', 'm', 'h', 'm'] ,'Hum':['h', 'h', 'h', 'h', 'n', 'n', 'n', 'h', 'n', 'n', 'n', 'h', 'n', 'h'], 'Wind':['w', 's', 'w', 'w', 'w', 's', 's', 'w', 'w', 'w', 's', 's', 'w', 's'], 'PlayTennis':['n', 'n', 'y', 'y', 'y', 'n', 'y', 'n', 'y', 'y', 'y', 'y', 'y', 'n'] }

practice = {'split1':['l', 'l', 'l', 'r', 'l', 'r', 'r', 'r'], 'splitb':['l','l', 'r', 'r', 'l', 'l', 'l', 'l'], 'C': ['1','1','1','1','0','0','0','0'] }

#two dataframes to check on function's function
practice=pd.DataFrame(data=practice)
tennis=pd.DataFrame(data=tennis)


#-----information gain
#impurity given is to decide if info gain is done using entropy, missclass error
#or gini index

#att_dataframe is a pandas dataframe of our data with target attribute at -1
#impurity is a string of entropy, mis (for misclass error), or Gini.
#attributes are a list of values (all str) taken by the attribute (A, T, G, C, etc ...)
#feature is the attribute name as a (str). For our genomic data, the values of 0, 1, 2, etc are all strings, not ints, by design.

def infoGain(Att_dataframe,impurity,feature,attributes):
    #use functions to find impurity
    targetImpurity = 0
    if(impurity == "entropy"):
        targetImpurity = entropy(Att_dataframe)
    elif(impurity == "mis"):
        targetImpurity = misclassError(Att_dataframe)
    else:
        targetImpurity = Gini(Att_dataframe)

    imps = []
    totalSum = 0
    for x in attributes:
        imp = 0
        #print("x is equal to", x)
        split = Att_dataframe[(Att_dataframe[feature] == x)]
        #print("split becomes", split)
        Sum = sumSplit(split)
        totalSum += Sum
        if(impurity == "entropy"):
            imp = entropy(split)
        elif(impurity == "mis"):
            imp = misclassError(split)
        else:
            imp = Gini(split)
        imps.append([imp,Sum])
    remainingImpurity = 0
    for x in imps:
        remainingImpurity += x[0] * (x[1]/totalSum)
    return targetImpurity - remainingImpurity


def sumSplit(Att_dataframe):
    probs = []
    Sum = 0
    Class_counts = Att_dataframe.iloc[:,-1].value_counts()
    #print("Class_counts has length", len(Class_counts))
    #print("class_counts looks like", Class_counts, "and has data type", type(Class_counts))
    
    if len(Class_counts) == 1:
        Sum = Class_counts[0]
    else:
        for x in list(range(0,len(Class_counts))):
            Sum = Sum + Class_counts[x]
        
    return Sum


#checking as a standalone function
#print(sumSplit(practice))
#print(infoGain(tennis, "entropy", "Outlook", ['s','o', 'r']))
#print(infoGain(practice, "entropy","splitb" ,['l','r']))
