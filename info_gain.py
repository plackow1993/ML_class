#information gain using functions.py

import pandas as pd
import numpy as np
import math
from functions import *
#same examples as used in funtions.py
tennis = {'Outlook':['s', 's', 'o', 'r', 'r', 'r', 'o', 's', 's', 'r', 's', 'o', 'o', 'r'], 'Temp':['h', 'h', 'h', 'm', 'c', 'c', 'c', 'm', 'c', 'm', 'm', 'm', 'h', 'm'] ,'Hum':['h', 'h', 'h', 'h', 'n', 'n', 'n', 'h', 'n', 'n', 'n', 'h', 'n', 'h'], 'Wind':['w', 's', 'w', 'w', 'w', 's', 's', 'w', 'w', 'w', 's', 's', 'w', 's'], 'PlayTennis':['n', 'm', 'y', 'y', 'y', 'n', 'y', 'n', 'y', 'y', 'y', 'y', 'y', 'n'] }

practice = {'split1':['l', 'l', 'l', 'r', 'l', 'r', 'r', 'r'], 'split2':['l','l', 'r', 'r', 'l', 'l', 'l', 'l'], 'C': [1,1,1,1,0,0,0,0] }

practice=pd.DataFrame(data=practice)
tennis=pd.DataFrame(data=tennis)


#information gain
#impurity given is to decide if info gain is done using entropy, missclass error
#or gini index
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
        split = Att_dataframe[(Att_dataframe[feature] == x)]
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

    for x in list(range(0,len(Class_counts))):
        Sum = Sum + Class_counts[x]
        
    return Sum



print(infoGain(practice,"entropy","split1",['l','r']))
print(infoGain(practice,"gini","split1",['l','r']))
