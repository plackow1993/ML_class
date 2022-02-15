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

practice = practice[(practice["split1"] == 'l')]
tennis = practice[(practice["split2"] == 'l')]


#information gain
#impurity given is to decide if info gain is done using entropy, missclass error
#or gini index
def infoGain(Att_dataframe,impurity):
    imp = 0
    if(impurity == "entropy"): imp = entropy(Att_dataframe)
    elif(impurity == "miss"): imp = misclassError(Att_dataframe)
    else: imp = Gini(Att_dataframe)
    counter = len(Att_dataframe)*-1 + 1
    totalSum = 0
    gainList = []
    while(counter > -1):
        Class_counts = Att_dataframe.iloc[:,counter].value_counts()
        gain = 0
        Sum = 0
        for x in list(range(0,len(Class_counts))):
            Sum = Sum + Class_counts[x]
        totalSum += Sum
        for x in list(range(0,len(Class_counts))):
            probs.append(Class_counts[x]/Sum * math.log(Class_counts[x]/Sum))
        counter += 1
        
        gainList.append([sum(gain)*-1,Sum])
    
    totalImp = 0
    for x in gainList:
        totalImp += x[0] * x[1]/totalSum
        
        
    return imp - totalImp
    
        
print("IG from practice, with entropy:",infoGain(practice,"entropy"))
print("IG from practice with gini:",infoGain(practice,"gini"))
print("IG from practice with misclass:",infoGain(practice,"miss"))
print("\nIG from tennis with entropy:",infoGain(tennis,"entropy"))
print("IG from tennis with gini:",infoGain(tennis,"gini"))
print("IG from tennis with misclass:",infoGain(tennis,"miss"))
   
    
