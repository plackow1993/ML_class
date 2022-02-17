
#starting code for chi squared information. Should be implemented as a pruning measure within the main code.

import pandas as pd
import numpy as np
import math



#creating a simple table of data for testing the functions (Mitchel example).

tennis = {'Outlook':['s', 's', 'o', 'r', 'r', 'r', 'o', 's', 's', 'r', 's', 'o', 'o', 'r'], 'Temp':['h', 'h', 'h', 'm', 'c', 'c', 'c', 'm', 'c', 'm', 'm', 'm', 'h', 'm'] ,'Hum':['h', 'h', 'h', 'h', 'n', 'n', 'n', 'h', 'n', 'n', 'n', 'h', 'n', 'h'], 'Wind':['w', 's', 'w', 'w', 'w', 's', 's', 'w', 'w', 'w', 's', 's', 'w', 's'], 'PlayTennis':['n', 'n', 'y', 'y', 'y', 'n', 'y', 'n', 'y', 'y', 'y', 'y', 'y', 'n'] }

#This is our example splits from class (2.1 lecture), used to check if the error functions are being coded correctly. Not needed for the chi-squared example.

practice = {'split1':['l', 'l', 'l', 'r', 'l', 'r', 'r', 'r'], 'split2':['l','l', 'r', 'r', 'l', 'l', 'l', 'l'], 'C': [1,1,1,1,0,0,0,0] }
Att_dataframe=pd.DataFrame(data=tennis)
#These are our inputs to this chi-squared function. Alpha is the critical value. Alpha*100 = 1 - %confidence
def chi_2(Att_dataframe, alpha, attribute_name, class_name):

    if alpha == 0:
        print('Chi is greater than the critical value so this test is a pass, keep the split (p<ALPHA)')
        result = "keep"
        
    else:
        #reads an already made excel file by Ken Plackowski using the INVCHI(DOF, Percentage) function. This can be updated if we realize that we need more degrees of freedom. I've left in the alpha=0.1 and 0.05, as these are the only two we need.

        chi=pd.read_excel("Chi2_values.xlsx")
        #make alphabetically sorted lists of the lables and the classes after a split this is to make sure the chi-squared counts align
        label_list = Att_dataframe[attribute_name].unique()
        class_list = Att_dataframe[class_name].unique()
        label_list.sort()
        class_list.sort()
        #Dataframe is the root node data frame. We will want to separate the node into n branches (where n = number of labels an attribute can take)



        #degrees of freedom Just count the number of class elements and labels for the specific attribute you are checking.

        label_count = Att_dataframe[attribute_name].nunique()
        class_count = Att_dataframe[class_name].nunique()
        deg_of_free=(label_count-1)*(class_count-1)

        print('DoF =', deg_of_free)

        if deg_of_free == 0:
            print("split is pure, keep the split")
        #this is the observed counts dataframe.

        else:
            obs_frame = pd.DataFrame()
            for x in label_list:
                att_split = Att_dataframe[(Att_dataframe[attribute_name]==x)]
                for y in class_list:
                    obs_frame.loc[x,y]=len(Att_dataframe[(Att_dataframe[attribute_name]==x) & (Att_dataframe[class_name]==y)])
            
            obs_frame.loc['Total'] = obs_frame.sum()
            obs_frame['Total'] = obs_frame.sum(axis=1)
            print("observed data is", obs_frame)
            
            #this is the expected counts dataframe
            exp_frame = pd.DataFrame()
            for x in label_list:
                for y in class_list:
                    exp_frame.loc[x,y] = (obs_frame.loc['Total',y]*obs_frame.loc[x,'Total'])/obs_frame.loc['Total','Total']
            print("expected data is", exp_frame)
            
            #this is the chi squared statistic
            chi_2 = 0
            for x in label_list:
                for y in class_list:
                    chi_2 = chi_2 + (obs_frame.loc[x,y]-exp_frame.loc[x,y])**2/exp_frame.loc[x,y]
            print('Chi squared =', chi_2)
            
            #this checks the test and give a result "pass" or "fail". Pass means that p<Alpha. p< alpha means X_calculated>X_alpha
            #deg_of_free-1 is there because i didnt feel like changing the index of the chi import.
            chi_crit = chi.loc[deg_of_free-1, alpha]
            print('critical value is', chi_crit)
            
            if chi_2 > chi_crit:
                print('Chi is greater than the critical value so this test is a pass, keep the split (p<ALPHA)')
                result = "keep"
            else:
                print('Chi is less than the critical value so this test is a fail, stop splitting here (p>ALPHA)')
                result = "stop"

    return result
    
print(chi_2(Att_dataframe, 0.05, "Outlook", "PlayTennis"))
print(chi_2(Att_dataframe, 0.00, "Outlook", "PlayTennis"))
