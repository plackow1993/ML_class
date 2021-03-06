
#Decision Tree Classifier.

#load in packages and other team written functions. The team wrote infogain, functions, and chi_2. Math is imported for log, which could be taylor approximated by a polynomial to the nth degree if really necessary. Pprint for personal visualizing of the nested dict (pythons final tree structure). Time for code run time checking (this was primarily used for an idea of the effectiveness of chi_2. Random is explained in a comment below.

import pandas as pd
import numpy as np
import math
from info_gain import infoGain, sumSplit
from functions import entropy, Gini, misclassError
from chi_2 import chi_2
import pprint
import time
import random

#Begin code testing time.
start = time.time()

#when pruning, using the random seed to get the same tree each time. At pruning step, the most prevalent class will be chosen, however, if there are more than 1 most common class, a random choice of a classifier will be made. This keeps that choice consistent on any machine.
random.seed(9000)


#reads an already made excel file by Ken Plackowski using the INVCHI(DOF, Percentage) excel function. This can be updated if we realize that we need more degrees of freedom. I've left in the alpha=0.1 and 0.05, as these are the only two we need. For chi squared 0% confidence level we always split. This is alpha = 1.
chi=pd.read_excel("Chi2_values.xlsx")

#loading in the training data
train_data = pd.read_csv("train.csv", names = ['Col1', 'Col2', 'Col3'])

#testing dataframes. Tennis from mitchell, practice from class. This will test our overall tree success. Smaller, so takes less time for troubleshooting.
tennis = {'Outlook':['sunny', 'sunny', 'overcast', 'rainy', 'rainy', 'rainy', 'overcast', 'sunny', 'sunny', 'rainy', 'sunny', 'overcast', 'overcast', 'rainy'], 'Temp':['h', 'h', 'h', 'm', 'c', 'c', 'c', 'm', 'c', 'm', 'm', 'm', 'h', 'm'] ,'Hum':['h', 'h', 'h', 'h', 'n', 'n', 'n', 'h', 'n', 'n', 'n', 'h', 'n', 'h'], 'Wind':['w', 's', 'w', 'w', 'w', 's', 's', 'w', 'w', 'w', 's', 's', 'w', 's'], 'PlayTennis':['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no'] }

#The practice dataframe was chosen to asses our information gain and various information theory errors (gini, mis, entropy)
practice = {'split1':['l', 'l', 'l', 'r', 'l', 'r', 'r', 'r'], 'split2':['l','l', 'r', 'r', 'l', 'l', 'l', 'l'], 'C': ['1','1','1','1','0','0','0','0'] }


#---------This is to rewrite the training data as a workable dataframe.
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

#Final att_dataframe for use in our code (training att_dataframe)
Att_dataframe.insert(len(list_of_string_lists[1]), "Result", train_data['Col3'], True)
print(Att_dataframe.iloc[0,:])
#-----------


#Use these if we want to test out practice or tennis.
#Att_dataframe=pd.DataFrame(data=practice)
#Att_dataframe=pd.DataFrame(data=tennis)




#Step 1, choose first root node by finding highest information gain.

#-----------this gives all sets of labels and keys and calculates information gain accross all possible roots. With a return on the key of the greatest Information gain.
#impurity measure can be either: Gini, entropy, or mis (for misclassification error). Att_dataframe is the dataframe of all attributes without a previous node. (when initialized, since there is no previous node, Att_dataframe is the complete set of data. Target attribute is always at position -1.
def maxIG(Att_dataframe, impurity_measure):
    IG_values = []
    name_values = []
    for x in Att_dataframe.columns:
        #only allows the attributes (not target) to be chosen
        if len(IG_values) < Att_dataframe.shape[1]-1:
            IG_values.append(infoGain(Att_dataframe, impurity_measure, x, Att_dataframe[x].unique()))
            name_values.append(x)
    
    #print("the name of the attribute values are", name_values)
    #print("ig values are", IG_values)
    max_IG_att = name_values[IG_values.index(max(IG_values))]
    #Finally, choose the attribute to split by and separate into subsets of split
    return max_IG_att, max(IG_values)


split_choice = maxIG(Att_dataframe,"Gini")[0]


class_list = Att_dataframe[Att_dataframe.iloc[:,-1].name].unique()
class_list.sort()
print("the classes are", class_list)
#can leave this in here to check on what we are keeping as the split. Can use split_choice to keep track of splits if we want to build a tree using the labels.
#print("split this node by attribute", split_choice)

#tests chi_squared for the split choice. Can comment out the observed and expected dataframes in chi_2 function file. I keep it to check numbers for sanity.

#print(chi_2(Att_dataframe, 1, split_choice, Att_dataframe.iloc[:,-1].name))
#make a set of sub dataframes set up by split chosen while removing the split column.
for x in Att_dataframe[split_choice].unique():
    split_x = Att_dataframe[(Att_dataframe[split_choice] == x)]
    new_sub_frame = split_x.drop(columns = split_choice, axis = 1)
    
    if new_sub_frame.shape[0] == 1:
        #print(new_sub_frame)
        print("The class is", (new_sub_frame.iloc[0,-1]))
        #print(new_sub_frame)
        #print(str(x))

branch_counter = 0

#Basic idea of the tree node structure collection created with assistance from the buildTree function of https://medium.com/@lope.ai/decision-trees-from-scratch-using-id3-python-coding-it-up-6b79e3458de4
#Our algorithm requires many revisions very different from the source above, but help with the recursion aspect was researched, so I wanted to give credit to a source for this. Some sources were checked but this source was referenced on correctness. You can see artifacts of the sourse in examples like if tree is...
def Tree(Att_dataframe, impurity, branch_counter, tree = None):

    target = Att_dataframe.iloc[:,-1].name
    node = maxIG(Att_dataframe,impurity)[0]
    #Initialize Tree
    if tree is None:
        tree = {}
        tree[node] = {}
        
    #Create new dataframes for the node, basecases go here! Note that I am using chi_squared as a stopping case in the elif. I may add hyperparameters like tree length or class count values to improve accuracy as more stopping cases.
    
    
    for x in Att_dataframe[node].unique():
        print("for node value", node)
        print("at attribute value", x)
        print("branch_counter at these values is, ", branch_counter)
        sub_frame = Att_dataframe[(Att_dataframe[node] == x)]
        sub_frame = sub_frame.drop(columns=node, axis=1)
        #This is a base case to assign a value of CLASS under that node. It essetially creates a leaf if the node is pure. A node will be pure if infomation gain is 0 for ALL possibile splts (maxIG statement right (this might be handled by chi_squared) OR if the number of unique values is 1.
        if len(Att_dataframe.iloc[:,-1].unique()) == 1 or maxIG(Att_dataframe, impurity)[1] == 0:
            #print(Att_dataframe)
            #print(Att_dataframe.iloc[:,-1].unique())
            #print(maxIG(Att_dataframe, impurity)[1])
            tree = Att_dataframe.iloc[:,-1].unique()[0]
            branch_counter = 0
            break            #print("this should stop")
        elif chi_2(Att_dataframe, 1, node, Att_dataframe.iloc[:,-1].name) == 0 or branch_counter > 30:
            #These three lines were put here to troubleshoot correct counts of each class at a stopping node.
            #print("number of IE=", len(Att_dataframe[(Att_dataframe[Att_dataframe.iloc[:,-1].name] == class_list[1])]))
            #print("number of EI=", len(Att_dataframe[(Att_dataframe[Att_dataframe.iloc[:,-1].name] == class_list[0])]))
            #print("number of N=", len(Att_dataframe[(Att_dataframe[Att_dataframe.iloc[:,-1].name] == class_list[2])]))
            class_column_names = []
            class_column_counts = []
            max_options = []
            for z in class_list:
                class_column_names.append(z)
                class_column_counts.append(len(Att_dataframe[(Att_dataframe[Att_dataframe.iloc[:,-1].name] == z)]))
            print(class_column_names)
            print(class_column_counts)
            for z in range(0, len(class_list)):
                if class_column_counts[z] == max(class_column_counts):
                    max_options.append(class_column_names[z])
            tree = random.choice(max_options)
            branch_counter = 0
            break
        else:
            branch_counter += 1
            print("branch counter is", branch_counter)
            tree[node][x] = Tree(sub_frame, "Gini", branch_counter)
           
    return tree
    
t=Tree(Att_dataframe, "Gini", branch_counter)
pprint.pprint(t)

end = time.time()
#def count_keys(dict_, counter=0):
#    for each_key in dict_:
#        if isinstance(dict_[each_key], dict):
#            # Recursive call
#            counter = count_keys(dict_[each_key], counter + 1)
#        else:
#            counter += 1
#        print(each_key)
#    return counter


#print('The length of the nested dictionary is {}'.format(count_keys(t)))
print("executable time", end-start)
