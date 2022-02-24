#This will be for use of utilizing the tree from "ken's classifier", This takes in a vector with keys -29,...,0,...,30, and provides a result from the following options ['EI', IE', 'N']
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
start_time = time.time()

#when pruning, using the random seed to get the same tree each time. At pruning step, the most prevalent class will be chosen, however, if there are more than 1 most common class, a random choice of a classifier will be made. This keeps that choice consistent on any machine.
random.seed(9000)


#reads an already made excel file by Ken Plackowski using the INVCHI(DOF, Percentage) excel function. This can be updated if we realize that we need more degrees of freedom. I've left in the alpha=0.1 and 0.05, as these are the only two we need. For chi squared 0% confidence level we always split. This is alpha = 1.
chi=pd.read_excel("Chi2_values.xlsx")

#loading in the training data
train_data = pd.read_csv("train.csv", names = ['Col1', 'Col2', 'Col3'])
test_data = pd.read_csv("test.csv", names = ['Col1', 'Col2'])
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
Att_dataframe = Att_dataframe.replace('N', 'G')


#Final att_dataframe for use in our code (training att_dataframe)
Att_dataframe.insert(len(list_of_string_lists[1]), "Result", train_data['Col3'], True)
print(Att_dataframe)

#print(Att_dataframe.iloc[0,:])
#-----------

#----------- Doing the same as above with the test data
column_names = []
#relabel the column with the 60 string DNA sequence as new labels -29 to 30. Assuming they are aligned at the middle of the sequence where the middle is the identifying factor, i would like the first split to register at 0. Checking this, it is true according to all three measures of uncertainty.
for x in list(range(0,len(test_data.iat[1,1]))):
    column_names.append(str(x-29))

#this turns the train.csv data into a more organized dataframe separated by position of base. Ordered from the first position (-29) to the last position (30), centered at 0. The last column is titled RESULT. Which is our target attribute.
TEST_dataframe = pd.DataFrame(columns = column_names)
list_of_string_lists = []
for x in list(range(0,test_data.shape[0])):
    list_of_string_lists.append(list(test_data.iat[x,1]))


#This att_dataframe is now the required format to run our code. Each position number is a string from -30 to 29 and the final column is labeled as "Result"
TEST_dataframe = pd.DataFrame(list_of_string_lists, columns = column_names)

#print(TEST_dataframe.iloc[838:840])
#Final att_dataframe for use in our code (training att_dataframe)
#Att_dataframe.insert(len(list_of_string_lists[1]), "Result", train_data['Col3'], True)
#print(Att_dataframe.iloc[0,:])
#-----------

##### We Can Run Everything Down Here using some hyperparameters to test many things at once.#########



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




class_list = Att_dataframe[Att_dataframe.iloc[:,-1].name].unique()
class_list.sort()

#can leave this in here to check on what we are keeping as the split. Can use split_choice to keep track of splits if we want to build a tree using the labels.
#print("split this node by attribute", split_choice)

#tests chi_squared for the split choice. Can comment out the observed and expected dataframes in chi_2 function file. I keep it to check numbers for sanity.

#print(chi_2(Att_dataframe, 1, split_choice, Att_dataframe.iloc[:,-1].name))
#make a set of sub dataframes set up by split chosen while removing the split column.


#variables inside:
    #cutoff int, use for pruning like chi_squared, or use =100 for no cutoff
    #branch_counter int, only useful for the size of the tree
    #impurity "str" (entropy, Gini, or mis)= impurity measure of interest
    #alpha float (0.01, 1, 0.05, 0.005): 1-%confidence for Chi squared.
    
cut_off = 12
branch_counter = 0
impurity = "entropy"
alpha = 0.005
#Basic idea of the tree node structure collection created with assistance from the buildTree function of https://medium.com/@lope.ai/decision-trees-from-scratch-using-id3-python-coding-it-up-6b79e3458de4
#Our algorithm requires many revisions very different from the source above, but help with the recursion aspect was researched, so I wanted to give credit to a source for this. Some sources were checked but this source was referenced on correctness. You can see artifacts of the sourse in examples like: if tree is...

#Att_dataframe = pandas dataframe of training data
#the rest are detailed at line 117
def Tree(Att_dataframe, impurity, branch_counter, cut_off, alpha, tree = None):

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
        elif chi_2(Att_dataframe, alpha, node, Att_dataframe.iloc[:,-1].name) == 0 or branch_counter > cut_off:
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
            print("branch counter is", branch_counter, cut_off)
            tree[node][x] = Tree(sub_frame, impurity, branch_counter, cut_off, alpha)
           
    return tree
    
t=Tree(Att_dataframe, impurity, branch_counter, cut_off, alpha)
pprint.pprint(t)



#this runs through the nested dictionary to check on all parts of an entry want this to take in a row of examples and try to guess the target. Was using this to get an idea of the structure of circumventing the list for help in understanding the prediction function below

def print_keys(nested_dict):
    for each_key in nested_dict:
    
        if nested_dict[each_key] in ['IE', 'EI', 'N']:
            print("the value here is,", nested_dict[each_key])
           
        else:
            new_dict = nested_dict[each_key]
            pprint.pprint(new_dict)
            print_keys(new_dict)
    return
    
#print(A.at[0, '0'])

#tree is the tree built from classifier, example is the set of examples to be predicted, and index_number is the row index in the dataframe of test samples. When we loop over index number, this will create a list of predictions.
print("start indices here")
Prediction_list = []
prediction_pairs = [0,0]
prediction_dataframe = pd.DataFrame(columns = ["Id", "Class"])

print(prediction_dataframe)

def prediction(tree, example, index_number):


    #if the tree is one element, aka leaf, only need to return the value here, as this is our prediction.
    if not isinstance(tree, dict):
        return tree
    else:
        #This is to get the attribtute that will serve as this stage's parent node. The code will choose a new parent node based upon the values within the tree and the example to be predicted.
        parent_node = next(iter(tree))
        
        #this is the value of that parent node at whatever the correct parent_node identifier is
        value_at_node = example.at[index_number,parent_node]
        
        #We want to see if this value at the parent node is in the existing tree. There are or will be some spots that have only 3 or 4 base options (including N) and we want a no prediction if that is the case. If there is more of the dictionary to parse through, then we need to run through to the next parent node.
        if value_at_node in tree[parent_node]:
            return prediction(tree[parent_node][value_at_node], example, index_number)
        else:
        #want to choose a value for class based upon training data distributions. Set seed back to 9000 to run code.
            #random.seed(time.time())
            #choice = random.choice(['N', 'N', 'IE', 'EI'])
            #random.seed(9000)
            return 'N'


for x in TEST_dataframe.index:
    print(x)
    prediction_pairs[0] = str(x+2001)
    prediction_pairs[1] = prediction(t, TEST_dataframe, x)
    prediction_dataframe.loc[x]=prediction_pairs

print(prediction_dataframe)

prediction_dataframe.to_csv('entropy995testpredictioncutoff12withNasGAssumption.csv', index=False)
#This is only for validation of our model with the training data, 1999/2000 matches with no stoppages
#total_correct= 0
#total_examples = 0
#new_df =
#for x in new_df.index:
#    total_examples += 1
#    if prediction_dataframe.iloc[x,-1] == new_df.iloc[x,-1]:
#        total_correct += 1
#    else:
#        print("not a match at position:", x)
#        print("the prediction is", prediction_dataframe.iloc[x,-1])
#print(total_examples)
#print(total_correct)
#print("the fraction of correct predictions is", total_correct/total_examples)

#----- This is simply for cost purposes. time to run should be proportional to tree size, all else being equal.
end_time = time.time()

print("This took", end_time-start_time, "seconds")
