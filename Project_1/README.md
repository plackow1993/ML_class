# ML_class

2/7/22 - Ken - functions.py contains each error function we want to use: misclass, gini, and entropy. A subset of a dataframe with the last column being the target classifications will work as input. I have two test DF's in the file to check on its utility if you like.

2/9/22 - Ken - chi_2.py contains the full computation of chi squared statistic. Will have to adjust to be called into our code, but it computes chi and makes a decision for any dataframe. The inputs will have to be alpha, attribute to split, key of target classes column.

Running the code for project 1:

In your working directory must exist the following documents from our submission but remain untouched:

    Chi2_values.xlsx - this has all alpha values that we can choose for our chi_2 alpha. These are for confidence levels of 95, 99, 99.1, 99.2, 99.3, 99.4, 99.5, 99.6, 99.7. 1-%Confidence = alpha. For a 0% confidence, set alpha =1 in the predictions file.

    test.csv - this is the testing data, without it you wont see a table of predictions made and will hit a bug AFTER the tree is built.

    train.csv - this is the training data to build the tree, absolutely necessary for the ATT_dataframe to be created.

    info_gain.py - required for informaiton gain

    functions.py - these consist of the three impurity measures requested
    
    chi_2.py - this calculates chi squared for each split
    
To run the project code:

    Att dataframe is already loaded from the training data. Inside the PREDICTION.PY file is the code that builds the tree and makes a prediction on the test data.
    
    In prediction.py, you can change your string for the impurity function, change your desired alpha value, and an additional branch cutoff value. To save the prediciton csv, uncomment the line 249, and change the name of the csv file to whatever description you like. We used the format [impurity][%confidence]testpredictioncutoff[cutoff]*.csv
    
    You can change these inputs at line 127. this is the only thing (besides the printing) that should be changed.
    
    Branch_counter was only to keep track of consistency with cutoff, but there was not a need to get rid of it.
    
    Cut-off: this is an added pruning measure that cuts the branches off at a certain length. Because this counts the node and an attribute together, the actual depth of the tree is cutoff/2. For example, for a cutoff value of 12, the depth is 6, counting only the nodes.
    
I ran this on python 3.9.7. From my understanding this will work with python 3.

run python prediction.py and your output will be a tree and a prediction for your test data.
One caveat, the test data has labels 2001-3190 and the prediction of the test data accomodates this. If there are different labels for your test data, youll have to update that to make a successful prediction with Kaggle.
    
    
