__author__ = 'eok'

import pandas as pd
import numpy as np
from utils import load_data, _load_from_file, hr_pre_process, build_standardize
from random import seed, shuffle
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
# logistic regression assigns a class. you left or you didn't. no probability.


pd.set_option('display.width', 180)

# data = load_data()
data = _load_from_file()

seed(0) # is the basis for randomization. it could be a datetime; any number. and random variables are instantiated from that seed.
# get the same random (shuffled) output each time. if diff everytime, cannot compare results.
# if you give none, it seeds from current time; which is different everytime you call.

# shuffle
idx = np.array(data.index)
shuffle(idx)
dat = data.ix[idx].reset_index(drop=True)

f = 0.25
N = len(dat)
# validation set
dat_val = dat.iloc[:int(f*N), :].reset_index(drop=True)  # first 25% (f%)
# training set
dat = hr_pre_process(dat.iloc[int(f*N):, :].reset_index(drop=True)) # remaining data


#: Let sklearn find stratified splits for us
#: Shuffle is unnecessary here, since we shuffled above, but whatever

# Stratified k folds preserves the ratio of your output classes. imp bc the ratio of leavers to stayers is unbalanced. e.g. 4:1
# For every model, make sure to preserve the number of folds.
SKF = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
# split data into k number of folds. drop 'left' b/c we want to predict who will leave
# gives you index at which to split data into training and test set. it gives you a generator which will iterate over tuples of training and testing.
# k_splits.next() says go through one iteration and return results. generators yield something at every step/iteration.
k_splits = SKF.split(dat.drop('left', axis=1), dat['left'])

cat_cols = ['salary_map', 'promotion', 'left', 'work_accident'] # these are categoricals that should be dropped for now.

#: Per split, train model and use run_model to collect probabilities, predictions, and actual values
#: Push into dataframe on each loop for analysis after all folds
accuracy = []
for train_idx, test_idx in k_splits: # SEE DOCUMENTATION STRATIFIEDKFOLD FOR MORE INFO!

    train = dat.iloc[train_idx, :]  # pull out training data
    test = dat.iloc[test_idx, :]  # pull out testing data

    train_, params = build_standardize(train, cat_cols) # the data is standardized (in particular way). pull the categoricals out in order to standardize the rest.
    test_, _ = build_standardize(test, cat_cols, params)

    init_weights = train['left'].value_counts().to_dict() # this is to represent the bias in my classes (leavers and stayers). unnecessary for logisticregression

    # training model step.
    tmp_model = LogisticRegression(fit_intercept=False, penalty='l1')  # don't want to add biases b/c the categorical/discrete variables (salary_map, promotion, work_accident) coalesce act as the constant of the function.
    # penalty is default l2. l1 is the absolute value of l. l is between 0 and 1 because it's a binary classification. error in a classifier is between 0 and 1.
    x_train = train_.drop('left', axis=1)
    y_train = train_['left']
    tmp_model.fit(x_train,y_train) # fit(X, y). X = standardized training data.
    # The target variable ('left') was dropped from the standardized training data.
    # Y = is the target variable 'left'

    # run model (make predictions)
    x_test = test_.drop('left', axis=1)
    y_test = test_['left']
    y_pred = tmp_model.predict(x_test)

    # Calculate accuracy = (# correct classifications)/(# inputs)
    acc = sum(y_test==y_pred)/float(len(y_test))

    accuracy.append(acc)

print 'Average training accuracy: {}%'.format(round(sum(accuracy)/len(accuracy), 3)*100.)





# re-parameterize with ENTIRE training set (because the previous tmp_models was just to calculate accuracy)
dat_, params = build_standardize(dat, cat_cols) # training
dat_val_, _ = build_standardize(hr_pre_process(dat_val), cat_cols, params) # validation data from above

# Implement model
model = LogisticRegression(fit_intercept=False, penalty='l1')
x_train = dat_.drop('left', axis=1)
y_train = dat_['left']
model.fit(x_train,y_train)


# run predictions on validation step
y_val = dat_val_['left'] # target variable from validation dataset
x_val = dat_val_.drop('left', axis=1) # input variables from validation dataset
y_pred = model.predict(x_val) # predict y using validation data and the model generated from training data

gen_acc = sum(y_val==y_pred)/float(len(y_val))   # accuracy = # of matches between targeted Y and predicted Y / # of observations of target in validation set
gen_err = 1. - gen_acc

print 'Expected generalization accuracy: {a}%  (error: {e}%)'.format(a=round(gen_acc, 3)*100., e=round(gen_err, 3)*100.)

# sergey's way
print 'model accuracy:',metrics.accuracy_score(y_val, y_pred) # this accuracy score is for classification only.



# Insights #

# Join predictions back in
dat_val = hr_pre_process(dat_val, drop_cols=['salary'])
dat_val['pred'] = y_pred

print 'Expected number of people leaving: {}'.format(dat_val['pred'].sum())
print 'Expected fraction of employees leaving: {}'.format(dat_val['pred'].sum()/float(len(dat_val['pred'])))


# logistic regression is not as robust as the random forest.
# the logistic regression predicts that many more people will attrit and predicts a one percentage point higher attrition rate than the random forest does.

# next: we can look at which people in the logistic regression the model got wrong and see if there are similarities
# between people that it got wrong vs. it got right.

# Type 1 vs. type 2. True Positives with False Positives and True Negatives with False Positives.

# predict_proba(X) - predicting of belonging to each class (0,1 stayer or leaver) as opposed to assigning classes to each observation/person.

# in a linear regression, predicting on a continuous spectrum where a target variable might land.


