import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier


pd.set_option('display.width', 180)

# -------------------- pre-processing ----------------

data = pd.read_csv('hr_data.csv')

data = data.rename(columns={'sales': 'dept'})

salary_dict = {'low': 0, 'medium': 1, 'high': 2}

data['salary_map'] = data['salary'].map(salary_dict)


def max_min_normalize(s):
    return s/(float(max(s)) - float(min(s)))

dat = data.drop(['dept', 'salary'], axis=1).apply(lambda x: max_min_normalize(x))

# -------------------- quick viz ----------------

# dat.apply(lambda x: sns.distplot(x))

plt.figure()
for i, s in dat.iteritems():
    sns.distplot(s, label=i, hist=True, kde=False)
plt.legend()

# -------------------- Modeling section ---------------- #

# shuffle
dat = dat.sample(frac=1.).reset_index(drop=True)

dat_store = dat.copy()

f = 0.25
N = len(dat)
# validation set
dat_val = dat.iloc[:int(f*N), :]  # first 25% (f%)
# training set
dat = dat.iloc[int(f*N):, :]  # remaining data

#: Let sklearn find stratified splits for us
SKF = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
k_splits = SKF.split(dat.drop('left', axis=1), dat['left'])

#: Per split, train model and use run_model to collect probabilities, predictions, and actual values
#: Push into dataframe on each loop for analysis after all folds
accuracy = []
for train_idx, test_idx in k_splits:

    train = dat.iloc[train_idx, :]  # pull out training data
    test = dat.iloc[test_idx, :]  # pull out testing data

    # training model step
    tmp_model = RandomForestClassifier().fit(train.drop('left', axis=1), train['left'])

    # [(f, i) for i, f in zip(tmp_model.feature_importances_, dat.drop('left', axis=1).columns)]

    # run model (make predictions)
    y = test['left']
    y_pred = tmp_model.predict(test.drop('left', axis=1))

    # Calculate accuracy = (# correct classifications)/(# inputs)
    acc = sum(y==y_pred)/float(len(y))

    accuracy.append(acc)

print 'expected accuracy: {}'.format(round(sum(accuracy)/len(accuracy), 3)*100.)

# ------------ up to this point, I can tweak/change model and/or model parameters ----------------

# Implement model
trained_model = RandomForestClassifier().fit(dat.drop('left', axis=1), dat['left'])

# run predictions on validation step
y = dat_val['left']
y_pred = trained_model.predict(dat_val.drop('left', axis=1))

gen_acc = 1. - sum(y==y_pred)/float(len(y))

print 'Expected generalization error: {}%'.format(round(gen_acc, 3)*100.)
