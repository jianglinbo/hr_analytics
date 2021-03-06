from pandas import DataFrame, Series, read_csv, set_option
from numpy import array, sqrt
from random import seed, shuffle

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

from utils import load_data

# Want reproducible results
seed(0)

# Force the console output to be wider
set_option('display.width', 180)

# Read in data
# data = read_csv('hr_data.csv')
data = load_data()

# -------------------- pre-processing ----------------


def hr_pre_process(dat, drop_cols=['salary', 'dept']):
    dat = dat.rename(columns={'sales': 'dept'})

    salary_dict = {'low': 0, 'medium': 1, 'high': 2}

    dat['salary_map'] = dat['salary'].map(salary_dict)

    for col in drop_cols:
        dat.drop(col, axis=1, inplace=True)

    # Quick cleaning
    dat = dat.rename(columns={'promotion_last_5years': 'promotion', 'Work_accident': 'work_accident',
                                'average_montly_hours': 'average_monthly_hours',
                                'number_project': 'number_projects'})

    return dat

# For expository purposes, I am leaving everything explicitly defined in this file.
# These functions should be moved to a general purpose utility folder

# -------------------- quick viz ---------------------

# dat.apply(lambda x: sns.distplot(x))
#
# for i, s in data.iteritems():
#     plt.figure()
#     sns.distplot(s, label=i, hist=True, kde=False)
#     plt.legend()

# Quick notes
# Categorical:
#   salary_map (3 levels; ordinal)
#   promotion_last_5years (binary)
#   left (binary)
#   Work_accident (binary)

# Continuous:
#   time_spend_company (# years; beta?)
#   average_monthly_hours (# hours/month; bimodal)
#   number_project (#; ~normal)
#   last_evaluation (0.-1.; ~uniform?)
#   satisfaction_level (0.-1.; ~uniform?)

cat_cols = ['salary_map', 'promotion', 'left', 'work_accident'] # define for later use

# --------------- Feature Scaling -----------


def max_min_scale(s):
    return s/float(max(s) - min(s))

# dat = data.apply(lambda x: max_min_scale(x))


def calc_mean(s):
    return sum(s)/float(len(s))


def calc_var(s):
    mean_s = calc_mean(s)
    return sum((s-mean_s)**2)/float(len(s))


# Standardize data - this is the z-score
def standardize(s, mean_s=None, var_s=None):
    if mean_s is None:
        mean_s = calc_mean(s)

    if var_s is None:
        var_s = calc_var(s)

    s_ = (s-mean_s)/sqrt(var_s)
    return s_, mean_s, var_s


def build_standardize(df, categoricals, params=None): # params e.g. variance, sd, mean. you want to calculate params for training data. you don't want it to do that for testing data.
    # Don't normalize categorical variables
    # Pull them out, hold them aside before processing
    df_hold = df[categoricals]
    df = df.drop(categoricals, axis=1)

    if params is None:
        params = {}
        df_norm = DataFrame(columns=df.columns)
        # Iterate across columns
        for lab, col in df.iteritems():
            col_, col_mean, col_var = standardize(col) # calculating mean and variance and saving them bc need to later apply them to testing data. each feature gets its own mean/variance
            params[lab] = {'mean': col_mean, 'var': col_var}
            df_norm[lab] = col_

    else:
        df_norm = DataFrame(columns=df.columns)
        # Iterate across columns
        for lab, col in df.iteritems():
            col_, col_mean, col_var = standardize(col, params[lab]['mean'], params[lab]['var'])
            df_norm[lab] = col_

    df_norm = df_norm.join(df_hold)
    return df_norm, params

# What did standardization do?
dat, _ = build_standardize(hr_pre_process(data), cat_cols)
# non-standardized (std = standard deviation)
print 'non-standardized average_monthly_hours s.d.: {}'.format(dat['average_monthly_hours'].std())
# standardized
print 'standardized average_monthly_hours s.d.: {}'.format(dat['average_monthly_hours'].std())


# -------------------- Modeling section ---------------- #

# In a real ML flow, you may not have access to validation data ahead of time.
# To emulate this, you can't include the validation set in the parameterization of the features
# So, while we calculated scaled values above, that was only for expository purposes
# To properly build a model, we first split into training/validation, then use the training set
# exclusively to determine our feature parameters (mean, std), and then apply those parameters
# to the validation set when we're ready.

# shuffle
idx = array(data.index)
shuffle(idx)
dat = data.ix[idx].reset_index(drop=True)

f = 0.25
N = len(dat)
# validation set
dat_val = dat.iloc[:int(f*N), :].reset_index(drop=True)  # first 25% (f%)
# training set
dat = hr_pre_process(dat.iloc[int(f*N):, :].reset_index(drop=True))  # remaining data


#: Let sklearn find stratified splits for us
#: Shuffle is unnecessary here, since we shuffled above, but whatever
SKF = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
k_splits = SKF.split(dat.drop('left', axis=1), dat['left'])

#: Per split, train model and use run_model to collect probabilities, predictions, and actual values
#: Push into dataframe on each loop for analysis after all folds
accuracy = []
for train_idx, test_idx in k_splits:

    train = dat.iloc[train_idx, :]  # pull out training data
    test = dat.iloc[test_idx, :]  # pull out testing data

    train_, params = build_standardize(train, cat_cols) # the data is standardized
    test_, _ = build_standardize(test, cat_cols, params)

    init_weights = train['left'].value_counts().to_dict()

    # training model step
    tmp_model = RandomForestClassifier(class_weight=init_weights).fit(train_.drop('left', axis=1), train_['left'])

    # run model (make predictions)
    y = test_['left']
    y_pred = tmp_model.predict(test_.drop('left', axis=1))

    # Calculate accuracy = (# correct classifications)/(# inputs)
    acc = sum(y==y_pred)/float(len(y))

    accuracy.append(acc)

print 'Average training accuracy: {}%'.format(round(sum(accuracy)/len(accuracy), 3)*100.)

# ------------ up to this point, I can tweak/change model and/or model parameters ----------------
# ---------- !! no decisions can be based on the results of the validation set !! ----------------

# re-parameterize with entire training set
dat_, params = build_standardize(dat, cat_cols)
dat_val_, _ = build_standardize(hr_pre_process(dat_val), cat_cols, params)

init_weights = dat_['left'].value_counts().to_dict()

# Implement model
model = RandomForestClassifier(class_weight=init_weights).fit(dat_.drop('left', axis=1), dat_['left'])

# feats = [(f, i) for i, f in zip(model.feature_importances_, dat_.drop('left', axis=1).columns)]
feats = Series(data=model.feature_importances_, index=dat_.drop('left', axis=1).columns, name='importance')

# run predictions on validation step
y = dat_val_['left']
y_pred = model.predict(dat_val_.drop('left', axis=1))

gen_acc = sum(y==y_pred)/float(len(y))
gen_err = 1. - gen_acc

print 'Expected generalization accuracy: {a}%  (error: {e}%)'.format(a=round(gen_acc, 3)*100., e=round(gen_err, 3)*100.)

# Insights #

# Join predictions back in
dat_val = hr_pre_process(dat_val, drop_cols=['salary'])
dat_val['pred'] = y_pred

print 'Expected number of people leaving: {}'.format(dat_val['pred'].sum())
print 'Expected fraction of employees leaving: {}'.format(dat_val['pred'].sum()/float(len(dat_val['pred'])))

# Department with highest turnover rates
loss = dat_val.groupby('dept').apply(lambda x: x['pred'].sum()/float(len(x['pred']))).sort_values(ascending=False)

dat_val['promotion'].value_counts()

# Anything interesting here?
d = dat_val.groupby('pred').mean()

# that's a little too course. let's look at these numbers broken out by department
dat_val.groupby(['dept', 'pred']).mean().unstack('pred')

dat_val.groupby(['time_spend_company', 'promotion'])['pred'].mean().unstack('promotion')




