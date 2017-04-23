__author__ = 'eok'


from sqlalchemy import create_engine
from conn_strings import conn_string
from pandas import read_csv, read_sql
import numpy as np
import pandas as pd

conn = create_engine(conn_string)

# loading the data from the csv
def _load_from_file():
    data = read_csv('hr_data.csv')
    return data

# pushes the df to the database
def _load_to_db(df):
    df.to_sql(name='kaggle_source', con=conn, if_exists='replace', chunksize=5000)

# pulls data from the database.
def load_data():
    data = read_sql('select * from kaggle_source',conn)
    return data

# data already loaded

# print 'reading from file...'
# df = _load_from_file()
# print 'pushing data to db...'
# _load_to_db(df)
# print 'loading data from db...'
# data = load_data()
# print 'done'
#


# once data is loaded in the database, we will get rid of functions to load from file and load to db, and just use load data.


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

    s_ = (s-mean_s)/np.sqrt(var_s)
    return s_, mean_s, var_s



def build_standardize(df, categoricals, params=None): # params e.g. variance, sd, mean. you want to calculate params for training data. you don't want it to do that for testing data.
    # Don't normalize categorical variables
    # Pull them out, hold them aside before processing
    df_hold = df[categoricals]
    df = df.drop(categoricals, axis=1)

    if params is None:
        params = {}
        df_norm = pd.DataFrame(columns=df.columns)
        # Iterate across columns
        for lab, col in df.iteritems():
            col_, col_mean, col_var = standardize(col) # calculating mean and variance and saving them bc need to later apply them to testing data. each feature gets its own mean/variance
            params[lab] = {'mean': col_mean, 'var': col_var}
            df_norm[lab] = col_

    else:
        df_norm = pd.DataFrame(columns=df.columns)
        # Iterate across columns
        for lab, col in df.iteritems():
            col_, col_mean, col_var = standardize(col, params[lab]['mean'], params[lab]['var'])
            df_norm[lab] = col_

    df_norm = df_norm.join(df_hold)
    return df_norm, params