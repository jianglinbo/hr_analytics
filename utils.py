__author__ = 'eok'


from sqlalchemy import create_engine
from conn_strings import conn_string
from pandas import read_csv, read_sql

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

print 'reading from file...'
df = _load_from_file()
print 'pushing data to db...'
_load_to_db(df)
print 'loading data from db...'
data = load_data()
print 'done'



# once data is loaded in the database, we will get rid of functions to load from file and load to db, and just use load data.
