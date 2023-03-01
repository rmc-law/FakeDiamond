import pandas as pd

# CONCRETENESS
# concrete words are defined as words with high mean concreteness ratings,
# abstract words are defined as words with low mean concreteness ratings
ratings_all = pd.read_csv('glasgow_nroms.csv')



concrete = ratings_all.query('CNC.M > 4 & Bigram == 0')
abstract = ratings_all.query('Conc_M < 1 & Bigram == 0')
