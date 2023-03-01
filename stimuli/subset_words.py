import pandas as pd

# CONCRETENESS
# concrete words are defined as words with high mean concreteness ratings,
# abstract words are defined as words with low mean concreteness ratings
ratings_all = pd.read_csv('Concreteness_ratings_Brysbaert_et_al_BRM.csv')

concrete = ratings_all.query('Conc_M > 4 & Bigram == 0')
abstract = ratings_all.query('Conc_M < 1 & Bigram == 0')
