Stimulus creation for semantic composition + semantic memory

1. Use LexOPS in R to find a set of concrete and abstract words that are matched on Zipf Frequency, Age of Acquisition, Length, Part of Speech 
(see 1_find_concreteness_word_pairs.Rmd for details). 
Seed number: 42
Input: Norming data (e.g., Brysbaert_etal_2014.csv or glasgow_norms_column_renamed.csv) 
Output: stim_long_brysbaert_all_possible_seed42.csv (all possible pairs of concrete vs. abstract words whilst matching the aforementioned variables) 

2. This long list of concrete and abstract nouns are then further subselected in the jupyter notebook `noun_selection.ipynb`.
- first restrict word length of each condition (abstract and concrete) to 7 (inclusive) and below. 
- then restrict word category to nouns

3. This resultant list is then visualise w.r.t. to their frequency, AoA, and Length. 

4. It's then saved as a .csv file with the filename `nouns_{date.today()}.csv`

5. The file is then converted to .xlsx in Excel for further hand-selection. 