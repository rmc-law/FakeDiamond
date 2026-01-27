# Stimulus creation 
The study uses minimal adjective-noun phrases to probe compositional semantic processing. Here are the steps taken to create stimuli.

1. Find pairs of concrete and abstract words: matched on Zipf frequency, age of acquisition, word length, using LexOPS (Taylor et al., 2020) implemented in R. (see `1_concreteness_manipulation.Rmd`). 

2. Subselecting nouns: the long list of nouns are then further subselected (see `2_subselect_nouns.ipynb`):
    - word length <= 7
    - concreteness standard deviation <= 1.2

3. Make phrases (see `3_make_phrases.ipynb`) and calculate transition probability and bigram frequency.

4. Make task questions: comprehension questions are created for 10% of the items, chosen at random. 

5. Construct stimulus lists for the experiment.