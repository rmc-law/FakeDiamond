# Stimulus creation 
The study uses minimal adjective-noun phrases to probe semantic processing in the brain. Here are the steps taken to create stimuli.

1. Find pairs of concrete and abstract words: matched on Zipf frequency, age of acquisition, word length, using LexOPS (Taylor et al., 2020) implemented in R. (see `1_find_concreteness_word_pairs.Rmd`). 

2. Subselecting nouns: the long list of nouns are then further subselected (see `2_subselect_nouns.ipynb`):
    - word length <= 7
    - concreteness standard deviation <= 1.2

3. Make phrases (see `3_make_and_subselect_phrases.ipynb`)

4. Calculate transition probability and bigram frequency

5. Make task questions: comprehension questions are created for 10% of the items, chosen at random. 