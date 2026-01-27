# A common framework for semantic memory and semantic composition
This is a repository for the publication: 

Law, R. M., Lambon Ralph, M. A., & Hauk, O. (2026). A common framework for semantic memory and semantic composition. Imaging Neuroscience. https://doi.org/10.1162/IMAG.a.1131 

OSF: https://osf.io/m82nb/

> **👋** Note: This repository is currently under active development. While it contains the core analysis scripts for the manuscript, we are currently tidying the documentation and validating the environment.

The study uses concurrent **MEG/EEG** and **structural MRI** to investigate whether the Anterior Temporal Lobe (ATL) supports both single-word meaning (semantic memory) and the construction of phrasal meaning (semantic composition) using shared neurocomputational principles.

Key features:
* **Stimulus generation:** Stimulus feature matching
* **Preprocessing:** Maxwell filtering (SSS), ICA, and automated trial rejection (Autoreject).
* **Source estimation:** L2-Minimum Norm Estimation (MNE) on distributed source space.
* **Univariate ROI analysis:** Mass-univariate ROI analysis using cluster-based permutation tests (Eelbrain).
* **Multivariate decoding analysis:** Time-resolved decoding and temporal generalization (Scikit-learn), plus hybrid Python/R mixed-effects modeling.

`environment.yml` contains all the python packages for this study. To create a conda environment from this file: `conda env create -f environment.yml`


## Data availability
The raw MEG, EEG, and MRI data are formatted according to the Brain Imaging Data Structure (BIDS). Data will be hosted at the MRC Cognition and Brain Sciences Unit (link to follow).