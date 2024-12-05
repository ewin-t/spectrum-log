This repository contains the code for Section 8 of the paper "Beating full state tomography for unentangled spectrum estimation" by Angelos Pelecanos, Xinyu Tan, Ewin Tang, and John Wright.

## Main Experiment

The main experiment is conducted using the script `test_two_dist.py`. This script performs the following tasks:
- Simulate two distributions that match on the first k moments, with k up to 3.
- Run the maximum likelihood tester. 

## Main Functions

The `schur.py` script contains the main function for evaluating a schur polynomial. 

The `rsk.py` script contains the main function for performing the RSK algorithm. 

The `fit_plot.py` script is used for curve fitting and generating plots based on the experimental data from `test_two_dist.py`.
