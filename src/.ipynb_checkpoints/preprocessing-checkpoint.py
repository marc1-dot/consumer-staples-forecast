"""
preprocessing.py
----------------
This script prepares and cleans the data collected by `data_loader.py`.
It merges price data and financial statement data, handles missing values
(including EPS gaps before 2019), and creates derived features for modeling.


Author: Marc Birchler
Course: Advanced Programming - HEC Lausanne (Fall 2025)
"""


# ============================
# Import required libraries
# ============================
import os
import pandas as pd
import numpy as np
