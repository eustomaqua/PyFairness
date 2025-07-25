# coding: utf-8
#
# Target:
#   Split one dataset into "training / validation / test" dataset
#
#   Research and Applications of Diversity in Ensemble Classification
#   Oracle bounds regarding fairness for majority voting
#


from copy import deepcopy
import gc
import numpy as np
from sklearn import preprocessing
from sklearn import model_selection
