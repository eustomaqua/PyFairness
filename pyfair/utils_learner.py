# coding: utf-8
#
# TARGET:
#   Measuring fairness via data manifolds
#       classifier-related, or ensemble
#


# sklearn
from sklearn import tree
from sklearn import naive_bayes
from sklearn import svm
from sklearn import linear_model
from sklearn import neighbors
from sklearn import neural_network

from sklearn.ensemble import (
    BaggingClassifier, AdaBoostClassifier,)
#   RandomForestClassifier, ExtraTreesClassifier,
#   GradientBoostingClassifier,  # HistGradientBoosting
#   VotingClassifier, StackingClassifier

# import lightgbm
# import fairgbm
from lightgbm import LGBMClassifier
from fairgbm import FairGBMClassifier
from pyfair.pkgs_AdaFair_py36 import AdaFair
from pyfair.marble.data_classify import EnsembleAlgorithm

# import sklearn
# skl_ver = sklearn.__version__
# if skl_ver.startswith('1.3'):
#     from experiment.utils.pkgs_AdaFair_py36 import AdaFair
# elif skl_ver.startswith('1.5.1'):
#     pass
# del skl_ver


AVAILABLE_ABBR_CLS = [
    'DT', 'NB', 'SVM', 'linSVM', 'MLP',
    'LR1', 'LR2', 'LM1', 'LM2', 'kNNu', 'kNNd', 
]   # ALG_NAMES    # 'lmSGD','LR'

FAIR_INDIVIDUALS = {
    'DT': tree.DecisionTreeClassifier(),
    'NB': naive_bayes.GaussianNB(),
    'LR': linear_model.LogisticRegression(max_iter=500),
    'LR1': linear_model.LogisticRegression(
        penalty=None, max_iter=500),  # not 'none'
    'LR2': linear_model.LogisticRegression(
        penalty='l2', max_iter=500),  # default

    'SVM': svm.SVC(),
    'linSVM': svm.LinearSVC(max_iter=5000),
    'kNNu': neighbors.KNeighborsClassifier(
        weights='uniform'),  # default
    'kNNd': neighbors.KNeighborsClassifier(
        weights='distance'),

    'MLP': neural_network.MLPClassifier(max_iter=1000),
    # 'NN': neural_network.MLPClassifier(
    #     solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,2)),
    'lmSGD': linear_model.SGDClassifier(),
    'LM1': linear_model.SGDClassifier(penalty='l1'),  # 'l1'
    'LM2': linear_model.SGDClassifier(penalty='l2'),  # default
}

# INDIVIDUALS = FAIR_INDIVIDUALS
# del FAIR_INDIVIDUALS

CONCISE_INDIVIDUALS = {
    'DT': tree.DecisionTreeClassifier(),
    'NB': naive_bayes.GaussianNB(),
    'LR': linear_model.LogisticRegression(),
    'SVM': svm.SVC(),
    'linSVM': svm.LinearSVC(),
    'kNNu': neighbors.KNeighborsClassifier(weights='uniform'),  # default
    'kNNd': neighbors.KNeighborsClassifier(weights='distance'),
    'MLP': neural_network.MLPClassifier(),
    'lmSGD': linear_model.SGDClassifier(),
}
