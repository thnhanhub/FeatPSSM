"""
implementation based on GcForest-PPI model
"""


import numpy as np
import pandas as pd
import scipy.io as sio
import utils.tools as utils
from gcForest.lib.gcforest.gcforest import GCForest
from gcForest.lib.gcforest.utils.config_utils import load_json
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import scale


path1 = 'gcforest4.json'  # TODO

config = load_json(path1)
gc = GCForest(config)


# TODO
