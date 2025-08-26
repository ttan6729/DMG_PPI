import operator
################################################
# fix the random see value so the results are re-producible
seed_value = 7
import os
# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy as np

np.random.seed(seed_value)
###############################################

import csv
import logging
#import datetime
#import matplotlib

# matplotlib.use('pdf')
# import matplotlib.pyplot as plt
from pandas import DataFrame

import time
import sys
#from sklearn.metrics import roc_curve, auc, precision_recall_curve
#from sklearn.model_selection import StratifiedKFold, KFold
#from sklearn.utils import class_weight
#from itertools import chain
#import argparse
import math
Dict_3mer_to_100vec={}

# dim: delimiter
def get_3mer_and_np100vec_from_a_line(line, dim):
    np100 = []
    line = line.rstrip('\n').rstrip(' ').split(dim)
    three_mer = line.pop(0)
    np100 += [float(i) for i in line]
    np100 = np.asarray(np100)
    return three_mer, np100

def LoadPro2Vec(fName):
    f = open(fName, "r")
    index = 0
    while True:
        line = f.readline().replace('"', '')
        if not line:
            break
        three_mer, np100vec = get_3mer_and_np100vec_from_a_line(line, '\t')
        Dict_3mer_to_100vec[three_mer] = np100vec

        index += 1
    print ("total number of unique 3mer is ", index)

def GetFeature(ThreeMer, Feature_dict):
    if (ThreeMer not in Feature_dict):
        print("[warning]: Feature_dict can't find ", ThreeMer, ". Returning 0")
        return 0
    else:
        return Feature_dict[ThreeMer]

def RetriveFeatureFromASequence(seq, Feature_dict):
    seq = seq.rstrip('\n').rstrip(' ')
    assert (len(seq) >= 3)
    Feature = []
    for index, item in enumerate(seq):
        sta = index - 1
        end = index + 1
        if ((sta < 0) or (end >= len(seq))):
            Feature.append(Feature_dict["<unk>"])
        else:
            Feature.append(GetFeature(seq[sta:sta+3], Feature_dict))
    return Feature

def CalProtVec(seqList): 
    result = []
    LoadPro2Vec("FeatureComputation/protVec_100d_3grams.csv")
    for key,value in Dict_3mer_to_100vec.items():
        Dict_3mer_to_100vec[key] = np.sum(value)
    # print(Dict_3mer_to_100vec["AAA"])
    max_key = max(Dict_3mer_to_100vec.keys(), key=(lambda k: Dict_3mer_to_100vec[k]))
    min_key = min(Dict_3mer_to_100vec.keys(), key=(lambda k: Dict_3mer_to_100vec[k]))
    max_value = Dict_3mer_to_100vec[max_key]
    min_value = Dict_3mer_to_100vec[min_key]

    for key,value in Dict_3mer_to_100vec.items():
        Dict_3mer_to_100vec[key] = (Dict_3mer_to_100vec[key] - min_value) / (max_value - min_value)
    for seq in seqList:
        vec = RetriveFeatureFromASequence(seq, Dict_3mer_to_100vec)
        result.append( sum(vec) / (len(vec)-2) )

    result = np.array(result,dtype='float').reshape(len(seqList),-1)
    print(f'ProtVec shape: {result.shape}')
    return result

