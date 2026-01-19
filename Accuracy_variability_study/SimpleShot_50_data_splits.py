import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
import math
from typing import List, Dict, Any, Tuple
import random
import tqdm
from collections import defaultdict
import os

# pytorch modules
from torch.utils.data import Dataset # primitive for the data
#from torchvision.transforms import ToTensor
from torch.utils.data import WeightedRandomSampler, DataLoader # wraps the data so its iterable
import torch
from torchmetrics import Accuracy
import torch.nn as nn # nn class our model inherits from
import torch.optim as optim
import torchvision.transforms.functional as F
from torchvision.transforms.transforms import RandomHorizontalFlip, RandomVerticalFlip
from torchvision import transforms
import torchvision.models as models


import time
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import FSL_funcs as FSL
import FSL_models as FSLm

if __name__ == '__main__':

    save_folder = 'SimpleShot_50_data_splits_results'

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    print(f"Saving to directory: {save_folder}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    cropped_scans = np.load(r'data/defSTM.npy')#.transpose((0,3,1,2))
    labels = np.load(r'data/defSTM_labels.npy')

    # since we only have 2 examples of label h1, we will augment it so we can at least carry out N=5,K=3, n_query=15, with this data.
    # not good practice in general but no other choice if we want to test on this experimental data
    h1s = cropped_scans[labels==18]
    h1s_flip = np.fliplr(h1s)
    h1s_rot1 = np.rot90(h1s, k=1)
    h1s_rot1_flip = np.fliplr(h1s_rot1)
    h1s_rot2 = np.rot90(h1s, k=2)
    h1s_rot2_flip = np.fliplr(h1s_rot2)
    h1s_rot3 = np.rot90(h1s, k=3)
    h1s_rot3_flip = np.fliplr(h1s_rot3)
    h1s = np.vstack( (h1s_flip, h1s_rot1, h1s_rot2, h1s_rot3, h1s_rot1_flip, h1s_rot2_flip, h1s_rot3_flip, h1s, h1s_rot1) )
    cropped_scans = np.vstack( (cropped_scans, h1s) )
    labels = np.hstack( (labels, 18*np.ones(18,) ) )

    # h2s are also less than 18 so we add 10 more of them
    h2s = cropped_scans[labels==19]
    h2s_flip = np.fliplr(h2s)
    cropped_scans = np.vstack( (cropped_scans, h2s_flip) )
    labels = np.hstack( (labels, 19*np.ones(10,) ) )

    # do same for their inverses
    h1s = cropped_scans[labels==23]
    h1s_flip = np.fliplr(h1s)
    h1s_rot1 = np.rot90(h1s, k=1)
    h1s_rot1_flip = np.fliplr(h1s_rot1)
    h1s_rot2 = np.rot90(h1s, k=2)
    h1s_rot2_flip = np.fliplr(h1s_rot2)
    h1s_rot3 = np.rot90(h1s, k=3)
    h1s_rot3_flip = np.fliplr(h1s_rot3)
    h1s = np.vstack( (h1s_flip, h1s_rot1, h1s_rot2, h1s_rot3, h1s_rot1_flip, h1s_rot2_flip, h1s_rot3_flip, h1s, h1s_rot1) )
    cropped_scans = np.vstack( (cropped_scans, h1s) )
    labels = np.hstack( (labels, 23*np.ones(18,) ) )

    # h2s are also less than 18 so we add 10 more of them
    h2s = cropped_scans[labels==24]
    h2s_flip = np.fliplr(h2s)
    cropped_scans = np.vstack( (cropped_scans, h2s_flip) )
    labels = np.hstack( (labels, 24*np.ones(10,) ) )

    # convert to torch tensors
    x = torch.tensor(cropped_scans).float().to(device)
    y = torch.tensor(labels).float().to(device)


    results_dict = {}

    n_splits = 2

    ####################################################################
    ####################################################################
    #### conv4
    ####################################################################

    ##################
    ##### models with the inverse features included, TiO2(110)
    ##################

    n_way= 2
    n_support = 3

    train_val_features = ['single dangling bond' , 'As A' ,  'C defect', 'siloxane inv', 
                        'single DV on Si(001)', 'C defect inv', 'double dangling bond', 'm1 inv',
                        'single dihydride', 'double dangling bond inv', 'As B inv', 'h1',
                        'h2', 't1 inv', 'g1 inv', 'h2 inv', 'h1 inv', 't1' , 'single dangling bond inv', 'As A inv' , 'm1', 'g1',  
                        'single dihydride inv', 'As B','single DV on Si(001) inv','siloxane']
    test_features = ['TiO2_vacancy', 'TiO2_hydroxyl']

    save_path = 'TiO2_40pix_inv'

    # lists of acc, precision, recall
    accuracy3 = []
    accuracy1 = []
    precision3 = []
    precision1 = []
    recall3 = []
    recall1 = []
    for i in range(n_splits):
        FSL.set_seed(42, DEVICE = device)
        avg_acc3, std_acc3, avg_precision3, std_precision3, avg_recall3, std_recall3, avg_acc1, std_acc1, avg_precision1, std_precision1, avg_recall1, std_recall1 = FSL.simple_shot_training_eval(x, y, train_val_features, test_features, n_way, save_path, i*20, inits=1, device=device)
        accuracy3.append(avg_acc3)
        accuracy1.append(avg_acc1)
        precision3.append(avg_precision3)
        precision1.append(avg_precision1)
        recall3.append(avg_recall3)
        recall1.append(avg_recall1)
    
    avg_acc3 = torch.mean(torch.tensor(accuracy3))
    avg_acc1 = torch.mean(torch.tensor(accuracy1))
    avg_precision3 = np.mean(precision3)
    avg_precision1 = np.mean(precision1)
    avg_recall3 = np.mean(recall3)
    avg_recall1 = np.mean(recall1)
    std_acc3 = torch.std(torch.tensor(accuracy3))
    std_acc1 = torch.std(torch.tensor(accuracy1))
    std_precision3 = np.std(precision3)
    std_precision1 = np.std(precision1)
    std_recall3 = np.std(recall3)
    std_recall1 = np.std(recall1)

    print('='*50)
    print(f'Final accuracy and standard deviation over all runs (3-shot): {avg_acc3:.4f} ± {std_acc3:.4f}')
    print(f'Final accuracy and standard deviation over all runs (1-shot): {avg_acc1:.4f} ± {std_acc1:.4f}')
    print('='*50)

    results_dict[f'SimpleShot({3},{15})_{save_path}'] = (avg_acc3, std_acc3, avg_precision3, std_precision3, avg_recall3, std_recall3)
    results_dict[f'SimpleShot({1},{15})_{save_path}'] = (avg_acc1, std_acc1, avg_precision1, std_precision1, avg_recall1, std_recall1)


    ##################
    ##### models with the inverse features included, Ge(001)
    ##################

    n_way = 4

    train_val_features = ['single dangling bond' , 'As A' ,  'C defect', 'siloxane inv', 
                    'single DV on Si(001)', 'C defect inv', 'TiO2_hydroxyl inv', 'double dangling bond', 
                    'single dihydride', 'double dangling bond inv', 'TiO2_vacancy', 'As B inv',
                    'single dangling bond inv', 'As A inv' , 'TiO2_hydroxyl', 'TiO2_vacancy inv', 
                    'single dihydride inv', 'As B','single DV on Si(001) inv','siloxane']
    test_features = ['t1', 'g1', 'm1', 'h2', 'h1']

    save_path = 'Ge(001)_40pix_inv'

    # lists of acc, precision, recall
    accuracy3 = []
    accuracy1 = []
    precision3 = []
    precision1 = []
    recall3 = []
    recall1 = []
    for i in range(n_splits):
        FSL.set_seed(42, DEVICE = device)
        avg_acc3, std_acc3, avg_precision3, std_precision3, avg_recall3, std_recall3, avg_acc1, std_acc1, avg_precision1, std_precision1, avg_recall1, std_recall1 = FSL.simple_shot_training_eval(x, y, train_val_features, test_features, n_way, save_path, i*20, inits=1, device=device)
        accuracy3.append(avg_acc3)
        accuracy1.append(avg_acc1)
        precision3.append(avg_precision3)
        precision1.append(avg_precision1)
        recall3.append(avg_recall3)
        recall1.append(avg_recall1)
    
    avg_acc3 = torch.mean(torch.tensor(accuracy3))
    avg_acc1 = torch.mean(torch.tensor(accuracy1))
    avg_precision3 = np.mean(precision3)
    avg_precision1 = np.mean(precision1)
    avg_recall3 = np.mean(recall3)
    avg_recall1 = np.mean(recall1)
    std_acc3 = torch.std(torch.tensor(accuracy3))
    std_acc1 = torch.std(torch.tensor(accuracy1))
    std_precision3 = np.std(precision3)
    std_precision1 = np.std(precision1)
    std_recall3 = np.std(recall3)
    std_recall1 = np.std(recall1)

    print('='*50)
    print(f'Final accuracy and standard deviation over all runs (3-shot): {avg_acc3:.4f} ± {std_acc3:.4f}')
    print(f'Final accuracy and standard deviation over all runs (1-shot): {avg_acc1:.4f} ± {std_acc1:.4f}')
    print('='*50)

    results_dict[f'SimpleShot({3},{15})_{save_path}'] = (avg_acc3, std_acc3, avg_precision3, std_precision3, avg_recall3, std_recall3)
    results_dict[f'SimpleShot({1},{15})_{save_path}'] = (avg_acc1, std_acc1, avg_precision1, std_precision1, avg_recall1, std_recall1)

    ##################
    ##### models with the inverse features included, Si(001)
    ##################

    n_way= 4

    train_val_features = ['single dangling bond' , 'As A' , 'siloxane', 'C defect', 'g1' , 'As A inv' ,
                  'siloxane inv', 'g1 inv', 'C defect inv', 'TiO2_vacancy', 'h1', 'm1', 'h2', 't1', 
                  'TiO2_vacancy inv', 'TiO2_hydroxyl inv', 'm1 inv', 'single dangling bond inv','TiO2_hydroxyl', 'h2 inv', 't1 inv', 'h1 inv']
    test_features = ['double dangling bond', 'As B', 'single dihydride', 'single DV on Si(001)']

    save_path = 'Si(001)_40pix_inv'

    # lists of acc, precision, recall
    accuracy3 = []
    accuracy1 = []
    precision3 = []
    precision1 = []
    recall3 = []
    recall1 = []
    for i in range(n_splits):
        FSL.set_seed(42, DEVICE = device)
        avg_acc3, std_acc3, avg_precision3, std_precision3, avg_recall3, std_recall3, avg_acc1, std_acc1, avg_precision1, std_precision1, avg_recall1, std_recall1 = FSL.simple_shot_training_eval(x, y, train_val_features, test_features, n_way, save_path, i*20, inits=1, device=device)
        accuracy3.append(avg_acc3)
        accuracy1.append(avg_acc1)
        precision3.append(avg_precision3)
        precision1.append(avg_precision1)
        recall3.append(avg_recall3)
        recall1.append(avg_recall1)
    
    avg_acc3 = torch.mean(torch.tensor(accuracy3))
    avg_acc1 = torch.mean(torch.tensor(accuracy1))
    avg_precision3 = np.mean(precision3)
    avg_precision1 = np.mean(precision1)
    avg_recall3 = np.mean(recall3)
    avg_recall1 = np.mean(recall1)
    std_acc3 = torch.std(torch.tensor(accuracy3))
    std_acc1 = torch.std(torch.tensor(accuracy1))
    std_precision3 = np.std(precision3)
    std_precision1 = np.std(precision1)
    std_recall3 = np.std(recall3)
    std_recall1 = np.std(recall1)

    print('='*50)
    print(f'Final accuracy and standard deviation over all runs (3-shot): {avg_acc3:.4f} ± {std_acc3:.4f}')
    print(f'Final accuracy and standard deviation over all runs (1-shot): {avg_acc1:.4f} ± {std_acc1:.4f}')
    print('='*50)

    results_dict[f'SimpleShot({3},{15})_{save_path}'] = (avg_acc3, std_acc3, avg_precision3, std_precision3, avg_recall3, std_recall3)
    results_dict[f'SimpleShot({1},{15})_{save_path}'] = (avg_acc1, std_acc1, avg_precision1, std_precision1, avg_recall1, std_recall1)

    ################################################################################
    ################################################################################

    # save the results dictionary as csv dataframe
    results_df = pd.DataFrame.from_dict(results_dict, orient='index', columns=[
        'avg_acc', 'std_dev_acc', 'avg_precision', 'std_dev_precision', 'avg_recall', 'std_dev_recall',])

    save_to = save_folder + '/' + 'SimpleShot_results_k_folds.csv'
    results_df.to_csv(save_to)
