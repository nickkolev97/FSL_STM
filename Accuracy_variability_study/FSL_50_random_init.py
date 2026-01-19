import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
import math
import time
import os

# pytorch modules
import torch
from torch.utils.data import Dataset, Subset, DataLoader, RandomSampler
#from torchvision.transforms import ToTensor
import torch.nn as nn # nn class our model inherits from
import torch.optim as optim
import torchvision.transforms.functional as F
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from torchvision.transforms.transforms import RandomHorizontalFlip, RandomVerticalFlip
from torchmetrics import Accuracy
import pytorch_lightning as pl
import tqdm
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torchvision import transforms
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import FSL_funcs as FSL
import FSL_models as FSLm

from typing import List, Dict, Any, Tuple
from collections import defaultdict
import random

import argparse

def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Script for FSL training and testing with random initialization.")
    parser.add_argument('--save_folder', type=str, required=True, help="Folder to save the trained model and results.")
    parser.add_argument('--substrate', type=str, required=True, help="Substrate type (e.g., 'Si', 'TiO2').", default='Si', choices=['Si', 'TiO2', 'Ge'])
    return parser.parse_args()

def train_save_match_proto_rel(n_way, n_support, n_query, n_train_episodes, train_features, val_features, test_features, folder,save_path, DEVICE='cpu'):    
    # initialise datasets
    trainDS = FSL.STM_bright_features(x_all, y_all, 40, features = train_features, training_set=True)
    testDS = FSL.STM_bright_features(x_all, y_all, 40, features = test_features)
    valDS = FSL.STM_bright_features(x_all, y_all, 40, features = val_features)

    # initialise episodic datasets
    trainDSep = FSL.EpisodeDataset(trainDS, n_way = n_way,
            n_support = n_support,
            n_query = n_query,
            n_episodes = n_train_episodes)

    # initialise data loaders
    train_loader = DataLoader(trainDSep, batch_size = None, num_workers = 0)

    n_query2 = 15
    n_episodes2 = 50

    # load our evaluation data
    test_episodes = FSL.EpisodeDataset(testDS, n_way = n_way,
            n_support = n_support,
            n_query = n_query2,
            n_episodes = n_test_episodes)

    val_episodes = FSL.EpisodeDataset(valDS, n_way = n_way,
            n_support = n_support,
            n_query = n_query2,
            n_episodes = n_episodes2)

    prot_net = FSLm.PrototypicalNetwork()
    prot_net.to(DEVICE)
    match_net = FSLm.MatchingNetwork()
    match_net.to(DEVICE)
    relation_net = FSLm.RelationNetwork(res=40)
    relation_net.to(DEVICE)

    # create folder if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    proto_net_save_path = folder + '/' + 'FSL_protonet(' + str(n_support) + ',' + str(n_query2) + ')_' + save_path
    match_net_save_path = folder + '/' + 'FSL_matchnet(' + str(n_support) + ',' + str(n_query2) + ')_' + save_path
    rel_net_save_path = folder + '/' + 'FSL_relnet(' + str(n_support) + ',' + str(n_query2) + ')_' + save_path

    num_epochs = 160

    # We want to repeat the training and evaluation for each of the 3 networks
    # 50 times and get an average accuracy and std_dev over the different initialisations

    match_net_accs = []
    match_net_precisions = []
    match_net_recalls = []

    n_inits = 2

    for i in range(n_inits):
        print('-'*50)
        print('-'*50)
        print(f'Training Matching Network {n_support}-shot')
        print('test features: ', test_features)
        print('-'*50)
        print('-'*50)
        learner_matchnet = FSL.train_evaluate(match_net, match_net_save_path, train_loader, val_episodes, epochs=num_epochs, onehot=True)

        print('-'*50)
        print('-'*50)
        print(f'Evaluating Matching Network {n_support}-shot')
        print('test features: ', test_features)
        print('-'*50)
        print('-'*50)
        eval_match = FSL.evaluate(learner_matchnet[0], match_net_save_path, test_episodes, test_features, onehot=True)
        match_acc = eval_match[3]
        match_prec_avg = eval_match[9]
        match_rec_avg = eval_match[10]
        match_net_accs.append(match_acc)
        match_net_precisions.append(match_prec_avg)
        match_net_recalls.append(match_rec_avg)


    match_acc = torch.mean(torch.tensor(match_net_accs))
    match_std_dev = torch.std(torch.tensor(match_net_accs))
    match_CI = 1.96*match_std_dev/torch.sqrt(torch.tensor([len(match_net_accs)])) # 95% confidence interval assuming normal distributions
    match_prec = torch.mean(torch.tensor(match_net_precisions))
    match_prec_std_dev = torch.std(torch.tensor(match_net_precisions))
    match_rec = torch.mean(torch.tensor(match_net_recalls))
    match_rec_std_dev = torch.std(torch.tensor(match_net_recalls))
    print(f'Matching Network {n_support}-shot: Accuracy: {match_acc*100:.2f} +/- {match_CI[0]*100:.2f} %, Precision: {match_prec*100:.2f} %, Recall: {match_rec*100:.2f} %. Std Dev: {match_std_dev*100:.2f} %')
    time.sleep(2)

    rel_net_accs = []
    rel_net_precisions = []
    rel_net_recalls = []

    for i in range(n_inits):
        print('-'*50)
        print('-'*50)
        print(f'Training Relation Network for {n_support}-shot')
        print('test features: ', test_features)
        print('-'*50)
        print('-'*50)
        learner_relnet = FSL.train_evaluate(relation_net, rel_net_save_path, train_loader, val_episodes, epochs=int(2.5*num_epochs), onehot=True)

        print('-'*50)
        print('-'*50)
        print(f'Evaluating Relation Network for {n_support}-shot')
        print('test features: ', test_features)
        print('-'*50)
        print('-'*50)
        eval_relnet = FSL.evaluate(learner_relnet[0], rel_net_save_path, test_episodes, test_features, onehot=True)
        rel_acc = eval_relnet[3]
        rel_prec_avg = eval_relnet[9]
        rel_rec_avg = eval_relnet[10]
        rel_net_accs.append(rel_acc)
        rel_net_precisions.append(rel_prec_avg)
        rel_net_recalls.append(rel_rec_avg)

    rel_acc = torch.mean(torch.tensor(rel_net_accs))
    rel_std_dev = torch.std(torch.tensor(rel_net_accs))
    rel_CI = 1.96*rel_std_dev/torch.sqrt(torch.tensor([len(rel_net_accs)])) # 95% confidence interval assuming normal distributions
    rel_prec = torch.mean(torch.tensor(rel_net_precisions))
    rel_prec_std_dev = torch.std(torch.tensor(rel_net_precisions))
    rel_rec = torch.mean(torch.tensor(rel_net_recalls))
    rel_rec_std_dev = torch.std(torch.tensor(rel_net_recalls))

    print(f'Relation Network {n_support}-shot: Accuracy: {rel_acc*100:.2f} +/- {rel_CI[0]*100:.2f} %, Precision: {rel_prec*100:.2f} %, Recall: {rel_rec*100:.2f} %. Std Dev: {rel_std_dev*100:.2f} %')

    proto_net_accs = []
    proto_net_precisions = []
    proto_net_recalls = []

    for i in range(n_inits):     
        print('-'*50)
        print('-'*50)
        print(f'Training Prototypical Network {n_support}-shot')
        print('test features: ', test_features)
        print('-'*50)
        print('-'*50)
        learner_protonet = FSL.train_evaluate(prot_net, proto_net_save_path, train_loader, val_episodes, epochs=num_epochs, onehot=True)
        
        print('-'*50)
        print('-'*50)
        print(f'Evaluating Prototypical Network {n_support}-shot')
        print('test features: ', test_features)
        print('-'*50)
        print('-'*50)
        eval_proto = FSL.evaluate(learner_protonet[0], proto_net_save_path, test_episodes, test_features, onehot=True)
        proto_acc = eval_proto[3]
        proto_prec_avg = eval_proto[9]
        proto_rec_avg = eval_proto[10]
        proto_net_accs.append(proto_acc)
        proto_net_precisions.append(proto_prec_avg)
        proto_net_recalls.append(proto_rec_avg)
    
    proto_acc = torch.mean(torch.tensor(proto_net_accs))
    proto_std_dev = torch.std(torch.tensor(proto_net_accs))
    proto_CI = 1.96*proto_std_dev/torch.sqrt(torch.tensor([len(proto_net_accs)])) # 95% confidence interval assuming normal distributions
    proto_prec = torch.mean(torch.tensor(proto_net_precisions))
    proto_prec_std_dev = torch.std(torch.tensor(proto_net_precisions))
    proto_rec = torch.mean(torch.tensor(proto_net_recalls))
    proto_rec_std_dev = torch.std(torch.tensor(proto_net_recalls))

    print(f'Prototypical Network {n_support}-shot: Accuracy: {proto_acc*100:.2f} +/- {proto_CI[0]*100:.2f} %, Precision: {proto_prec*100:.2f} %, Recall: {proto_rec*100:.2f} %. Std Dev: {proto_std_dev*100:.2f} %')
    time.sleep(2)        

    return proto_acc*100, proto_std_dev*100, match_acc*100, match_std_dev*100, rel_acc*100, rel_std_dev*100, proto_prec, proto_prec_std_dev, match_prec, match_prec_std_dev, rel_prec, rel_prec_std_dev, proto_rec, proto_rec_std_dev, match_rec, match_rec_std_dev, rel_rec, rel_rec_std_dev


if __name__ == "__main__":
    args = parse_arguments()
    print(f"Saving to directory: {args.save_folder}")
    
    # define device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    cropped_scans = np.load(r'data/defSTM.npy')
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

    x_all = torch.tensor(cropped_scans).float().to(DEVICE)
    y_all = torch.tensor(labels).float().to(DEVICE)

    res = 40 # upsample all small crops to this

    # we don't change these parameters between different experiments
    n_query = 15
    n_train_episodes = 5
    n_test_episodes = 100

    # dictionary to save all the accuracies, precisions, recalls etc
    results_dict = {}

    if args.substrate == 'TiO2':
        ##################
        ##### 3-shot models with the inverse features included, TiO2(110)
        ##################

        n_way= 2
        n_support = 3

        train_features = ['single dangling bond' , 'As A' ,  'C defect', 'siloxane inv', 
                        'single DV on Si(001)', 'C defect inv', 'double dangling bond', 'm1 inv',
                        'single dihydride', 'double dangling bond inv', 'As B inv', 'h1',
                        'h2', 't1 inv', 'g1 inv', 'h2 inv', 'h1 inv', 't1' ]
        val_features = ['single dangling bond inv', 'As A inv' , 'm1', 'g1',  
                        'single dihydride inv', 'As B','single DV on Si(001) inv','siloxane']
        test_features = ['TiO2_vacancy', 'TiO2_hydroxyl']


        save_path = 'TiO2(110)_40pix_inv'

        results = train_save_match_proto_rel(n_way, n_support, n_query, n_train_episodes, train_features, val_features, test_features, args.save_folder, save_path, DEVICE)
        results_dict[f'({n_support},{n_query})_{save_path}'] = results

        

        ##################
        ##### 1-shot models with the inverse features included, TiO2(110)
        ##################

        n_way= 2
        n_support = 1

            
        train_features = ['single dangling bond' , 'As A' ,  'C defect', 'siloxane inv', 
                        'single DV on Si(001)', 'C defect inv', 'double dangling bond', 'm1 inv',
                        'single dihydride', 'double dangling bond inv', 'As B inv', 'h1',
                        'h2', 't1 inv', 'g1 inv', 'h2 inv', 'h1 inv', 't1' ]
        val_features = ['single dangling bond inv', 'As A inv' , 'm1', 'g1',  
                        'single dihydride inv', 'As B','single DV on Si(001) inv','siloxane']
        test_features = ['TiO2_vacancy', 'TiO2_hydroxyl']
        
        save_path = 'TiO2(110)_40pix_inv'

        results = train_save_match_proto_rel(n_way, n_support, n_query, n_train_episodes, train_features, val_features, test_features,  args.save_folder, save_path, DEVICE)
        results_dict[f'({n_support},{n_query})_{save_path}'] = results

    ##################################################################

    if args.substrate == 'Si':
            
        ##################
        ##### 3-shot models with the inverse features included, Si(001)
        ##################

        n_way= 4
        n_support = 3

        train_features = ['single dangling bond' , 'As A' , 'siloxane', 'C defect', 'g1' , 'As A inv' ,
                  'siloxane inv', 'g1 inv', 'C defect inv', 'TiO2_vacancy', 'h1', 'm1', 'h2', 't1', 
                  'TiO2_vacancy inv', 'TiO2_hydroxyl']
        val_features = ['m1 inv', 'single dangling bond inv','TiO2_hydroxyl inv', 'h2 inv', 't1 inv', 'h1 inv']
        test_features = ['double dangling bond', 'As B', 'single dihydride', 'single DV on Si(001)']

        save_path = 'Si(001)_40pix_inv'

        results = train_save_match_proto_rel(n_way, n_support, n_query, n_train_episodes, train_features, val_features, test_features,  args.save_folder, save_path, DEVICE)
        results_dict[f'({n_support},{n_query})_{save_path}'] = results

        ##################
        ##### 1-shot models with the inverse features included, Si(001)
        ##################

        n_way= 4
        n_support = 1

        train_features = ['single dangling bond' , 'As A' , 'siloxane', 'C defect', 'g1' , 'As A inv' ,
                  'siloxane inv', 'g1 inv', 'C defect inv', 'TiO2_vacancy', 'h1', 'm1', 'h2', 't1', 
                  'TiO2_vacancy inv', 'TiO2_hydroxyl']
        val_features = ['m1 inv', 'single dangling bond inv','TiO2_hydroxyl inv', 'h2 inv', 't1 inv', 'h1 inv']
        test_features = ['double dangling bond', 'As B', 'single dihydride', 'single DV on Si(001)']

        save_path = 'Si(001)_40pix_inv'

        results = train_save_match_proto_rel(n_way, n_support,  n_query, n_train_episodes, train_features, val_features, test_features,  args.save_folder, save_path, DEVICE)
        results_dict[f'({n_support},{n_query})_{save_path}'] = results

    if args.substrate == 'Ge':
       
        ##################
        ##### 3-shot models with the inverse features included, Ge(001)
        ##################

        n_way = 4
        n_support = 3

        train_features = ['TiO2_vacancy', 'As A', 'single DV on Si(001)', 'siloxane inv', 'TiO2_hydroxyl inv', 'As A inv', 'siloxane', 'single dihydride', 'single dihydride inv', 'single dangling bond inv', 'C defect', 'As B', 'double dangling bond inv', 'single DV on Si(001) inv']
        val_features = ['As B inv', 'C defect inv', 'TiO2_hydroxyl', 'TiO2_vacancy inv', 'double dangling bond', 'single dangling bond']
        
        test_features = ['t1', 'g1', 'm1', 'h2', 'h1']

        save_path = 'Ge(001)_40pix_inv'

        print('Training features: ', train_features)
        print('Validation features: ', val_features)

        results = train_save_match_proto_rel(n_way, n_support,  n_query, n_train_episodes, train_features, val_features, test_features,  args.save_folder, save_path)
        results_dict[f'({n_support},{n_query})_{save_path}'] = results

        # ################################################################################################
        # ##################
        # ##### 3-shot models with the inverse features included, Ge(001)
        # ##################

        n_way= 4
        n_support = 3

        train_features = ['single dangling bond inv', 'As A inv' , 'C defect', 'siloxane inv', 
                        'single DV on Si(001)', 'C defect inv', 'TiO2_hydroxyl inv', 'double dangling bond', 
                        'single dihydride', 'double dangling bond inv', 'TiO2_vacancy', 'As B inv','TiO2_hydroxyl',
                        'TiO2_vacancy inv','siloxane' ]
        val_features = ['single dangling bond' , 'As A' ,
                        'single dihydride inv', 'As B','single DV on Si(001) inv']
        test_features = ['t1', 'g1', 'm1', 'h2', 'h1']

        save_path = 'Ge(001)_40pix_inv'

        print('Training features: ', train_features)
        print('Validation features: ', val_features)

        results = train_save_match_proto_rel(n_way, n_support,  n_query, n_train_episodes, train_features, val_features, test_features,  args.save_folder, save_path)
        results_dict[f'({n_support},{n_query})_{save_path}'] = results

        ##################

    
    results = train_save_match_proto_rel(n_way, n_support,  n_query, n_train_episodes, train_features, val_features, test_features, args.save_folder, save_path)
    results_dict[f'({n_support},{n_query})_{save_path}'] = results



    # save the results in a pandas dataframe
    results_df = pd.DataFrame.from_dict(results_dict, orient='index', columns=[
        'proto_acc', 'proto_std_dev', 'match_acc', 
        'match_std_dev', 'rel_acc', 'rel_std_dev',
        'proto_prec_avg', 'proto_prec_std_dev', 'match_prec_avg', 
        'match_prec_std_dev', 'rel_prec_avg', 'rel_prec_std_dev',
        'proto_rec_avg', 'proto_rec_std_dev', 'match_rec_avg',
        'match_rec_std_dev', 'rel_rec_avg', 'rel_rec_std_dev' 
    ])

    save_to = args.save_folder + f'/few_shot_results_{args.substrate}_50_inits.csv'
    results_df.to_csv(save_to)