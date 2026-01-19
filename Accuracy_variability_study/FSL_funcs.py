import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
import math
import time
import os

from .. import FSL_models as FSLm

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
from sklearn.model_selection import train_test_split 

from typing import List, Dict, Any, Tuple
from collections import defaultdict
import random


class CustomDataset(Dataset):
    def __init__(self, images, labels, transformations=None):
        self.images = images
        self.labels = labels
        self.transformations = transformations

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        # apply augmentations if wanted (only for training)
        if self.transformations!=None:
            image = self.transformations(image)
        return image, label

class rotation(object):
    '''
    Rotates an image by a random angle from the given list of angles
    '''
    def __init__(self, angles = [90,180,270,360]):
        self.angles = angles

    def __call__(self, img):
        angle = random.choice(self.angles)
        return F.rotate(img, angle)


class STM_bright_features(Dataset):

    def __init__(self, images, labels, res,
            features: List[str] = None, training_set = False, episodic=True):# a list of features in the dataset e.g. dangling bond
            # sizes: float = 1.0, # this is another thing that could be fed into it to help distinguish different features
                                 # since they're not all of the same size. For now we make all of them 11 pixels big
            # we could include other meta data in here too
        # )

        self.all_features = ['single dangling bond', 'double dangling bond' ,
                             'As A' , 'As B' , 'dimer vacancy', 'single DV on Si(001)', 'siloxane',
                             'C defect', 'single dihydride', 'single dangling bond inv',
                             'double dangling bond inv' , 'As A inv' , 'As B inv', 'dimer vacancy inv',
                             'single DV on Si(001) inv', 'siloxane inv', 'C defect inv',
                             'single dihydride inv', 'h1', 'h2', 't1', 'g1', 'm1', 'h1 inv',
                             'h2 inv', 't1 inv', 'g1 inv', 'm1 inv',
                             'TiO2_vacancy', 'TiO2_hydroxyl', 'TiO2_vacancy inv', 'TiO2_hydroxyl inv']

        self.features = features
        #self.sizes = sizes
       # print(self.features)
        self.images = images
        self.labels = labels

        self.episodic = episodic
     
        self.res = res
        # all crops should be of size (self.res,self.res)
        self.transform = transforms.Resize((self.res, self.res))

        # select only the images that are given in the features list
        self.refine_images()

        # if it's a training set, we will perform some augmentations
        self.training_set = training_set
        self.transformations = transforms.Compose([rotation([90,180,270,360]),
                                      RandomHorizontalFlip()] )


    @property
    def classlist(self) -> List[str]: # returns a list of strings
        return self.features

    @property
    def class_to_indices(self) -> Dict[str, List[int]]:
        # takes in a string describing the class, returns a dictionary with the class
        # string as its key and a list with the indices of that class as its value
        if not hasattr(self, "_class_to_indices"):
            self._class_to_indices = defaultdict(list)
            for i, image in enumerate(self.images):
                self._class_to_indices[self.features[int(self.labels[i])]].append(i)
          
        return self._class_to_indices

    def refine_images(self):
        # picks out the images (and their corresponding labels) according to what was given
        # in the features list
        fin_indices = [] # list that will contain the indices of the features we want in this dataset
        for feature in self.all_features:
            if feature in self.features:
                fin_indices.append(self.all_features.index(feature))
        #        print(self.all_features.index(feature), feature, 'in feature')
        #    else:
        #        print(self.all_features.index(feature), feature, 'not in feature')

        
        # images is of shape (num_samples, channels, res, res), labels is of shape (num_samples)
        fin_images = []
        fin_labels = []
        # the labels atm are from 0 to len(all_features)-1. If we have a dataset consisting of
        # a list less than all_features, then we need to reassign the labels so they go from
        # 0 to len(features)-1.
        for idx in fin_indices:
            # num of samples in this class
            num_samples_class = self.labels[self.labels==idx].shape
            # give this a new y_true label, based off its index in the features list
            new_y = self.features.index(self.all_features[idx])
            fin_labels.append(new_y*torch.ones(num_samples_class))
            fin_images.append(self.images[self.labels==idx,:,:,:])

        self.images = torch.vstack(fin_images)
        self.labels = torch.hstack(fin_labels)

        return

    def __len__(self):
        return len(self.images)


    def __getitem__(self, idx) -> Dict:
        # takes in an index and returns a dictionary in the form
        # data = {'label' = y_value, 'image' = stm crop}

        data = {}

        data['label'] = self.labels[idx]

        data['image'] = self.images[idx]

        if data['image'].shape != (self.res,self.res):
          data['image'] = self.transform(data['image'])

        if self.training_set:
          data['image'] = self.transformations(data['image'])

        # make mean 0 for each channel
        data['image'] = data['image'] - torch.mean(data['image'], dim=(0,1), keepdim=True)

        if self.episodic:
            return data
        else:
            return data['image'], data['label']

    def split(self, test_size=0.3, random_state=42, episodic=False) -> Tuple['STM_bright_features', 'STM_bright_features']:
        """
        Splits the current dataset into training and validation datasets.
        
        Args:
            test_size (float): Proportion of the dataset to include in the validation split.
            random_state (int): Random seed for reproducibility.
            
        Returns:
            train_dataset (STM_bright_features): The training split.
            val_dataset (STM_bright_features): The validation split.
        """
        # Use sklearn's train_test_split to split indices or data directly
        # We split the refined images and labels
        X_train, X_val, y_train, y_val = train_test_split(
            self.images, 
            self.labels, 
            test_size=test_size, 
            random_state=random_state,
            stratify=self.labels # Ensure class distribution is preserved
        )
        
        # Create new dataset instances
        # Note: We pass the already refined features list, so the new datasets 
        # will inherit the specific features of this parent dataset.
        # We set training_set=True for the train split, and False for validation.
        
        train_dataset = STM_bright_features(
            images=X_train, 
            labels=y_train, 
            res=self.res, 
            features=self.features, 
            training_set=True,
            episodic=episodic
        )
        
        val_dataset = STM_bright_features(
            images=X_val, 
            labels=y_val, 
            res=self.res, 
            features=self.features, 
            training_set=False,
            episodic=episodic
        )
        
        return train_dataset, val_dataset



class EpisodeDataset(Dataset):

    def __init__(self,
        dataset,
        n_way = 4, # The number of classes to sample per episode.
        n_support = 3, # The number of samples per class to use as support.
        n_query = 20, # The number of samples per class to use as query.
        n_episodes = 100, # The number of episodes to generate.
    ):
        self.dataset = dataset

        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query
        self.n_episodes = n_episodes

    def __getitem__(self, index:int) -> Tuple[Dict,dict]:
        '''
        This method returns an episode from the dataset.
        Each episode consists of a support set and a query set, 
        each set being a dictionary.
        '''
        # seed a random sampler so the index always returns the same episode.
        rng = random.Random(index)

        # pick out n_way classes for this episode
        episode_classlist = rng.sample(self.dataset.classlist, self.n_way)
        #print(episode_classlist)

        support, query = [], []
        for c in episode_classlist:
            # go through each class and make up the support and query datasets
           # print(c)
            # dataset indices for this class
            all_indices = self.dataset.class_to_indices[c]
           # print(len(all_indices))
            # sample the support and query sets for this class
            indices = rng.sample(all_indices, self.n_support + self.n_query)
            items = [self.dataset[i] for i in indices] # this will be a list of dictionaries

            # we define a new label, or target, for each class for this episode and assign
            # it to the image. This is so it more closely resembles what it will end up
            # doing in the end.
            for item in items:
                item["target"] = torch.tensor(episode_classlist.index(c))

            # split the support and query sets
            support += items[:self.n_support]
            query += items[self.n_support:]

        # now we have 2 lists
        # each item in the list is a dictionary
        # each dictionary is of the form {'label': true y_value, 'image': stm crop, 'target': y_value for this episode'}
        # we want to collate all of these dictionaries so that we have two large dictionaries that can be used easily for batch training
        # i.e they should be of the form
        # support = {'image': numpy array of shape (number of crops, num_channels, res, res),
        #            'target': numpy array of shape (number of crops, y_values for this episode) 
        #            'true_target': numpy array of shape (number of crops, true y_values)}
        # (we don't include the true y_values as they're not needed and this will speed up computation)
        # and similar for the query dict

        # collate the support and query sets
        support = self.collate_dicts(support)
        query = self.collate_dicts(query)

        # add a list of the possible outsomes to the support and query dictionaries
        support["classlist"] = episode_classlist
        query["classlist"] = episode_classlist

        return support, query

    def __len__(self):
        return self.n_episodes

    def episode_info(self, support, query):
        # gives a summary of the episode.

        print("Support Set:")
        print("Classlist: {}".format(support['classlist']) )
        print("Image Shape: {}".format(support['image'].shape) )
        print("Target Shape: {}".format(support['target'].shape) )
        print()
        print("Query Set:")
        print("Classlist: {}".format(query['classlist']) )
        print("Image Shape: {}".format(query['image'].shape) )
        print("Target Shape: {}".format(query['target'].shape) )

    def collate_dicts(self, list_of_dicts):
        images = []
        targets = []
        labels = []
        for item in list_of_dicts:
            images.append(item['image'])
            targets.append(item['target'])
            labels.append(item['label'])

        images = torch.stack(images, dim=0)
        targets = torch.stack(targets, dim=0)
        labels = torch.stack(labels)

        return {'image': images, 'target':targets, 'label': labels}

# Function to save the model
def save_model(model, path):
    torch.save(model.state_dict(), path)

# function to load model
def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model

def train_evaluate(fslNet, file_name, train_loader, val_episodes, n_way, epochs = 200, onehot=True):
  '''
  Function to train and evaluate a few-shot learner.
  Inputs:
    fslNet: the few-shot learning network to be trained
    file_name: the path to save the best model
    train_loader: the DataLoader for the training episodes
    val_episodes: list of validation episodes
    epochs: number of training epochs
    n_way = number of classes per episode
  Outputs:
    learner: the trained few-shot learner
    metric: the accuracy metric used
    last_total_acc: the accuracy on the last validation epoch
    accuracies: list of accuracies across all validation epochs
    std_dev: standard deviation of the accuracies
    confidenceInt: 95% confidence interval of the accuracies
  '''
  # define the FSL
  learner = FSLm.FewShotLearner(fslNet, n_way) 
  print(f'Doing {epochs} epochs of training')

  DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  class EvalCallback(pl.Callback):
      def __init__(self, val_episodes, file_name, onehot):
          self.val_episodes = val_episodes
          self.file_name = file_name
          self.onehot = onehot
          self.best_acc = 0.0
          self.accuracies = []
          self.metric = Accuracy(task = 'multiclass', num_classes=n_way).to(DEVICE)
          self.std_dev = torch.tensor(0.0)
          self.confidenceInt = torch.tensor([0.0])
          self.last_total_acc = 0.0
          self.best_model_state = None

      def on_train_epoch_end(self, trainer, pl_module):
          epoch = trainer.current_epoch
          # evaluate
          pl_module.eval()
          print(f'Validation Epoch {epoch}')

          for idx in range(len(val_episodes)):
            support, query = val_episodes[idx]
            # get the embeddings
            logits = pl_module.FSLnet(query, support)
            if not self.onehot:
              logits = torch.round(logits)
            # compute the accuracy
            acc = self.metric(logits, query["target"].to(DEVICE))
        
          # compute the total accuracy across all episodes
          total_acc = self.metric.compute()
          self.metric.reset()
          self.last_total_acc = total_acc   
          print(f"Epoch accuracy: {total_acc}")
            
          if epoch == 0:
            self.best_acc = total_acc
            self.best_model_state = {k: v.cpu().clone() for k, v in pl_module.state_dict().items()}
            # save
            try:
                save_model(pl_module, self.file_name + '.pth')
                print(f'Saved initial model at epoch 0')
            except Exception as e:
                print(f"Warning: Failed to save model to disk (disk full?): {e}")
          
          print(f"Total accuracy, averaged across all episodes: {total_acc}")
          print(f"Finished epoch: {epoch}")
          
          if total_acc > self.best_acc:
            print(f'New best accuracy achieved (saving model): {total_acc}, previous best: {self.best_acc}')
            self.best_acc = total_acc
            self.best_model_state = {k: v.cpu().clone() for k, v in pl_module.state_dict().items()}
            # save
            try:
                save_model(pl_module, self.file_name + '.pth')
            except Exception as e:
                print(f"Warning: Failed to save model to disk (disk full?): {e}")

          self.accuracies.append(total_acc.cpu().item())
          if len(self.accuracies) > 0:
             self.std_dev = torch.std(torch.tensor(self.accuracies))
             self.confidenceInt = 1.96*self.std_dev/torch.sqrt(torch.tensor([len(self.accuracies)]))
          
          pl_module.train()

  eval_cb = EvalCallback(val_episodes, file_name, onehot)

  trainer = pl.Trainer(accelerator=DEVICE.type, devices = 1, 
                       max_epochs=epochs, callbacks=[eval_cb],
                       enable_checkpointing=False, logger=False, 
                       enable_progress_bar=False)
  trainer.fit(learner, train_loader)


  # load best model
  if eval_cb.best_model_state is not None:
      print("Loading best model from memory...")
      learner.load_state_dict(eval_cb.best_model_state)
  else:
      print("Warning: No best model state found in memory, using last state.")

  return learner, eval_cb.metric, eval_cb.last_total_acc, eval_cb.accuracies, eval_cb.std_dev, eval_cb.confidenceInt

def evaluate(fslNet, save_path, test_episodes, test_features, onehot=True, DEVICE='cpu', n_way=4):
  '''
  Function to evaluate a few-shot learner.
    Inputs:
        fslNet: the few-shot learning network to be evaluated
        save_path: path to save the confusion matrix
        test_episodes: list of test episodes
        test_features: list of features in the test set
    Outputs:
        learner: the evaluated few-shot learner
        confidenceInt: 95% confidence interval of the accuracies
        metric: the accuracy metric used
        total_acc: the accuracy on the test set
        accuracies: list of accuracies across all test episodes
        std_dev: standard deviation of the accuracies
  '''
  learner = fslNet
  # list of accuracies
  accuracies = []

  # evaluate
  learner.eval()
  learner = learner.to(DEVICE)
  # instantiate the accuracy metric
  metric = Accuracy(task = 'multiclass', num_classes=n_way).to(DEVICE)
  all_predicted_labels = []
  all_true_labels = []
  for idx in range(len(test_episodes)):
    support, query = test_episodes[idx]
    # get true labels -> temporary targets mapping
    # support = {'image': numpy array of shape (number of crops, num_channels, res, res),
    #            'target': numpy array of shape (number of crops, y_values for this episode)
    #            'true_target': numpy array of shape (number of crops, true y_values)}
    label_to_target = {target.item(): label.item() for label, target in zip(support['label'], support['target'])}
    # get the embeddings
    logits = learner.FSLnet(query, support)
    #print('logits before: ',logits)
    if not onehot:
      logits = torch.round(logits)
    # compute the accuracy
    acc = metric(logits, query["target"].to(DEVICE))
    accuracies.append(acc)
    if onehot:
        # convert the logits to their true label predictions
        predicted_targets = torch.argmax(logits, dim=1)
    else:
        predicted_targets = logits
    predicted_labels = torch.zeros(predicted_targets.shape)
    for i in range(len(predicted_targets)):
      predicted_labels[i] = label_to_target[predicted_targets[i].item()]
    # append to lists of all_labels
    all_predicted_labels.append(predicted_labels)
    all_true_labels.append(query['label'])

  std_dev = torch.std(torch.tensor(accuracies))
  total_acc = torch.mean(torch.tensor(accuracies))
  confidenceInt = 1.96*std_dev/torch.sqrt(torch.tensor([len(accuracies)])) # 95% confidence interval assumin normal distributions
  print(f"Total accuracy, averaged across all episodes: {total_acc*100} +/- {confidenceInt[0]*100}")

  # make a confusion matrix, and calculate the micro-average precision+recall
  all_predicted_labels = torch.cat(all_predicted_labels).cpu().numpy()
  all_true_labels = torch.cat(all_true_labels).cpu().numpy()
  confusion_matrxi = plot_confusion_matrix(all_true_labels, all_predicted_labels, np.unique(all_true_labels), test_features, save_path)
 
  # get precision and recall from the cm for each class
  precisions = {}
  recalls = {}
  for i in range(confusion_matrxi.shape[0]):
    TP = confusion_matrxi[i,i]
    FP = np.sum(confusion_matrxi[:,i]) - TP
    FN = np.sum(confusion_matrxi[i,:]) - TP
    precision = TP/(TP+FP) if (TP+FP)>0 else 0
    recall = TP/(TP+FN) if (TP+FN)>0 else 0
    print(f'Class {test_features[i]}: Precision: {precision*100:.2f} %, Recall: {recall*100:.2f} %')
    time.sleep(1)
    precisions[i] = precision
    recalls[i] = recall

  average_precision = np.mean( list(precisions.values()) )
  average_recall = np.mean( list(recalls.values()) )

  return learner, confidenceInt[0]*100, metric, total_acc, accuracies, std_dev, confidenceInt, precisions, recalls, average_precision, average_recall


def evaluate_simpleshot(network, save_path, test_episodes, test_features, n_way, onehot=True, device='cpu'):
    # list of accuracies
    accuracies = []

    # evaluate
    network.eval()
    network.to(device)
    # instantiate the accuracy metric
    metric = Accuracy(task = 'multiclass', num_classes=n_way).to(device)
    pbar = tqdm.tqdm(range(len(test_episodes)))
    all_predicted_labels = []
    all_true_labels = []
    for episode_idx in pbar:
        support, query = test_episodes[episode_idx]
        # get true labels -> temporary targets mapping
        # support = {'image': numpy array of shape (number of crops, num_channels, res, res),
        #            'target': numpy array of shape (number of crops, y_values for this episode)
        #            'true_target': numpy array of shape (number of crops, true y_values)}
        support_crops, support_labels = support['image'].to(device), support['target'].to(device)
        query_crops, query_labels = query['image'].to(device), query['target'].to(device)

        label_to_target = {target.item(): label.item() for label, target in zip(support['label'], support['target'])}
        # get the logits/embeddings?
        logits, distances, x_q, x_s_norm = network(query_crops, support_crops, support_labels, n_way)
        if not onehot:
            logits = torch.round(logits)
        # compute the accuracy
        acc = metric(logits, query["target"].to(device))
        pbar.set_description(f"Episode {episode_idx} // Accuracy: {acc.item():.2f}")
        accuracies.append(acc)
        metric.reset()
        if onehot:
            # convert the logits to their true label predictions
            predicted_targets = torch.argmax(logits, dim=1)
        else:
            predicted_targets = logits
        predicted_labels = torch.zeros(predicted_targets.shape)
        for i in range(len(predicted_targets)):
            predicted_labels[i] = label_to_target[predicted_targets[i].item()]
        # append to lists of all_labels
        all_predicted_labels.append(predicted_labels)
        all_true_labels.append(query['label'])

    std_dev = torch.std(torch.tensor(accuracies))
    total_acc = torch.mean(torch.tensor(accuracies))
    confidenceInt = 1.96*std_dev/torch.sqrt(torch.tensor([len(accuracies)])) # 95% confidence interval assumin normal distributions
    print(f"Total accuracy, averaged across all episodes: {total_acc*100} +/- {confidenceInt[0]*100}")

    # make a confusion matrix, and calculate the micro-average precision+recall
    all_predicted_labels = torch.cat(all_predicted_labels).cpu().numpy()
    all_true_labels = torch.cat(all_true_labels).cpu().numpy()
    confusion_matrxi = plot_confusion_matrix(all_true_labels, all_predicted_labels, np.unique(all_true_labels), test_features, save_path)
    
    # get precision and recall from the cm for each class
    precisions = {}
    recalls = {}
    for i in range(confusion_matrxi.shape[0]):
        TP = confusion_matrxi[i,i]
        FP = np.sum(confusion_matrxi[:,i]) - TP
        FN = np.sum(confusion_matrxi[i,:]) - TP
        precision = TP/(TP+FP) if (TP+FP)>0 else 0
        recall = TP/(TP+FN) if (TP+FN)>0 else 0
        print(f'Class {test_features[i]}: Precision: {precision*100:.2f} %, Recall: {recall*100:.2f} %')
        time.sleep(1)
        precisions[i] = precision
        recalls[i] = recall

    average_precision = np.mean( list(precisions.values()) )
    average_recall = np.mean( list(recalls.values()) )

    return network, metric, total_acc, accuracies, std_dev, confidenceInt, precisions, recalls, average_precision, average_recall


def plot_confusion_matrix(y_true, y_pred, labels, label_names, save_path, save_fig=False):
    '''
    Plots and returns the confusion matrix.
    Inputs:
        y_true: true labels
        y_pred: predicted labels
        labels: list of label indices
        label_names: list of label names
        save_path: path to save the confusion matrix figure
        save_fig: whether to save the figure or not
    Outputs:
        cm: confusion matrix
    '''
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    if save_fig:
        fig, ax = plt.subplots(figsize=(10, 10))
        disp.plot(cmap=plt.cm.Blues, ax=ax)
        plt.title('Confusion Matrix')
        plt.savefig(save_path + '_confusion_matrix.png')
    return cm

def set_seed(seed=42, DEVICE='cpu'):
    '''
    Set seeds for reproducibility.
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE == 'cuda':
        torch.cuda.manual_seed_all(seed)
    return

def train_save_match_proto_rel(x_all, y_all, n_way, n_support, n_query, n_train_episodes, n_test_episodes, train_features, val_features, test_features, save_path, DEVICE='cpu', folder='FSL_models'):    
    # initialise datasets
    trainDS = STM_bright_features(x_all, y_all, 40, features = train_features, training_set=True)
    testDS = STM_bright_features(x_all, y_all, 40, features = test_features)
    valDS = STM_bright_features(x_all, y_all, 40, features = val_features)

    # initialise episodic datasets
    trainDSep = EpisodeDataset(trainDS, n_way = n_way,
            n_support = n_support,
            n_query = n_query,
            n_episodes = n_train_episodes)
    valDSep = EpisodeDataset(valDS, n_way = n_way,
            n_support = n_support,
            n_query = n_query,
            n_episodes = n_test_episodes)

    # initialise data loaders
    train_loader = DataLoader(trainDSep, batch_size = None, num_workers = 0)
    val_loader = DataLoader(valDSep, batch_size = None, num_workers = 0)

    n_query2 = 15
    n_episodes2 = 50

    # load our evaluation data
    test_episodes = EpisodeDataset(testDS, n_way = n_way,
            n_support = n_support,
            n_query = n_query2,
            n_episodes = n_test_episodes)

    val_episodes = EpisodeDataset(valDS, n_way = n_way,
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

    print('-'*50)
    print('-'*50)
    print(f'Training Matching Network {n_support}-shot')
    print('test features: ', test_features)
    print('-'*50)
    print('-'*50)
    time.sleep(2)
    learner_matchnet = train_evaluate(match_net, match_net_save_path, train_loader, val_episodes, n_way, epochs=num_epochs, onehot=True)

    print('-'*50)
    print('-'*50)
    print(f'Evaluating Matching Network {n_support}-shot')
    print('test features: ', test_features)
    print('-'*50)
    print('-'*50)
    time.sleep(2)
    eval_match = evaluate(learner_matchnet[0], match_net_save_path, test_episodes, test_features, onehot=True, DEVICE=DEVICE, n_way=n_way)
    match_acc = eval_match[3]
    match_CI = eval_match[1]
    match_precisions = eval_match[7]
    match_recalls = eval_match[8]
    match_prec_avg = eval_match[9]
    match_rec_avg = eval_match[10]

    print('-'*50)
    print('-'*50)
    print(f'Training Relation Network for {n_support}-shot')
    print('test features: ', test_features)
    print('-'*50)
    print('-'*50)
    time.sleep(2)
    learner_relnet = train_evaluate(relation_net, rel_net_save_path, train_loader, val_episodes, n_way, epochs=int(2.5*num_epochs), onehot=True)
    
    print('-'*50)
    print('-'*50)
    print(f'Evaluating Relation Network for {n_support}-shot')
    print('test features: ', test_features)
    print('-'*50)
    print('-'*50)
    time.sleep(2)
    eval_relnet = evaluate(learner_relnet[0], rel_net_save_path, test_episodes, test_features, onehot=True, DEVICE=DEVICE, n_way=n_way)
    rel_acc = eval_relnet[3]
    rel_CI = eval_relnet[1]
    rel_precisions = eval_relnet[7]
    rel_recalls = eval_relnet[8]
    rel_prec_avg = eval_relnet[9]
    rel_rec_avg = eval_relnet[10]

    print('-'*50)
    print('-'*50)
    print(f'Training Prototypical Network {n_support}-shot')
    print('test features: ', test_features)
    print('-'*50)
    print('-'*50)
    time.sleep(2)
    learner_protonet = train_evaluate(prot_net, proto_net_save_path, train_loader, val_episodes, n_way, epochs=num_epochs, onehot=True)
    
    print('-'*50)
    print('-'*50)
    print(f'Evaluating Prototypical Network {n_support}-shot')
    print('test features: ', test_features)
    print('-'*50)
    print('-'*50)
    time.sleep(2)
    eval_proto = evaluate(learner_protonet[0], proto_net_save_path, test_episodes, test_features, onehot=True, DEVICE=DEVICE, n_way=n_way)
    proto_acc = eval_proto[3]
    proto_CI = eval_proto[1]
    proto_precisions = eval_proto[7]
    proto_recalls = eval_proto[8]
    proto_prec_avg = eval_proto[9]
    proto_rec_avg = eval_proto[10]

    return proto_acc, proto_CI, match_acc, match_CI, rel_acc, rel_CI, proto_prec_avg, match_prec_avg, rel_prec_avg, proto_rec_avg, match_rec_avg, rel_rec_avg


# test accuracy function
def simpleshot_testAccuracy(model, dataloader, device='cpu'):
    
    model.eval()
    accuracy = 0.0
    total = 0.0
    
    with torch.no_grad():
        for data in dataloader:
            crops, labels = data
            total += labels.size(0)
            
            crops, labels = crops.to(device), labels.to(device)
            # run the model on the test set to predict labels
            outputs = model(crops.float())
            # the label with the highest probability will be our prediction
            _, predicted = torch.max(outputs.data, 1)
            #_, labels = torch.max(labels.data, 1)
            accuracy += (predicted == labels).sum().item()
    
    # compute the accuracy over all test images
    accuracy = (100 * accuracy / total)
    
    return(accuracy)

def simpleshot_train(model, optimizer, dataloader_train, dataloader_test, loss_, num_epochs, path, device='cpu'):

    best_accuracy = 0
    
    model.train()
    # Iterate over the training data
    for epoch in range(num_epochs):
        running_train_loss = 0.0
        running_test_loss = 0.0
        
        model=model.float()
        
        # train the model
        model.train()
        for i, (crops, labels) in enumerate(dataloader_train):
            # Get the crops and labels
            crops, labels = crops.to(device), labels.to(device)
           
            # Zero the parameter gradients
            optimizer.zero_grad()
            # get prediction
            outputs = model(crops.float())
            loss = loss_(outputs, labels.long())
            running_train_loss += loss.item()
        
            # Backward pass
            loss.backward()
            optimizer.step()

        accuracy = simpleshot_testAccuracy(model, dataloader_train, device)
        print('epoch', epoch, 'train accuracy over whole train set: %d %%' % (accuracy))
    
                
        # get the test accuracy
        model.eval()
        for i, (crops, labels) in enumerate(dataloader_test):
            # Get the crops and labels
            crops, labels = crops.to(device), labels.to(device)
            # get prediction and loss
            pred = model(crops.float())
            loss = loss_(pred, labels.long())
            
            running_test_loss += loss.item()
            
        accuracy = simpleshot_testAccuracy(model, dataloader_test, device)
        print('epoch', epoch, 'test accuracy over whole test set: %d %%' % (accuracy))

        # save the model if the accuracy is the best
        if accuracy > best_accuracy:
            print('Saving model from epoch', epoch)
            save_model(model, path)
            best_accuracy = accuracy
        
        print('Epoch: %d loss: %.3f' % (epoch + 1, running_test_loss / len(dataloader_test)))


def simple_shot_training_eval(x, y, train_val_features, test_features, n_way, substrate, state=42, inits = 50, device='cpu'):
    batch_size = 50
    # Create the datasets
    train_val_data = STM_bright_features(x, y, res=40, features=train_val_features, episodic=False)
    train_data, val_data = train_val_data.split(test_size=0.2, random_state=state)
    test_data = STM_bright_features(x, y, res=40, features = test_features, episodic=True)
    episodic_test_data_3shot = EpisodeDataset(test_data, n_way=n_way, n_support=3, n_query=15, n_episodes=100)
    episodic_test_data_1shot = EpisodeDataset(test_data, n_way=n_way, n_support=1, n_query=15, n_episodes=100)
    # Create the sampler and data loader
    data_loader_train = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    data_loader_val = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    # Compute the sample weights
    # Get the labels from the dataset
    # Note: train_data.labels is a 1D tensor of indices after refine_images
    labels_tensor = train_val_data.labels 
    # Count samples per class
    class_sample_count = torch.bincount(labels_tensor.long())
    # Calculate weights (inverse frequency)
    weight = 1.0 / class_sample_count.float()
    samples_weight = weight.numpy()

    # Define the loss functions, and optimizer
    criterion_weighted = nn.CrossEntropyLoss(weight = torch.tensor(samples_weight).float().to(device)) # weighted loss function
    #criterion = nn.CrossEntropyLoss() # not weighted loss function

    tot_accuracies_list3 = []
    tot_accuracies_list1 = []
    tot_precisions_list3 = []
    tot_precisions_list1 = []
    tot_recalls_list3 = []
    tot_recalls_list1 = []

    for i in range(inits):
        # define the model    
        model = FSLm.NeuralNetwork(crop_size=40, n_outputs=len(train_val_features), fc_layers=2, fc_nodes=100, dropout=0.2).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size= 20)#, gamma=0.3) # lr=0.5*lr after every 5 epochs
        number_epochs = 1

        n_query = 15

        simpleshot_train(model, optimizer, data_loader_train, data_loader_val, criterion_weighted, number_epochs, f'SimpleShotembeddor{substrate}.pth', device=device)

        simple_shot = FSLm.SimpleShot(model)

        # test the simple shot model
        # we need to do it over 100 eps for both 3-shot and 1-shot
        simple_shot, metric3, total_acc3, accuracies3, std_dev3, confidenceInt3, precisions3, recalls3, average_precision3, average_recall3 = evaluate_simpleshot(simple_shot, f'SimpleShotembeddor_{substrate}_(3,{n_query})_eval', episodic_test_data_3shot, test_features, n_way, onehot=False, device = device)
        tot_accuracies_list3.append(total_acc3)
        tot_precisions_list3.append(average_precision3)
        tot_recalls_list3.append(average_recall3)

        simple_shot, metric1, total_acc1, accuracies1, std_dev1, confidenceInt1, precisions1, recalls1, average_precision1, average_recall1 = evaluate_simpleshot(simple_shot, f'SimpleShotembeddor_{substrate}_(3,{n_query})_eval', episodic_test_data_1shot, test_features, n_way, onehot=False, device = device)
        tot_accuracies_list1.append(total_acc1)
        tot_precisions_list1.append(average_precision1)
        tot_recalls_list1.append(average_recall1)
    
    avg_acc3 = torch.mean(torch.tensor(tot_accuracies_list3))
    avg_acc1 = torch.mean(torch.tensor(tot_accuracies_list1))
    avg_precision3 = np.mean(tot_precisions_list3)
    avg_precision1 = np.mean(tot_precisions_list1)
    avg_recall3 = np.mean(tot_recalls_list3)
    avg_recall1 = np.mean(tot_recalls_list1)
    std_acc3 = torch.std(torch.tensor(tot_accuracies_list3))
    std_acc1 = torch.std(torch.tensor(tot_accuracies_list1))
    std_precision3 = np.std(tot_precisions_list3)
    std_precision1 = np.std(tot_precisions_list1)
    std_recall3 = np.std(tot_recalls_list3)
    std_recall1 = np.std(tot_recalls_list1)

    print('='*50)
    print(f'Final averaged results over all runs:')
    print(f'3-shot: Average Accuracy: {avg_acc3*100:.2f} % +/- {std_acc3*100:.2f} %, Average Precision: {avg_precision3*100:.2f} % +/- {std_precision3*100:.2f} %, Average Recall: {avg_recall3*100:.2f} % +/- {std_recall3*100:.2f} %')
    print('='*50)
    print(f'1-shot: Average Accuracy: {avg_acc1*100:.2f} % +/- {std_acc1*100:.2f} %, Average Precision: {avg_precision1*100:.2f} % +/- {std_precision1*100:.2f} %, Average Recall: {avg_recall1*100:.2f} % +/- {std_recall1*100:.2f} %')
    print('='*50)

    return avg_acc3, std_acc3, avg_precision3, std_precision3, avg_recall3, std_recall3, avg_acc1, std_acc1, avg_precision1, std_precision1, avg_recall1, std_recall1

