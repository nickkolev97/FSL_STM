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

from typing import List, Dict, Any, Tuple
from collections import defaultdict
import random

import argparse
import FSL_funcs as FSL

def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Script for FSL training and testing with random initialization.")
    parser.add_argument('--save_folder', type=str, required=True, help="Folder to save the trained model and results.")
    parser.add_argument('--substrate', type=str, required=True, help="Substrate type (e.g., 'Si', 'TiO2').", default='Si', choices=['Si', 'TiO2', 'Ge'])
    return parser.parse_args()


# class CustomDataset(Dataset):
#     def __init__(self, images, labels, transformations=None):
#         self.images = images
#         self.labels = labels
#         self.transformations = transformations

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, idx):
#         image = self.images[idx]
#         label = self.labels[idx]
#         # apply augmentations if wanted (only for training)
#         if self.transformations!=None:
#             image = self.transformations(image)
#         return image, label

# class rotation(object):
#     def __init__(self, angles = [90,180,270,360]):
#         self.angles = angles

#     def __call__(self, img):
#         angle = random.choice(self.angles)
#         return F.rotate(img, angle)

# class STM_bright_features(Dataset):

#     def __init__(self, images, labels, res,
#             features: List[str] = None, training_set = False):# a list of features in the dataset e.g. dangling bond
#             # sizes: float = 1.0, # this is another thing that could be fed into it to help distinguish different features
#                                  # since they're not all of the same size. For now we make all of them 11 pixels big
#             # we could include other meta data in here too
#         # )

#         self.all_features = ['single dangling bond', 'double dangling bond' ,
#                              'As A' , 'As B' , 'dimer vacancy', 'single DV on Si(001)', 'siloxane',
#                              'C defect', 'single dihydride', 'single dangling bond inv',
#                              'double dangling bond inv' , 'As A inv' , 'As B inv', 'dimer vacancy inv',
#                              'single DV on Si(001) inv', 'siloxane inv', 'C defect inv',
#                              'single dihydride inv', 'h1', 'h2', 't1', 'g1', 'm1', 'h1 inv',
#                              'h2 inv', 't1 inv', 'g1 inv', 'm1 inv',
#                              'TiO2_vacancy', 'TiO2_hydroxyl', 'TiO2_vacancy inv', 'TiO2_hydroxyl inv']

#         self.features = features
#         #self.sizes = sizes
#        # print(self.features)
#         self.images = images
#         self.labels = labels
     
#         self.res = res
#         # all crops should be of size (self.res,self.res)
#         self.transform = transforms.Resize((self.res, self.res))

#         # select only the images that are given in the features list
#         self.refine_images()

#         # if it's a training set, we will perform some augmentations
#         self.training_set = training_set
#         self.transformations = transforms.Compose([rotation([90,180,270,360]),
#                                       RandomHorizontalFlip()] )


#     @property
#     def classlist(self) -> List[str]: # returns a list of strings
#         return self.features

#     @property
#     def class_to_indices(self) -> Dict[str, List[int]]:
#         # takes in a string describing the class, returns a dictionary with the class
#         # string as its key and a list with the indices of that class as its value
#         if not hasattr(self, "_class_to_indices"):
#             self._class_to_indices = defaultdict(list)
#             for i, image in enumerate(self.images):
#                 self._class_to_indices[self.features[int(self.labels[i])]].append(i)
          
#         return self._class_to_indices

#     def refine_images(self):
#         # picks out the images (and their corresponding labels) according to what was given
#         # in the features list
#         fin_indices = [] # list that will contain the indices of the features we want in this dataset
#         for feature in self.all_features:
#             if feature in self.features:
#                 fin_indices.append(self.all_features.index(feature))
#         #        print(self.all_features.index(feature), feature, 'in feature')
#         #    else:
#         #        print(self.all_features.index(feature), feature, 'not in feature')

        
#         # images is of shape (num_samples, channels, res, res), labels is of shape (num_samples)
#         fin_images = []
#         fin_labels = []
#         # the labels atm are from 0 to len(all_features)-1. If we have a dataset consisting of
#         # a list less than all_features, then we need to reassign the labels so they go from
#         # 0 to len(features)-1.
#         for idx in fin_indices:
#             # num of samples in this class
#             num_samples_class = self.labels[self.labels==idx].shape
#             # give this a new y_true label, based off its index in the features list
#             new_y = self.features.index(self.all_features[idx])
#             fin_labels.append(new_y*torch.ones(num_samples_class))
#             fin_images.append(self.images[self.labels==idx,:,:,:])

#         self.images = torch.vstack(fin_images)
#         self.labels = torch.hstack(fin_labels)

#         return


#     def __len__(self):
#         return len(self.images)


#     def __getitem__(self, idx) -> Dict:
#         # takes in an index and returns a dictionary in the form
#         # data = {'label' = y_value, 'image' = stm crop}

#         data = {}

#         data['label'] = self.labels[idx]

#         data['image'] = self.images[idx]

#         if data['image'].shape != (self.res,self.res):
#           data['image'] = self.transform(data['image'])

#         if self.training_set:
#           data['image'] = self.transformations(data['image'])

#         # make mean 0 for each channel
#         data['image'] = data['image'] - torch.mean(data['image'], dim=(0,1), keepdim=True)

#         return data

# class EpisodeDataset(Dataset):

#     def __init__(self,
#         dataset,
#         n_way = 4, # The number of classes to sample per episode.
#         n_support = 3, # The number of samples per class to use as support.
#         n_query = 20, # The number of samples per class to use as query.
#         n_episodes = 100, # The number of episodes to generate.
#     ):
#         self.dataset = dataset

#         self.n_way = n_way
#         self.n_support = n_support
#         self.n_query = n_query
#         self.n_episodes = n_episodes

#     def __getitem__(self, index:int) -> Tuple[Dict,dict]:
#         # This method returns an episode from the dataset

#         # seed a random sampler so the index always returns the same episode.
#         rng = random.Random(index)

#         # pick out n_way classes for this episode
#         episode_classlist = rng.sample(self.dataset.classlist, self.n_way)
#         #print(episode_classlist)

#         support, query = [], []
#         for c in episode_classlist:
#             # go through each class and make up the support and query datasets
#            # print(c)
#             # dataset indices for this class
#             all_indices = self.dataset.class_to_indices[c]
#            # print(len(all_indices))
#             # sample the support and query sets for this class
#             indices = rng.sample(all_indices, self.n_support + self.n_query)
#             items = [self.dataset[i] for i in indices] # this will be a list of dictionaries

#             # we define a new label, or target, for each class for this episode and assign
#             # it to the image. This is so it more closely resembles what it will end up
#             # doing in the end.
#             for item in items:
#                 item["target"] = torch.tensor(episode_classlist.index(c))

#             # split the support and query sets
#             support += items[:self.n_support]
#             query += items[self.n_support:]

#         # now we have 2 lists
#         # each item in the list is a dictionary
#         # each dictionary is of the form {'label': true y_value, 'image': stm crop, 'target': y_value for this episode'}
#         # we want to collate all of these dictionaries so that we have two large dictionaries that can be used easily for batch training
#         # i.e they should be of the form
#         # support = {'image': numpy array of shape (number of crops, num_channels, res, res),
#         #            'target': numpy array of shape (number of crops, y_values for this episode) 
#         #            'true_target': numpy array of shape (number of crops, true y_values)}
#         # (we don't include the true y_values as they're not needed and this will speed up computation)
#         # and similar for the query dict

#         # collate the support and query sets
#         support = self.collate_dicts(support)
#         query = self.collate_dicts(query)

#         # add a list of the possible outsomes to the support and query dictionaries
#         support["classlist"] = episode_classlist
#         query["classlist"] = episode_classlist

#         return support, query

#     def __len__(self):
#         return self.n_episodes

#     def episode_info(self, support, query):
#         # gives a summary of the episode.

#         print("Support Set:")
#         print("Classlist: {}".format(support['classlist']) )
#         print("Image Shape: {}".format(support['image'].shape) )
#         print("Target Shape: {}".format(support['target'].shape) )
#         print()
#         print("Query Set:")
#         print("Classlist: {}".format(query['classlist']) )
#         print("Image Shape: {}".format(query['image'].shape) )
#         print("Target Shape: {}".format(query['target'].shape) )

#     def collate_dicts(self, list_of_dicts):
#         images = []
#         targets = []
#         labels = []
#         for item in list_of_dicts:
#             images.append(item['image'])
#             targets.append(item['target'])
#             labels.append(item['label'])

#         images = torch.stack(images, dim=0)
#         targets = torch.stack(targets, dim=0)
#         labels = torch.stack(labels)

#         return {'image': images, 'target':targets, 'label': labels}

# def init_layer(L):
#     # Initialization using fan-in
#     if isinstance(L, nn.Conv2d):
#         n = L.kernel_size[0]*L.kernel_size[1]*L.out_channels
#         L.weight.data.normal_(0,math.sqrt(2.0/float(n)))
#     elif isinstance(L, nn.BatchNorm2d):
#         L.weight.data.fill_(1)
#         L.bias.data.fill_(0)

# class ConvBlock(nn.Module):
#     def __init__(self, indim, outdim, pool = True, padding = 1):
#         super(ConvBlock, self).__init__()
#         self.indim  = indim
#         self.outdim = outdim
#         self.C      = nn.Conv2d(indim, outdim, 3, padding= padding)
#         self.BN     = nn.BatchNorm2d(outdim)
#         self.relu   = nn.ReLU(inplace=True)

#         self.parametrized_layers = [self.C, self.BN, self.relu]

#         self.pool   = nn.MaxPool2d(2)
#         self.parametrized_layers.append(self.pool)

#         for layer in self.parametrized_layers:
#             init_layer(layer)

#         self.trunk = nn.Sequential(*self.parametrized_layers)


#     def forward(self,x):
#         out = self.trunk(x)
#         return out

# class ConvNet(nn.Module):
#     def __init__(self):
#         super(ConvNet,self).__init__()
#         trunk = []
#         for i in range(4):
#             indim = 2 if i == 0 else 64
#             outdim = 64
#             B = ConvBlock(indim, outdim, pool = True)
#             trunk.append(B)
#         trunk.append(nn.Flatten())
        

#         self.trunk = nn.Sequential(*trunk)


#     def forward(self,x):
#         out = self.trunk(x)
        
#         return out

class ResNet18Embedding(nn.Module):
    def __init__(self, input_channels=2):
        super().__init__()        
        # Load pretrained model
        self.resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)


        # 0. Input Normalization
        # Since we can't use ImageNet stats for 2-channel STM data, we use a 
        # BatchNorm layer to learn the normalization parameters (mean/std) from the data.
        self.input_norm = nn.BatchNorm2d(input_channels)

        # 1. Handle Input Channels
        # ConvNet uses indim=2. ResNet expects 3. We modify the first layer.
        if input_channels != 3:
            old_conv = self.resnet.conv1
            self.resnet.conv1 = nn.Conv2d(
                input_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias
            )

        # 2. Remove Classification Head to get embeddings (512 dim)
        self.resnet.fc = nn.Identity()

        # 3. Freeze all weights
        for param in self.resnet.parameters():
            param.requires_grad = False

        # 4. Unfreeze the last few layers for fine-tuning
        # Unfreeze the modified input layer
        if input_channels != 3:
            for param in self.resnet.conv1.parameters():
                param.requires_grad = True
        
        # Unfreeze the input normalization layer so it can learn
        for param in self.input_norm.parameters():
            param.requires_grad = True
        
        # Unfreeze the last residual block (layer4)
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True

    def forward(self, x):
        # Upsample to 224x224 as ResNet was trained on this resolution
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Normalize input
        x = self.input_norm(x)
        
        return self.resnet(x)

def get_resnet18_embedding_layer():
    """
    Returns a pretrained ResNet18 embedding layer with frozen weights 
    (except the last block) adapted for 2-channel input.
    """
    return ResNet18Embedding(input_channels=2)

# # class MatchingNetwork(nn.Module):

# #     def __init__(self):
# #         super().__init__()
# #         # define the embedding layer

# #         self.embedding_layer = ConvNet()

# #        # self.embedding_layer = EmbeddingNetwork(channels, crop_size)
# #         self.cos_dist = nn.CosineSimilarity(dim=1)
# #         self.softmax = nn.Softmax(dim=0)

# #     def forward(self, query, support):
# #         # compute embeddings for query and support sets
# #         support["embeddings"] = self.embedding_layer(support["image"]) # f(x)
# #        # print(support['embeddings'].shape)
# #         query["embeddings"] = self.embedding_layer(query["image"]) # g(x_i), for us g = f


# #         # compute the cosine distances between the query embeddings and the support
# #         # query['embeddings'] is a tensor of shape (n_samples, dimensions of embedding vector space)
# #         cos_distances = []
# #         for embedding in support["embeddings"]:
# #           cos_distances.append(torch.exp(self.cos_dist(query["embeddings"], embedding)))
# #          # cos_distances.append(torch.cdist(query["embeddings"].unsqueeze(0), embedding.unsqueeze(0), p=2).squeeze(0)) # c(f(x), g(x_i))
# #         '''
# #         # support["prototypes"] is a tensor of shape
# #         # (n_way, dimensions of embedding vector space)

# #         # compute the distances between the query embeddings and the prototypes
# #         # query['embeddings'] is a tensor of shape (n_samples, dimensions of embedding vector space)
# #         distances = torch.cdist(query["embeddings"].unsqueeze(0), prototypes.unsqueeze(0), p=2).squeeze(0)
# #         '''

# #         cos_distances = torch.stack(cos_distances).squeeze(-1) # c(f(x),g(x_i))
# #         # cos_distances is of shape (n_support, n_query). We have a cosine distance vector between each
# #         # of the support embeddings and the query embeddings and then we take the exponential of it.
# #         attentions = self.softmax(cos_distances)

# #         support["attentions"] = attentions # a = e^c(f(x),g(x_i))/sum_(j=1)^k e^c(f(x),g(x_j))

# #         # output using integer labels
# #       #  y = torch.matmul(support["target"].float().to(DEVICE), support["attentions"]).float()

# #         # output using one hot encoding for targets (got better accuracy)
# #         y = torch.matmul( support["attentions"].T, torch.nn.functional.one_hot(support["target"]).float().to(DEVICE) )
# #         # NOTE:FIGURE OUT THE SHAPE OF THIS??
# #         # PRINT(Y.SHAPE)

# #         # the final predictions should be (where we use einstein summation convention):
# #         # y = a(x,x_i)y_i. With a(x,x_i) = e^{c(f(x),g(x_i))}/sum_{j=1}^{k}e^{c(f(x),g(x_j))}

# #         return y

# # class RelationNetwork(nn.Module):

# #     def __init__(self, res):
# #         super().__init__()
# #         # define the embedding layer
# #         self.embedding_layer = ConvNet()

# #         self.fc_nodes = 100

# #         # the embedding vectors are of size
# #         if res==20:
# #           start = 128
# #         elif res==40:
# #           start = 512 # larger embed dim

# #         self.relation_module = nn.Sequential(
# #                                nn.Linear(start, self.fc_nodes),
# #                                nn.Dropout(0.2),
# #                                nn.ReLU(),
# #                                nn.BatchNorm1d(self.fc_nodes),
# #                                nn.Linear(self.fc_nodes, self.fc_nodes),
# #                                nn.Dropout(0.2),
# #                                nn.ReLU(),
# #                                nn.BatchNorm1d(self.fc_nodes),
# #                                nn.Linear(self.fc_nodes, 1),
# #                                nn.Dropout(0.2),
# #                                nn.ReLU(),
# #                         )

# #     def forward(self, query, support):
# #         # compute embeddings for query and support sets
# #         # input is a (num_channels, res, res)
# #         support["embeddings"] = self.embedding_layer(support["image"]) # f(x)
# #         query["embeddings"] = self.embedding_layer(query["image"]) # g(x_i), for us g = f

# #         # sum up the embeddings of the support vectors in the same class
# #         support_embeds = []
# #         for idx in range(len(support["classlist"])):
# #             embeds = support["embeddings"][support["target"] == idx]
# #             support_embeds.append(embeds)

# #         # support_embeds is a list of torch tensors of shape
# #         # (n_support, dimensions of embedding vector space)

# #         support_embeds = torch.stack(support_embeds)
# #         # support_embeds now a tensor of shape
# #         # (n_way, n_support, dimensions of embedding vector space)

# #         # we compute the sums of these support vectors
# #         # sums has shape (n_way, dimensions of embedding vector)
# #         sums = support_embeds.sum(dim=1)
# #         support["sums"] = sums/torch.sum(sums)

# #         relation_scores = {}
# #         for qvector in query['embeddings']:
# #             # qvector.shape = (dim_emb)
# #             relation_scores[qvector] = []
# #             concats = []
# #             for svector in sums:
# #                 # svector.shape = (dim_emb)
# #                 concat = torch.cat((qvector,svector))
# #                 # concat.shape = (2*dim_emb)
# #                 concats.append(concat)
# #             relation_scores[qvector] = self.relation_module(torch.stack(concats)).squeeze(1)
# #             # relation_scores[qvector].shape = (n_way)

# #         # relation_scores is a dictionary that has the query vectors as keys and their relation scores as values
# #         fin_rel_scores = torch.stack([rel_score for rel_score in relation_scores.values()])
# #         # fin_rel_scores.shape = (n_way*n_query, n_way)

# #         return fin_rel_scores

# # class PrototypicalNetwork(nn.Module):

# #     def __init__(self, embedding_layer = ConvNet()):
# #         super().__init__()
# #         # define the embedding layer

# #         self.embedding_layer = embedding_layer

# #     def forward(self, query, support):
# #         # compute embeddings for query and support sets
# #         support["embeddings"] = self.embedding_layer(support["image"])
# #         query["embeddings"] = self.embedding_layer(query["image"])

# #         # now we need to compute the prototype for each class
# #         # this was the 'average' class member
# #         support_embeds = []
# #         for idx in range(len(support["classlist"])):
# #             embeds = support["embeddings"][support["target"] == idx]
# #             support_embeds.append(embeds)
# #         # support_embeds is a list of torch tensors of shape
# #         # (n_support, dimensions of embedding vector space)

# #         support_embeds = torch.stack(support_embeds)
# #         # support_embeds now a tensor of shape
# #         # (n_way, n_support, dimensions of embedding vector space)

# #         # we compute the mean of these support vectors to get prototypes
# #         prototypes = support_embeds.mean(dim=1)
# #         support["prototypes"] = prototypes

# #         # support["prototypes"] is a tensor of shape
# #         # (n_way, dimensions of embedding vector space)

# #         # compute the distances between the query embeddings and the prototypes
# #         # query['embeddings'] is a tensor of shape (n_samples, dimensions of embedding vector space)
# #         distances = torch.cdist(query["embeddings"].unsqueeze(0), prototypes.unsqueeze(0), p=2).squeeze(0)
# #         # distances is a tensor of dimensions (n_samples, n_ways)
# #         distances = distances ** 2

# #         # the negative of the distances give the final output logits
# #         logits = - distances

# #         return logits

# # class FewShotLearner(pl.LightningModule):

# #     def __init__(self,
# #         FSLnet: nn.Module,
# #         learning_rate: float = 0.001,
# #     ):
# #         super().__init__()
# #         self.save_hyperparameters(ignore=['FSLnet'])
# #         self.FSLnet = FSLnet
# #         self.learning_rate = learning_rate

# #         self.loss = nn.CrossEntropyLoss()
# #         self.metrics = nn.ModuleDict({
# #             'accuracy': Accuracy(task="multiclass", num_classes=n_way)
# #         })

# #     def configure_optimizers(self):
# #         optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
# #         return optimizer

# #     def step(self, batch, batch_idx, tag: str):
# #         support, query = batch

# #         logits = self.FSLnet(query, support)
# #         loss = self.loss(logits, query["target"])

# #         output = {"loss": loss}
# #         for k, metric in self.metrics.items():
# #             output[k] = metric(logits, query["target"])

# #         for k, v in output.items():
# #             self.log(f"{k}/{tag}", v)
# #         return output

# #     def training_step(self, batch, batch_idx):
# #         return self.step(batch, batch_idx, "train")

# #     def validation_step(self, batch, batch_idx):
# #         return self.step(batch, batch_idx, "val")

# #     def test_step(self, batch, batch_idx):
# #         return self.step(batch, batch_idx, "test")

# # # Function to save the model
# # def save_model(model, path):
# #     torch.save(model.state_dict(), path)

# # # function to load model
# # def load_model(model, path):
# #     model.load_state_dict(torch.load(path))
# #     return model

# def train_evaluate(fslNet, file_name, train_loader, val_episodes, epochs = 200, onehot=True):
#   # define the FSL
#   learner = FewShotLearner(fslNet) 
#   print(f'Doing {epochs} epochs of training')
  
#   class EvalCallback(pl.Callback):
#       def __init__(self, val_episodes, file_name, onehot):
#           self.val_episodes = val_episodes
#           self.file_name = file_name
#           self.onehot = onehot
#           self.best_acc = 0.0
#           self.accuracies = []
#           self.metric = Accuracy(task = 'multiclass', num_classes=n_way).to(DEVICE)
#           self.std_dev = torch.tensor(0.0)
#           self.confidenceInt = torch.tensor([0.0])
#           self.last_total_acc = 0.0
#           self.best_model_state = None

#       def on_train_epoch_end(self, trainer, pl_module):
#           epoch = trainer.current_epoch
#           # evaluate
#           pl_module.eval()
#           print(f'Validation Epoch {epoch}')

#           for idx in range(len(val_episodes)):
#             support, query = val_episodes[idx]
#             # get the embeddings
#             logits = pl_module.FSLnet(query, support)
#             if not self.onehot:
#               logits = torch.round(logits)

#             # compute the accuracy
#             acc = self.metric(logits, query["target"].to(DEVICE))
        
#           # compute the total accuracy across all episodes
#           total_acc = self.metric.compute()
#           self.metric.reset()
#           self.last_total_acc = total_acc   
#           print(f"Epoch accuracy: {total_acc}")
            
#           if epoch == 0:
#             self.best_acc = total_acc
#             self.best_model_state = {k: v.cpu().clone() for k, v in pl_module.state_dict().items()}
#             # save
#             try:
#                 save_model(pl_module, self.file_name + '.pth')
#                 print(f'Saved initial model at epoch 0')
#             except Exception as e:
#                 print(f"Warning: Failed to save model to disk (disk full?): {e}")
          
#           print(f"Total accuracy, averaged across all episodes: {total_acc}")
#           print(f"Finished epoch: {epoch}")
          
#           if total_acc > self.best_acc:
#             print(f'New best accuracy achieved (saving model): {total_acc}, previous best: {self.best_acc}')
#             self.best_acc = total_acc
#             self.best_model_state = {k: v.cpu().clone() for k, v in pl_module.state_dict().items()}
#             # save
#             try:
#                 save_model(pl_module, self.file_name + '.pth')
#             except Exception as e:
#                 print(f"Warning: Failed to save model to disk (disk full?): {e}")

#           self.accuracies.append(total_acc.cpu().item())
#           if len(self.accuracies) > 0:
#              self.std_dev = torch.std(torch.tensor(self.accuracies))
#              self.confidenceInt = 1.96*self.std_dev/torch.sqrt(torch.tensor([len(self.accuracies)]))
          
#           pl_module.train()

#   eval_cb = EvalCallback(val_episodes, file_name, onehot)

#   trainer = pl.Trainer(accelerator=DEVICE, devices = 1, 
#                        max_epochs=epochs, callbacks=[eval_cb],
#                        enable_checkpointing=False, logger=False, 
#                        enable_progress_bar=False)
#   trainer.fit(learner, train_loader)


#   # load best model
#   if eval_cb.best_model_state is not None:
#       print("Loading best model from memory...")
#       learner.load_state_dict(eval_cb.best_model_state)
#   else:
#       print("Warning: No best model state found in memory, using last state.")

#   return learner, eval_cb.metric, eval_cb.last_total_acc, eval_cb.accuracies, eval_cb.std_dev, eval_cb.confidenceInt

# def evaluate(fslNet, save_path, test_episodes, test_features, onehot=True):
#   learner = fslNet
#   # list of accuracies
#   accuracies = []

#   # evaluate
#   learner.eval()
#   learner = learner.to(DEVICE)
#   # instantiate the accuracy metric
#   metric = Accuracy(task = 'multiclass', num_classes=n_way).to(DEVICE)
#   all_predicted_labels = []
#   all_true_labels = []
#   for idx in range(len(test_episodes)):
#     support, query = test_episodes[idx]
#     # get true labels -> temporary targets mapping
#     # support = {'image': numpy array of shape (number of crops, num_channels, res, res),
#     #            'target': numpy array of shape (number of crops, y_values for this episode)
#     #            'true_target': numpy array of shape (number of crops, true y_values)}
#     label_to_target = {target.item(): label.item() for label, target in zip(support['label'], support['target'])}
#     # get the embeddings
#     logits = learner.FSLnet(query, support)
#     #print('logits before: ',logits)
#     if not onehot:
#       logits = torch.round(logits)
#     # compute the accuracy
#     acc = metric(logits, query["target"].to(DEVICE))
#     accuracies.append(acc)
#     if onehot:
#         # convert the logits to their true label predictions
#         predicted_targets = torch.argmax(logits, dim=1)
#     else:
#         predicted_targets = logits
#     predicted_labels = torch.zeros(predicted_targets.shape)
#     for i in range(len(predicted_targets)):
#       predicted_labels[i] = label_to_target[predicted_targets[i].item()]
#     # append to lists of all_labels
#     all_predicted_labels.append(predicted_labels)
#     all_true_labels.append(query['label'])

#   std_dev = torch.std(torch.tensor(accuracies))
#   total_acc = torch.mean(torch.tensor(accuracies))
#   confidenceInt = 1.96*std_dev/torch.sqrt(torch.tensor([len(accuracies)])) # 95% confidence interval assumin normal distributions
#   print(f"Total accuracy, averaged across all episodes: {total_acc*100} +/- {confidenceInt[0]*100}")

#   # make a confusion matrix, and calculate the micro-average precision+recall
#   all_predicted_labels = torch.cat(all_predicted_labels).cpu().numpy()
#   all_true_labels = torch.cat(all_true_labels).cpu().numpy()
#   confusion_matrxi = plot_confusion_matrix(all_true_labels, all_predicted_labels, np.unique(all_true_labels), test_features, save_path)
 
#   # get precision and recall from the cm for each class
#   precisions = {}
#   recalls = {}
#   for i in range(confusion_matrxi.shape[0]):
#     TP = confusion_matrxi[i,i]
#     FP = np.sum(confusion_matrxi[:,i]) - TP
#     FN = np.sum(confusion_matrxi[i,:]) - TP
#     precision = TP/(TP+FP) if (TP+FP)>0 else 0
#     recall = TP/(TP+FN) if (TP+FN)>0 else 0
#     print(f'Class {test_features[i]}: Precision: {precision*100:.2f} %, Recall: {recall*100:.2f} %')
#     time.sleep(1)
#     precisions[i] = precision
#     recalls[i] = recall

#   average_precision = np.mean( list(precisions.values()) )
#   average_recall = np.mean( list(recalls.values()) )

#   return learner, confidenceInt[0]*100, metric, total_acc, accuracies, std_dev, confidenceInt, precisions, recalls, average_precision, average_recall

# def plot_confusion_matrix(y_true, y_pred, labels, label_names, save_path, save_fig=False):
#     cm = confusion_matrix(y_true, y_pred, labels=labels)
#     disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
#     if save_fig:
#         fig, ax = plt.subplots(figsize=(10, 10))
#         disp.plot(cmap=plt.cm.Blues, ax=ax)
#         plt.title('Confusion Matrix')
#         plt.savefig(save_path + '_confusion_matrix.png')
#     return cm

# def set_seed(seed=42, DEVICE='cpu'):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if DEVICE == 'cuda':
#         torch.cuda.manual_seed_all(seed)
#     return

# def train_save_match_proto_rel(n_way, n_support, n_query, n_train_episodes, train_features, val_features, test_features, folder,save_path):    
#     # initialise datasets
#     trainDS = FSL.STM_bright_features(x_all, y_all, 40, features = train_features, training_set=True)
#     testDS = FSL.STM_bright_features(x_all, y_all, 40, features = test_features)
#     valDS = FSL.STM_bright_features(x_all, y_all, 40, features = val_features)

#     # initialise episodic datasets
#     trainDSep = FSL.EpisodeDataset(trainDS, n_way = n_way,
#             n_support = n_support,
#             n_query = n_query,
#             n_episodes = n_train_episodes)

#     # initialise data loaders
#     train_loader = DataLoader(trainDSep, batch_size = None, num_workers = 0)

#     n_query2 = 15
#     n_episodes2 = 50

#     # load our evaluation data
#     test_episodes = FSL.EpisodeDataset(testDS, n_way = n_way,
#             n_support = n_support,
#             n_query = n_query2,
#             n_episodes = n_test_episodes)

#     val_episodes = FSL.EpisodeDataset(valDS, n_way = n_way,
#             n_support = n_support,
#             n_query = n_query2,
#             n_episodes = n_episodes2)

#     FSL.set_seed(DEVICE = DEVICE)
#     prot_net = FSL.PrototypicalNetwork()
#     prot_net.to(DEVICE)
#     match_net = FSL.MatchingNetwork()
#     match_net.to(DEVICE)
#     relation_net = FSL.RelationNetwork(res=40)
#     relation_net.to(DEVICE)

#     # create folder if it doesn't exist
#     if not os.path.exists(folder):
#         os.makedirs(folder)
    
#     proto_net_save_path = folder + '/' + 'FSL_protonet(' + str(n_support) + ',' + str(n_query2) + ')_' + save_path
#     match_net_save_path = folder + '/' + 'FSL_matchnet(' + str(n_support) + ',' + str(n_query2) + ')_' + save_path
#     rel_net_save_path = folder + '/' + 'FSL_relnet(' + str(n_support) + ',' + str(n_query2) + ')_' + save_path

#     num_epochs = 5

#     # We want to repeat the training and evaluation for each of the 3 networks
#     # 50 times and get an average accuracy and std_dev over the different initialisations

#     print('-'*50)
#     print('-'*50)
#     print(f'Training Matching Network {n_support}-shot')
#     print('test features: ', test_features)
#     print('-'*50)
#     print('-'*50)
#     learner_matchnet = FSL.train_evaluate(match_net, match_net_save_path, train_loader, val_episodes, epochs=num_epochs, onehot=True)

#     print('-'*50)
#     print('-'*50)
#     print(f'Evaluating Matching Network {n_support}-shot')
#     print('test features: ', test_features)
#     print('-'*50)
#     print('-'*50)
#     eval_match = FSL.evaluate(learner_matchnet[0], match_net_save_path, test_episodes, test_features, onehot=True)
#     match_acc = eval_match[3]
#     match_prec_avg = eval_match[9]
#     match_rec_avg = eval_match[10]

#     print('-'*50)
#     print('-'*50)
#     print(f'Training Relation Network for {n_support}-shot')
#     print('test features: ', test_features)
#     print('-'*50)
#     print('-'*50)
#     learner_relnet = FSL.train_evaluate(relation_net, rel_net_save_path, train_loader, val_episodes, epochs=int(2.5*num_epochs), onehot=True)

#     print('-'*50)
#     print('-'*50)
#     print(f'Evaluating Relation Network for {n_support}-shot')
#     print('test features: ', test_features)
#     print('-'*50)
#     print('-'*50)
#     eval_relnet = FSL.evaluate(learner_relnet[0], rel_net_save_path, test_episodes, test_features, onehot=True)
#     rel_acc = eval_relnet[3]
#     rel_prec_avg = eval_relnet[9]
#     rel_rec_avg = eval_relnet[10]


#     print('-'*50)
#     print('-'*50)
#     print(f'Training Prototypical Network {n_support}-shot')
#     print('test features: ', test_features)
#     print('-'*50)
#     print('-'*50)
#     learner_protonet = FSL.train_evaluate(prot_net, proto_net_save_path, train_loader, val_episodes, epochs=num_epochs, onehot=True)
    
#     print('-'*50)
#     print('-'*50)
#     print(f'Evaluating Prototypical Network {n_support}-shot')
#     print('test features: ', test_features)
#     print('-'*50)
#     print('-'*50)
#     eval_proto = evaluate(learner_protonet[0], proto_net_save_path, test_episodes, test_features, onehot=True)
#     proto_acc = eval_proto[3]
#     proto_prec_avg = eval_proto[9]
#     proto_rec_avg = eval_proto[10]

#     return proto_acc*100, match_acc*100, rel_acc*100, proto_prec_avg, match_prec_avg, rel_prec_avg, proto_rec_avg, match_rec_avg, rel_rec_avg


if __name__ == "__main__":
    args = parse_arguments()
    print(f"Saving to directory: {args.save_folder}")
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    
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

    # get rid of randomness in training if wanted
    # SEED = 42

    # random.seed(SEED)
    # np.random.seed(SEED)
    # torch.manual_seed(SEED)
    # torch.cuda.manual_seed(SEED)

    # torch.backends.cudnn.benchmark = False
    # torch.backends.cuda.matmul.allow_tf32 = False
    # torch.backends.cudnn.allow_tf32 = False

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

        train_val_features = ['single dangling bond' , 'As A' ,  'C defect', 'siloxane inv', 
                        'single DV on Si(001)', 'C defect inv', 'double dangling bond', 'm1 inv',
                        'single dihydride', 'double dangling bond inv', 'As B inv', 'h1',
                        'h2', 't1 inv', 'g1 inv', 'h2 inv', 'h1 inv', 't1' , 'single dangling bond inv', 'As A inv' , 'm1', 'g1',  
                        'single dihydride inv', 'As B','single DV on Si(001) inv','siloxane']
        
        num_sets = 2
        set_size = 6

        unique_samples = set()

        while len(unique_samples) < num_sets:
            sample = tuple(sorted(random.sample(train_val_features, set_size)))
            unique_samples.add(sample)
        
        # Convert to list if you want
        val_features_list = list(unique_samples)
        # get remainder of features for training
        train_features_list = []
       
        for feature_list in val_features_list:
            train_features = list(set(train_val_features) - set(feature_list))
            train_features_list.append(train_features)

        test_features = ['TiO2_vacancy', 'TiO2_hydroxyl']

        save_path = 'TiO2(110)_40pix_inv'

        proto_accs = []
        match_accs = []
        rel_accs = []
        proto_precs = []
        match_precs = []
        rel_precs = []
        proto_recs = []
        match_recs = []
        rel_recs = []
        for train_features, val_features in zip(train_features_list, val_features_list):
            print('Training with train features: ', train_features, ' and val features: ', val_features)
            results = FSL.train_save_match_proto_rel(n_way, n_support, n_query, n_train_episodes, train_features, val_features, test_features, args.save_folder, save_path)
            proto_acc, match_acc, rel_acc, proto_prec_avg, match_prec_avg, rel_prec_avg, proto_rec_avg, match_rec_avg, rel_rec_avg = results
            proto_accs.append(proto_acc)
            match_accs.append(match_acc)
            rel_accs.append(rel_acc)
            proto_precs.append(proto_prec_avg)
            match_precs.append(match_prec_avg)
            rel_precs.append(rel_prec_avg)
            proto_recs.append(proto_rec_avg)
            match_recs.append(match_rec_avg)
            rel_recs.append(rel_rec_avg)
        proto_acc_avg = np.mean(proto_accs)
        match_acc_avg = np.mean(match_accs)
        rel_acc_avg = np.mean(rel_accs)
        proto_prec_avg = np.mean(proto_precs)
        match_prec_avg = np.mean(match_precs)
        rel_prec_avg = np.mean(rel_precs)
        proto_rec_avg = np.mean(proto_recs)
        match_rec_avg = np.mean(match_recs)
        rel_rec_avg = np.mean(rel_recs)
        proto_acc_std = np.std(proto_accs)
        match_acc_std = np.std(match_accs)
        rel_acc_std = np.std(rel_accs)
        proto_precs_std = np.std(proto_precs)
        match_precs_std = np.std(match_precs)
        rel_precs_std = np.std(rel_precs)
        proto_recs_std = np.std(proto_recs)
        match_recs_std = np.std(match_recs)
        rel_recs_std = np.std(rel_recs)

        print(f'Average over {num_sets} different train/val splits for {n_way}-way, {n_support}-shot:') 
        print(f'ProtoNet Accuracy: {proto_acc_avg} % +/- {proto_acc_std}%, MatchNet Accuracy: {match_acc_avg} % +/- {match_acc_std}%, RelNet Accuracy: {rel_acc_avg} % +/- {rel_acc_std}%')
        print(f'ProtoNet Precision: {proto_prec_avg} +/- {proto_precs_std}, MatchNet Precision: {match_prec_avg} +/- {match_precs_std}, RelNet Precision: {rel_prec_avg} +/- {rel_precs_std}')
        print(f'ProtoNet Recall: {proto_rec_avg} +/- {proto_recs_std}, MatchNet Recall: {match_rec_avg} +/- {match_recs_std}, RelNet Recall: {rel_rec_avg} +/- {rel_recs_std}')       

        results_dict[f'({n_support},{n_query})_{save_path}'] = (proto_acc_avg, proto_acc_std, match_acc_avg, match_acc_std,
                                                                rel_acc_avg, rel_acc_std, proto_prec_avg, proto_precs_std,
                                                                match_prec_avg, match_precs_std, rel_prec_avg, rel_precs_std,
                                                                proto_rec_avg, proto_recs_std, match_rec_avg, match_recs_std, rel_rec_avg, rel_recs_std)

        ##################
        ##### 1-shot models with the inverse features included, TiO2(110)
        ##################

        n_way = 2
        n_support = 1

        train_val_features = ['single dangling bond' , 'As A' ,  'C defect', 'siloxane inv', 
                        'single DV on Si(001)', 'C defect inv', 'double dangling bond', 'm1 inv',
                        'single dihydride', 'double dangling bond inv', 'As B inv', 'h1',
                        'h2', 't1 inv', 'g1 inv', 'h2 inv', 'h1 inv', 't1' , 'single dangling bond inv', 'As A inv' , 'm1', 'g1',  
                        'single dihydride inv', 'As B','single DV on Si(001) inv','siloxane']
        
        num_sets = 2
        set_size = 6

        unique_samples = set()

        while len(unique_samples) < num_sets:
            sample = tuple(sorted(random.sample(train_val_features, set_size)))
            unique_samples.add(sample)

        # Convert to list if you want
        val_features_list = list(unique_samples)
        # get remainder of features for training
        train_features_list = []
        for feature_list in val_features_list:
            train_features = [feat for feat in train_val_features if feat not in feature_list]
            train_features_list.append(train_features)

        test_features = ['TiO2_vacancy', 'TiO2_hydroxyl']

        save_path = 'TiO2(110)_40pix_inv'

        proto_accs = []
        match_accs = []
        rel_accs = []
        proto_precs = []
        match_precs = []
        rel_precs = []
        proto_recs = []
        match_recs = []
        rel_recs = []
        for train_features, val_features in zip(train_features_list, val_features_list):
            print('Training with train features: ', train_features, ' and val features: ', val_features)
            results = FSL.train_save_match_proto_rel(n_way, n_support, n_query, n_train_episodes, train_features, val_features, test_features, args.save_folder, save_path)
            proto_acc, match_acc, rel_acc, proto_prec_avg, match_prec_avg, rel_prec_avg, proto_rec_avg, match_rec_avg, rel_rec_avg = results
            proto_accs.append(proto_acc)
            match_accs.append(match_acc)
            rel_accs.append(rel_acc)
            proto_precs.append(proto_prec_avg)
            match_precs.append(match_prec_avg)
            rel_precs.append(rel_prec_avg)
            proto_recs.append(proto_rec_avg)
            match_recs.append(match_rec_avg)
            rel_recs.append(rel_rec_avg)
        proto_acc_avg = np.mean(proto_accs)
        match_acc_avg = np.mean(match_accs)
        rel_acc_avg = np.mean(rel_accs)
        proto_prec_avg = np.mean(proto_precs)
        match_prec_avg = np.mean(match_precs)
        rel_prec_avg = np.mean(rel_precs)
        proto_rec_avg = np.mean(proto_recs)
        match_rec_avg = np.mean(match_recs)
        rel_rec_avg = np.mean(rel_recs)
        proto_acc_std = np.std(proto_accs)
        match_acc_std = np.std(match_accs)
        rel_acc_std = np.std(rel_accs)
        proto_precs_std = np.std(proto_precs)
        match_precs_std = np.std(match_precs)
        rel_precs_std = np.std(rel_precs)
        proto_recs_std = np.std(proto_recs)
        match_recs_std = np.std(match_recs)
        rel_recs_std = np.std(rel_recs)

        print(f'Average over {num_sets} different train/val splits for {n_way}-way, {n_support}-shot:') 
        print(f'ProtoNet Accuracy: {proto_acc_avg} % +/- {proto_acc_std}%, MatchNet Accuracy: {match_acc_avg} % +/- {match_acc_std}%, RelNet Accuracy: {rel_acc_avg} % +/- {rel_acc_std}%')
        print(f'ProtoNet Precision: {proto_prec_avg} +/- {proto_precs_std}, MatchNet Precision: {match_prec_avg} +/- {match_precs_std}, RelNet Precision: {rel_prec_avg} +/- {rel_precs_std}')
        print(f'ProtoNet Recall: {proto_rec_avg} +/- {proto_recs_std}, MatchNet Recall: {match_rec_avg} +/- {match_recs_std}, RelNet Recall: {rel_rec_avg} +/- {rel_recs_std}')       

        results_dict[f'({n_support},{n_query})_{save_path}'] = (proto_acc_avg, proto_acc_std, match_acc_avg, match_acc_std,
                                                                rel_acc_avg, rel_acc_std, proto_prec_avg, proto_precs_std,
                                                                match_prec_avg, match_precs_std, rel_prec_avg, rel_precs_std,
                                                                proto_rec_avg, proto_recs_std, match_rec_avg, match_recs_std, rel_rec_avg, rel_recs_std)
        
    ##################################################################

    if args.substrate == 'Si':
            
        ##################
        ##### 3-shot models with the inverse features included, Si(001)
        ##################

        n_way= 4
        n_support = 3
 
        save_path = 'Si(001)_40pix_inv'

        train_val_features = ['single dangling bond' , 'As A' , 'siloxane', 'C defect', 'g1' , 'As A inv' ,
                  'siloxane inv', 'g1 inv', 'C defect inv', 'TiO2_vacancy', 'h1', 'm1', 'h2', 't1', 
                  'TiO2_vacancy inv', 'TiO2_hydroxyl inv', 'm1 inv', 'single dangling bond inv','TiO2_hydroxyl', 'h2 inv', 't1 inv', 'h1 inv']

        print('total features len: ', len(train_val_features))
        print('total features length set:', len(list(set(train_val_features))))
        print('features: ', train_val_features)
        print('features set: ', list(set(train_val_features)))

        num_sets = 2
        set_size = 8

        unique_samples = set()

        while len(unique_samples) < num_sets:
            sample = tuple(sorted(random.sample(train_val_features, set_size)))
            unique_samples.add(sample)

        # Convert to list if you want
        val_features_list = list(unique_samples)
        # get remainder of features for training
        train_features_list = []
        for feature_list in val_features_list:
            train_features = [feat for feat in train_val_features if feat not in feature_list]
            train_features_list.append(train_features)

        for i, (train_features, val_features) in enumerate(zip(train_features_list, val_features_list)):
            print(f'Split {i}: val has {len(list(set(val_features)))} features, train has {len(list(set(train_features)))} features')
            if len(val_features) < n_way:
                print(f'val_features too small!!!!')
                continue

        test_features = ['double dangling bond', 'As B', 'single dihydride', 'single DV on Si(001)']

        proto_accs = []
        match_accs = []
        rel_accs = []
        proto_precs = []
        match_precs = []
        rel_precs = []
        proto_recs = []
        match_recs = []
        rel_recs = []
        for train_features, val_features in zip(train_features_list, val_features_list):
            print('Training with train features: ', train_features
                  , ' and val features: ', val_features,
                  ' and test features: ', test_features)
            results = FSL.train_save_match_proto_rel(n_way, n_support, n_query, n_train_episodes, train_features, val_features, test_features, args.save_folder, save_path)
            proto_acc, match_acc, rel_acc, proto_prec_avg, match_prec_avg, rel_prec_avg, proto_rec_avg, match_rec_avg, rel_rec_avg = results
            proto_accs.append(proto_acc)
            match_accs.append(match_acc)
            rel_accs.append(rel_acc)
            proto_precs.append(proto_prec_avg)
            match_precs.append(match_prec_avg)
            rel_precs.append(rel_prec_avg)
            proto_recs.append(proto_rec_avg)
            match_recs.append(match_rec_avg)
            rel_recs.append(rel_rec_avg)
        proto_acc_avg = np.mean(proto_accs)
        match_acc_avg = np.mean(match_accs)
        rel_acc_avg = np.mean(rel_accs)
        proto_prec_avg = np.mean(proto_precs)
        match_prec_avg = np.mean(match_precs)
        rel_prec_avg = np.mean(rel_precs)
        proto_rec_avg = np.mean(proto_recs)
        match_rec_avg = np.mean(match_recs)
        rel_rec_avg = np.mean(rel_recs)
        proto_acc_std = np.std(proto_accs)
        match_acc_std = np.std(match_accs)
        rel_acc_std = np.std(rel_accs)
        proto_precs_std = np.std(proto_precs)
        match_precs_std = np.std(match_precs)
        rel_precs_std = np.std(rel_precs)
        proto_recs_std = np.std(proto_recs)
        match_recs_std = np.std(match_recs)
        rel_recs_std = np.std(rel_recs)

        print(f'Average over {num_sets} different train/val splits for {n_way}-way, {n_support}-shot:') 
        print(f'ProtoNet Accuracy: {proto_acc_avg} % +/- {proto_acc_std}%, MatchNet Accuracy: {match_acc_avg} % +/- {match_acc_std}%, RelNet Accuracy: {rel_acc_avg} % +/- {rel_acc_std}%')
        print(f'ProtoNet Precision: {proto_prec_avg} +/- {proto_precs_std}, MatchNet Precision: {match_prec_avg} +/- {match_precs_std}, RelNet Precision: {rel_prec_avg} +/- {rel_precs_std}')
        print(f'ProtoNet Recall: {proto_rec_avg} +/- {proto_recs_std}, MatchNet Recall: {match_rec_avg} +/- {match_recs_std}, RelNet Recall: {rel_rec_avg} +/- {rel_recs_std}')       

        results_dict[f'({n_support},{n_query})_{save_path}'] = (proto_acc_avg, proto_acc_std, match_acc_avg, match_acc_std,
                                                                rel_acc_avg, rel_acc_std, proto_prec_avg, proto_precs_std,
                                                                match_prec_avg, match_precs_std, rel_prec_avg, rel_precs_std,
                                                                proto_rec_avg, proto_recs_std, match_rec_avg, match_recs_std, rel_rec_avg, rel_recs_std)

        ##################
        ##### 1-shot models with the inverse features included, Si(001)
        ##################

        n_way= 4
        n_support = 1
 
        save_path = 'Si(001)_40pix_inv'

        train_val_features = ['single dangling bond' , 'As A' , 'siloxane', 'C defect', 'g1' , 'As A inv' ,
                  'siloxane inv', 'g1 inv', 'C defect inv', 'TiO2_vacancy', 'h1', 'm1', 'h2', 't1', 
                  'TiO2_vacancy inv', 'TiO2_hydroxyl inv', 'm1 inv', 'single dangling bond inv','TiO2_hydroxyl', 'h2 inv', 't1 inv', 'h1 inv']

        print('total features len: ', len(train_val_features))
        print('total features length set:', len(list(set(train_val_features))))
        print('features: ', train_val_features)
        print('features set: ', list(set(train_val_features)))

        num_sets = 2
        set_size = 8

        unique_samples = set()

        while len(unique_samples) < num_sets:
            sample = tuple(sorted(random.sample(train_val_features, set_size)))
            unique_samples.add(sample)

        # Convert to list if you want
        val_features_list = list(unique_samples)
        # get remainder of features for training
        train_features_list = []
        for feature_list in val_features_list:
            train_features = [feat for feat in train_val_features if feat not in feature_list]
            train_features_list.append(train_features)

        for i, (train_features, val_features) in enumerate(zip(train_features_list, val_features_list)):
            print(f'Split {i}: val has {len(list(set(val_features)))} features, train has {len(list(set(train_features)))} features')
            if len(val_features) < n_way:
                print(f'val_features too small!!!!')
                continue

        test_features = ['double dangling bond', 'As B', 'single dihydride', 'single DV on Si(001)']

        proto_accs = []
        match_accs = []
        rel_accs = []
        proto_precs = []
        match_precs = []
        rel_precs = []
        proto_recs = []
        match_recs = []
        rel_recs = []
        for train_features, val_features in zip(train_features_list, val_features_list):
            print('Training with train features: ', train_features
                  , ' and val features: ', val_features,
                  ' and test features: ', test_features)
            results = FSL.train_save_match_proto_rel(n_way, n_support, n_query, n_train_episodes, train_features, val_features, test_features, args.save_folder, save_path)
            proto_acc, match_acc, rel_acc, proto_prec_avg, match_prec_avg, rel_prec_avg, proto_rec_avg, match_rec_avg, rel_rec_avg = results
            proto_accs.append(proto_acc)
            match_accs.append(match_acc)
            rel_accs.append(rel_acc)
            proto_precs.append(proto_prec_avg)
            match_precs.append(match_prec_avg)
            rel_precs.append(rel_prec_avg)
            proto_recs.append(proto_rec_avg)
            match_recs.append(match_rec_avg)
            rel_recs.append(rel_rec_avg)
        proto_acc_avg = np.mean(proto_accs)
        match_acc_avg = np.mean(match_accs)
        rel_acc_avg = np.mean(rel_accs)
        proto_prec_avg = np.mean(proto_precs)
        match_prec_avg = np.mean(match_precs)
        rel_prec_avg = np.mean(rel_precs)
        proto_rec_avg = np.mean(proto_recs)
        match_rec_avg = np.mean(match_recs)
        rel_rec_avg = np.mean(rel_recs)
        proto_acc_std = np.std(proto_accs)
        match_acc_std = np.std(match_accs)
        rel_acc_std = np.std(rel_accs)
        proto_precs_std = np.std(proto_precs)
        match_precs_std = np.std(match_precs)
        rel_precs_std = np.std(rel_precs)
        proto_recs_std = np.std(proto_recs)
        match_recs_std = np.std(match_recs)
        rel_recs_std = np.std(rel_recs)

        print(f'Average over {num_sets} different train/val splits for {n_way}-way, {n_support}-shot:') 
        print(f'ProtoNet Accuracy: {proto_acc_avg} % +/- {proto_acc_std}%, MatchNet Accuracy: {match_acc_avg} % +/- {match_acc_std}%, RelNet Accuracy: {rel_acc_avg} % +/- {rel_acc_std}%')
        print(f'ProtoNet Precision: {proto_prec_avg} +/- {proto_precs_std}, MatchNet Precision: {match_prec_avg} +/- {match_precs_std}, RelNet Precision: {rel_prec_avg} +/- {rel_precs_std}')
        print(f'ProtoNet Recall: {proto_rec_avg} +/- {proto_recs_std}, MatchNet Recall: {match_rec_avg} +/- {match_recs_std}, RelNet Recall: {rel_rec_avg} +/- {rel_recs_std}')       

        results_dict[f'({n_support},{n_query})_{save_path}'] = (proto_acc_avg, proto_acc_std, match_acc_avg, match_acc_std,
                                                                rel_acc_avg, rel_acc_std, proto_prec_avg, proto_precs_std,
                                                                match_prec_avg, match_precs_std, rel_prec_avg, rel_precs_std,
                                                                proto_rec_avg, proto_recs_std, match_rec_avg, match_recs_std, rel_rec_avg, rel_recs_std)

    if args.substrate == 'Ge':
            
        ##################
        ##### 3-shot models with the inverse features included, Ge(001)
        ##################
     
        n_way= 4
        n_support = 3

        train_val_features = ['single dangling bond' , 'As A' ,  'C defect', 'siloxane inv', 
                        'single DV on Si(001)', 'C defect inv', 'TiO2_hydroxyl inv', 'double dangling bond', 
                        'single dihydride', 'double dangling bond inv', 'TiO2_vacancy', 'As B inv',
                        'single dangling bond inv', 'As A inv' , 'TiO2_hydroxyl', 'TiO2_vacancy inv', 
                        'single dihydride inv', 'As B','single DV on Si(001) inv','siloxane']
        
        save_path = 'Ge(001)_40pix_inv'

        num_sets = 2
        set_size = 5

        unique_samples = set()

        while len(unique_samples) < num_sets:
            sample = tuple(sorted(random.sample(train_val_features, set_size)))
            unique_samples.add(sample)

        # Convert to list if you want
        val_features_list = list(unique_samples)
        # get remainder of features for training
        train_features_list = []
        
        for feature_list in val_features_list:
            train_features = list(set(train_val_features) - set(feature_list))
            train_features_list.append(train_features)

        for i, (train_features, val_features) in enumerate(zip(train_features_list, val_features_list)):
            print(f'Split {i}: val has {len(val_features)} features, train has {len(train_features)} features')
            if len(val_features) < n_way:
                print(f'val_features too small!!!!')
                continue

        test_features = ['t1', 'g1', 'm1', 'h2', 'h1']

        proto_accs = []
        match_accs = []
        rel_accs = []
        proto_precs = []
        match_precs = []
        rel_precs = []
        proto_recs = []
        match_recs = []
        rel_recs = []
        for train_features, val_features in zip(train_features_list, val_features_list):
            print('Training with train features: ', train_features
                  , ' and val features: ', val_features)
            results = FSL.train_save_match_proto_rel(n_way, n_support, n_query, n_train_episodes, train_features, val_features, test_features, args.save_folder, save_path)
            proto_acc, match_acc, rel_acc, proto_prec_avg, match_prec_avg, rel_prec_avg, proto_rec_avg, match_rec_avg, rel_rec_avg = results
            proto_accs.append(proto_acc)
            match_accs.append(match_acc)
            rel_accs.append(rel_acc)
            proto_precs.append(proto_prec_avg)
            match_precs.append(match_prec_avg)
            rel_precs.append(rel_prec_avg)
            proto_recs.append(proto_rec_avg)
            match_recs.append(match_rec_avg)
            rel_recs.append(rel_rec_avg)
        proto_acc_avg = np.mean(proto_accs)
        match_acc_avg = np.mean(match_accs)
        rel_acc_avg = np.mean(rel_accs)
        proto_prec_avg = np.mean(proto_precs)
        match_prec_avg = np.mean(match_precs)
        rel_prec_avg = np.mean(rel_precs)
        proto_rec_avg = np.mean(proto_recs)
        match_rec_avg = np.mean(match_recs)
        rel_rec_avg = np.mean(rel_recs)
        proto_acc_std = np.std(proto_accs)
        match_acc_std = np.std(match_accs)
        rel_acc_std = np.std(rel_accs)
        proto_precs_std = np.std(proto_precs)
        match_precs_std = np.std(match_precs)
        rel_precs_std = np.std(rel_precs)
        proto_recs_std = np.std(proto_recs)
        match_recs_std = np.std(match_recs)
        rel_recs_std = np.std(rel_recs)

        print(f'Average over {num_sets} different train/val splits for {n_way}-way, {n_support}-shot:') 
        print(f'ProtoNet Accuracy: {proto_acc_avg} % +/- {proto_acc_std}%, MatchNet Accuracy: {match_acc_avg} % +/- {match_acc_std}%, RelNet Accuracy: {rel_acc_avg} % +/- {rel_acc_std}%')
        print(f'ProtoNet Precision: {proto_prec_avg} +/- {proto_precs_std}, MatchNet Precision: {match_prec_avg} +/- {match_precs_std}, RelNet Precision: {rel_prec_avg} +/- {rel_precs_std}')
        print(f'ProtoNet Recall: {proto_rec_avg} +/- {proto_recs_std}, MatchNet Recall: {match_rec_avg} +/- {match_recs_std}, RelNet Recall: {rel_rec_avg} +/- {rel_recs_std}')       

        results = FSL.train_save_match_proto_rel(n_way, n_support, n_query, n_train_episodes, train_features, val_features, test_features,  args.save_folder, save_path)
        results_dict[f'({n_support},{n_query})_{save_path}'] = results

        results_dict[f'({n_support},{n_query})_{save_path}'] = (proto_acc_avg, proto_acc_std, match_acc_avg, match_acc_std,
                                                                rel_acc_avg, rel_acc_std, proto_prec_avg, proto_precs_std,
                                                                match_prec_avg, match_precs_std, rel_prec_avg, rel_precs_std,
                                                                proto_rec_avg, proto_recs_std, match_rec_avg, match_recs_std, rel_rec_avg, rel_recs_std)

        ##################
        ##### 1-shot models with the inverse features included, Ge(001)
        ##################
        
        n_way= 4
        n_support = 1

        train_val_features = ['single dangling bond' , 'As A' ,  'C defect', 'siloxane inv', 
                        'single DV on Si(001)', 'C defect inv', 'TiO2_hydroxyl inv', 'double dangling bond', 
                        'single dihydride', 'double dangling bond inv', 'TiO2_vacancy', 'As B inv',
                        'single dangling bond inv', 'As A inv' , 'TiO2_hydroxyl', 'TiO2_vacancy inv', 
                        'single dihydride inv', 'As B','single DV on Si(001) inv','siloxane']
        
        save_path = 'Ge(001)_40pix_inv'

        num_sets = 2
        set_size = 5

        unique_samples = set()

        while len(unique_samples) < num_sets:
            sample = tuple(sorted(random.sample(train_val_features, set_size)))
            unique_samples.add(sample)

        # Convert to list if you want
        val_features_list = list(unique_samples)
        # get remainder of features for training
        train_features_list = []
        for feature_list in val_features_list:
            train_features = list(set(train_val_features) - set(feature_list))
            train_features_list.append(train_features)
        
        for i, (train_features, val_features) in enumerate(zip(train_features_list, val_features_list)):
            print(f'Split {i}: val has {len(val_features)} features, train has {len(train_features)} features')
            if len(val_features) < n_way:
                print(f'val_features too small!!!!')
                continue

        test_features = ['t1', 'g1', 'm1', 'h2', 'h1']

        proto_accs = []
        match_accs = []
        rel_accs = []
        proto_precs = []
        match_precs = []
        rel_precs = []
        proto_recs = []
        match_recs = []
        rel_recs = []
        for train_features, val_features in zip(train_features_list, val_features_list):
            print('Training with train features: ', train_features
                  , ' and val features: ', val_features)
            results = FSL.train_save_match_proto_rel(n_way, n_support, n_query, n_train_episodes, train_features, val_features, test_features, args.save_folder, save_path)
            proto_acc, match_acc, rel_acc, proto_prec_avg, match_prec_avg, rel_prec_avg, proto_rec_avg, match_rec_avg, rel_rec_avg = results
            proto_accs.append(proto_acc)
            match_accs.append(match_acc)
            rel_accs.append(rel_acc)
            proto_precs.append(proto_prec_avg)
            match_precs.append(match_prec_avg)
            rel_precs.append(rel_prec_avg)
            proto_recs.append(proto_rec_avg)
            match_recs.append(match_rec_avg)
            rel_recs.append(rel_rec_avg)

        proto_acc_avg = np.mean(proto_accs)
        match_acc_avg = np.mean(match_accs)
        rel_acc_avg = np.mean(rel_accs)
        proto_prec_avg = np.mean(proto_precs)
        match_prec_avg = np.mean(match_precs)
        rel_prec_avg = np.mean(rel_precs)
        proto_rec_avg = np.mean(proto_recs)
        match_rec_avg = np.mean(match_recs)
        rel_rec_avg = np.mean(rel_recs)
        proto_acc_std = np.std(proto_accs)
        match_acc_std = np.std(match_accs)
        rel_acc_std = np.std(rel_accs)
        proto_precs_std = np.std(proto_precs)
        match_precs_std = np.std(match_precs)
        rel_precs_std = np.std(rel_precs)
        proto_recs_std = np.std(proto_recs)
        match_recs_std = np.std(match_recs)
        rel_recs_std = np.std(rel_recs)

        print(f'Average over {num_sets} different train/val splits for {n_way}-way, {n_support}-shot:') 
        print(f'ProtoNet Accuracy: {proto_acc_avg} % +/- {proto_acc_std}%, MatchNet Accuracy: {match_acc_avg} % +/- {match_acc_std}%, RelNet Accuracy: {rel_acc_avg} % +/- {rel_acc_std}%')
        print(f'ProtoNet Precision: {proto_prec_avg} +/- {proto_precs_std}, MatchNet Precision: {match_prec_avg} +/- {match_precs_std}, RelNet Precision: {rel_prec_avg} +/- {rel_precs_std}')
        print(f'ProtoNet Recall: {proto_rec_avg} +/- {proto_recs_std}, MatchNet Recall: {match_rec_avg} +/- {match_recs_std}, RelNet Recall: {rel_rec_avg} +/- {rel_recs_std}')       

        results_dict[f'({n_support},{n_query})_{save_path}'] = (proto_acc_avg, proto_acc_std, match_acc_avg, match_acc_std,
                                                                rel_acc_avg, rel_acc_std, proto_prec_avg, proto_precs_std,
                                                                match_prec_avg, match_precs_std, rel_prec_avg, rel_precs_std,
                                                                proto_rec_avg, proto_recs_std, match_rec_avg, match_recs_std, rel_rec_avg, rel_recs_std)

    # save the results in a pandas dataframe
    results_df = pd.DataFrame.from_dict(results_dict, orient='index', columns=[
        'proto_acc', 'proto_std_dev', 'match_acc', 
        'match_std_dev', 'rel_acc', 'rel_std_dev',
        'proto_prec_avg', 'proto_prec_std_dev', 'match_prec_avg', 
        'match_prec_std_dev', 'rel_prec_avg', 'rel_prec_std_dev',
        'proto_rec_avg', 'proto_rec_std_dev', 'match_rec_avg',
        'match_rec_std_dev', 'rel_rec_avg', 'rel_rec_std_dev' 
    ])

    save_to = args.save_folder + f'/few_shot_results_{args.substrate}_k_fold.csv'
    results_df.to_csv(save_to)