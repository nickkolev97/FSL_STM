import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import math

# pytorch modules
import torch
from torch.utils.data import Dataset, Subset, DataLoader, RandomSampler
import torch.nn as nn 
import torch.optim as optim
from torchmetrics import Accuracy
import pytorch_lightning as pl
import tqdm
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torchvision import transforms

from typing import List, Dict, Any, Tuple
from collections import defaultdict
import random


###################################
### embedding network which is kept the same for every FSL network 
###################################
class EmbeddingNetwork(nn.Module):

    def __init__(self, channels, crop_size):
        super().__init__()
        fc_nodes = 100
        dropout = 0.2
        self.convolutional_relu_stack = nn.Sequential(
            nn.Conv2d(channels, 32, 3, stride=1, padding='valid'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=1, padding='valid'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=1, padding='valid'),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features = (crop_size-6)**2 *128, out_features= fc_nodes),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.BatchNorm1d(fc_nodes),
        )

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(fc_nodes, fc_nodes),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.BatchNorm1d(fc_nodes)
        )



    def forward(self, x):
        x = self.convolutional_relu_stack(x)
        for i in range(2):
            x = self.linear_relu_stack(x)

        return x
    
###################################
### Matching network implementation
###################################
class MatchingNetwork(nn.Module):

    def __init__(self, channels, crop_size):
        super().__init__()
        # define the embedding layer
        self.embedding_layer = EmbeddingNetwork(channels, crop_size)
        self.cos_dist = nn.CosineSimilarity(dim=1)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, query, support):
        # compute embeddings for query and support sets
        support["embeddings"] = self.embedding_layer(support["image"]) # f(x)
        query["embeddings"] = self.embedding_layer(query["image"]) # g(x_i), for us g = f


        # compute the cosine distances between the query embeddings and the support
        # query['embeddings'] is a tensor of shape (n_samples, dimensions of embedding vector space)
        cos_distances = []
        for embedding in support["embeddings"]:
          cos_distances.append(torch.exp(self.cos_dist(query["embeddings"], embedding)))
         # cos_distances.append(torch.cdist(query["embeddings"].unsqueeze(0), embedding.unsqueeze(0), p=2).squeeze(0)) # c(f(x), g(x_i))
        '''
        # support["prototypes"] is a tensor of shape
        # (n_way, dimensions of embedding vector space)

        # compute the distances between the query embeddings and the prototypes
        # query['embeddings'] is a tensor of shape (n_samples, dimensions of embedding vector space)
        distances = torch.cdist(query["embeddings"].unsqueeze(0), prototypes.unsqueeze(0), p=2).squeeze(0)
        '''

        cos_distances = torch.stack(cos_distances).squeeze(-1) # c(f(x),g(x_i))
        # cos_distances is of shape (n_support, n_query). We have a cosine distance vector between each
        # of the support embeddings and the query embeddings and then we take the exponential of it.
        attentions = self.softmax(cos_distances)

        support["attentions"] = attentions # a = e^c(f(x),g(x_i))/sum_(j=1)^k e^c(f(x),g(x_j))

        # output using integer labels
      #  y = torch.matmul(support["target"].float().to(DEVICE), support["attentions"]).float()

        # output using one hot encoding for targets (got better accuracy)
        y = torch.matmul( support["attentions"].T, torch.nn.functional.one_hot(support["target"]).float().to(DEVICE) )

        # the final predictions should be (where we use einstein summation convention):
        # y = a(x,x_i)y_i. With a(x,x_i) = e^{c(f(x),g(x_i))}/sum_{j=1}^{k}e^{c(f(x),g(x_j))}

        return y



###################################
### Prototypical network implementation
###################################
class PrototypicalNetwork(nn.Module):

    def __init__(self, channels, crop_size):
        super().__init__()
        # define the embedding layer
        self.embedding_layer = EmbeddingNetwork(channels, crop_size)


    def forward(self, query, support):
        # compute embeddings for query and support sets
        support["embeddings"] = self.embedding_layer(support["image"])
        query["embeddings"] = self.embedding_layer(query["image"])

        # now we need to compute the prototype for each class
        # this was the 'average' class member
        support_embeds = []
        for idx in range(len(support["classlist"])):
            embeds = support["embeddings"][support["target"] == idx]
            support_embeds.append(embeds)
        # support_embeds is a list of torch tensors of shape
        # (n_support, dimensions of embedding vector space)

        support_embeds = torch.stack(support_embeds)
        # support_embeds now a tensor of shape
        # (n_way, n_support, dimensions of embedding vector space)

        # we compute the mean of these support vectors to get prototypes
        prototypes = support_embeds.mean(dim=1)
        support["prototypes"] = prototypes

        # support["prototypes"] is a tensor of shape
        # (n_way, dimensions of embedding vector space)

        # compute the distances between the query embeddings and the prototypes
        # query['embeddings'] is a tensor of shape (n_samples, dimensions of embedding vector space)
        distances = torch.cdist(query["embeddings"].unsqueeze(0), prototypes.unsqueeze(0), p=2).squeeze(0)
        # distances is a tensor of dimensions (n_samples, n_ways)
        distances = distances ** 2

        # the negative of the distances give the final output logits
        logits = - distances

        return logits

###################################
### Relation network implementation
###################################
class RelationNetwork(nn.Module):

    def __init__(self, channels, crop_size, classes):
        super().__init__()
        # define the embedding layer
        self.embedding_layer = EmbeddingNetwork(channels, crop_size)

        self.fc_nodes = 100

        # the embedding vectors are of size 100, so the input for the first layer
        # will be n_shot*200 (200 since they're concatenated)
        self.relation_module = nn.Sequential(
            nn.Linear(classes*200, self.fc_nodes),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.BatchNorm1d(self.fc_nodes),
            nn.Linear(self.fc_nodes, classes),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.BatchNorm1d(classes)
        )

    def forward(self, query, support):
        # compute embeddings for query and support sets
        support["embeddings"] = self.embedding_layer(support["image"]) # f(x)
        query["embeddings"] = self.embedding_layer(query["image"]) # g(x_i), for us g = f

        # sum up the embeddings of the support vectors in the same class
        support_embeds = []
        for idx in range(len(support["classlist"])):
            embeds = support["embeddings"][support["target"] == idx]
            support_embeds.append(embeds)
        # support_embeds is a list of torch tensors of shape
        # (n_support, dimensions of embedding vector space)

        support_embeds = torch.stack(support_embeds)
        # support_embeds now a tensor of shape
        # (n_shot, n_support, dimensions of embedding vector space)

        # we compute the sums of these support vectors
        sums = support_embeds.sum(dim=1)
        support["sums"] = sums #/torch.sum(sums)

        # each query vector now needs to be concatenated with each sum support vector
        # i.e. change each query vector from shape (dimensions of embedding vector space) to shape (n_shot, 2*dimensions of embedding vector)
        # overall, needs to be (n_query, n_shot,  2*dimensions of embedding vector)
        concats = []
        for vector in support["sums"]:
          vector = torch.tile(vector.unsqueeze(1), (1, query["embeddings"].shape[0] ))
          concats.append(torch.cat((query["embeddings"].T, vector), dim=0))

        concats = torch.flatten(torch.stack(concats).T, start_dim=1, end_dim=2)
        query["concats"] = concats # these will be fed into the relation module

        # feed through relation module
        relation_score = self.relation_module(concats)

        return relation_score

###################################
### pytorch lightning network
### it can take any of the three networks above (matching, prototypical, or relation)
### as the FS network
###################################
class FewShotLearner(pl.LightningModule):

    def __init__(self,
        FSLnet: nn.Module,
        learning_rate: float = 0.001,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['matchnet'])
        self.FSLnet = FSLnet
        self.learning_rate = learning_rate

        self.loss = nn.CrossEntropyLoss()
        self.metrics = nn.ModuleDict({
            'accuracy': Accuracy(task="multiclass", num_classes=n_way)
        })

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def step(self, batch, batch_idx, tag: str):
        support, query = batch

        logits = self.FSLnet(query, support)
        loss = self.loss(logits, query["target"])

        output = {"loss": loss}
        for k, metric in self.metrics.items():
            output[k] = metric(logits, query["target"])

        for k, v in output.items():
            self.log(f"{k}/{tag}", v)
        return output

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "test")