import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
import math

# pytorch modules
import torch
from torch.utils.data import Dataset, Subset, DataLoader, RandomSampler
#from torchvision.transforms import ToTensor
import torch.nn as nn # nn class our model inherits from
import torch.optim as optim
import torchvision.transforms.functional as F
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

# define device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

cropped_scans = np.load(r'/content/drive/My Drive/Colab Notebooks/AsH3 identification/FSL/defSTM.npy')#.transpose((0,3,1,2))
labels = np.load(r'/content/drive/My Drive/Colab Notebooks/AsH3 identification/FSL/defSTM_labels.npy')

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

res = 40

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
    def __init__(self, angles = [90,180,270,360]):
        self.angles = angles

    def __call__(self, img):
        angle = random.choice(self.angles)
        return F.rotate(img, angle)

class STM_bright_features(Dataset):

    def __init__(self, images, labels, res,
            features: List[str] = None, training_set = False):# a list of features in the dataset e.g. dangling bond
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
        fin_indices = [] # list that will conatin the indices of the features we want in this dataset
        for feature in self.all_features:
            if feature in self.features:
                fin_indices.append(self.all_features.index(feature))
              #  print(self.all_features.index(feature), feature)
        # images is of shape (num_samples, channels, res, res), labels is of shape (num_samples)
        fin_images = []
        fin_labels = []
        # the labels atm are from 0 to len(all_features)-1. If we have a dataset consisting of
        # a list less than all_features, then we need to reassign the labels so they go from
        # 0 to len(features)-1.
      #  print(fin_indices)
        for idx, i in zip(fin_indices, range(len(fin_indices)) ):
            # num of samples in this class
            num_samples_class = self.labels[self.labels==idx].shape
       #     print(num_samples_class, i)
            # give this a new y_true label
            fin_labels.append(i*torch.ones(num_samples_class))
        #    print(fin_labels[-1])
            fin_images.append(self.images[self.labels==idx,:,:,:])

        self.images = torch.vstack(fin_images)
        self.labels = torch.hstack(fin_labels)

        #print(fin_labels)

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

        return data

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
        # This method returns an episode from the dataset

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

def init_layer(L):
    # Initialization using fan-in
    if isinstance(L, nn.Conv2d):
        n = L.kernel_size[0]*L.kernel_size[1]*L.out_channels
        L.weight.data.normal_(0,math.sqrt(2.0/float(n)))
    elif isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)

class ConvBlock(nn.Module):
    def __init__(self, indim, outdim, pool = True, padding = 1):
        super(ConvBlock, self).__init__()
        self.indim  = indim
        self.outdim = outdim
        self.C      = nn.Conv2d(indim, outdim, 3, padding= padding)
        self.BN     = nn.BatchNorm2d(outdim)
        self.relu   = nn.ReLU(inplace=True)

        self.parametrized_layers = [self.C, self.BN, self.relu]

        self.pool   = nn.MaxPool2d(2)
        self.parametrized_layers.append(self.pool)

        for layer in self.parametrized_layers:
            init_layer(layer)

        self.trunk = nn.Sequential(*self.parametrized_layers)


    def forward(self,x):
        out = self.trunk(x)
        return out

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet,self).__init__()
        trunk = []
        for i in range(4):
            indim = 2 if i == 0 else 64
            outdim = 64
            B = ConvBlock(indim, outdim, pool = True)
            trunk.append(B)
        trunk.append(nn.Flatten())

        self.trunk = nn.Sequential(*trunk)


    def forward(self,x):
        out = self.trunk(x)
        return out

class MatchingNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        # define the embedding layer

        self.embedding_layer = ConvNet()

       # self.embedding_layer = EmbeddingNetwork(channels, crop_size)
        self.cos_dist = nn.CosineSimilarity(dim=1)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, query, support):
        # compute embeddings for query and support sets
        support["embeddings"] = self.embedding_layer(support["image"]) # f(x)
       # print(support['embeddings'].shape)
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

class RelationNetwork(nn.Module):

    def __init__(self, res):
        super().__init__()
        # define the embedding layer
        self.embedding_layer = ConvNet()

        self.fc_nodes = 100

        # the embedding vectors are of size
        if res==20:
          start = 128
        elif res==40:
          start = 512

        self.relation_module = nn.Sequential(
                               nn.Linear(start, self.fc_nodes),
                               nn.Dropout(0.2),
                               nn.ReLU(),
                               nn.BatchNorm1d(self.fc_nodes),
                               nn.Linear(self.fc_nodes, self.fc_nodes),
                               nn.Dropout(0.2),
                               nn.ReLU(),
                               nn.BatchNorm1d(self.fc_nodes),
                               nn.Linear(self.fc_nodes, 1),
                               nn.Dropout(0.2),
                               nn.ReLU(),
                        )

    def forward(self, query, support):
        # compute embeddings for query and support sets
        # input is a (num_channels, res, res)
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
        # (n_way, n_support, dimensions of embedding vector space)

        # we compute the sums of these support vectors
        # sums has shape (n_way, dimensions of embedding vector)
        sums = support_embeds.sum(dim=1)
        support["sums"] = sums/torch.sum(sums)

        relation_scores = {}
        for qvector in query['embeddings']:
            # qvector.shape = (dim_emb)
            relation_scores[qvector] = []
            concats = []
            for svector in sums:
                # svector.shape = (dim_emb)
                concat = torch.cat((qvector,svector))
                # concat.shape = (2*dim_emb)
                concats.append(concat)
            relation_scores[qvector] = self.relation_module(torch.stack(concats)).squeeze(1)
            # relation_scores[qvector].shape = (n_way)

        # relation_scores is a dictionary that has the query vectors as keys and their relation scores as values
        fin_rel_scores = torch.stack([rel_score for rel_score in relation_scores.values()])
        # fin_rel_scores.shape = (n_way*n_query, n_way)

        return fin_rel_scores

class PrototypicalNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        # define the embedding layer

        self.embedding_layer = ConvNet()

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

# Function to save the model
def save_model(model, path):
    torch.save(model.state_dict(), path)

# function to load model
def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model

def train_evaluate(fslNet, file_name, epochs = 200, onehot=True):
  # define the FSL
  learner = FewShotLearner(fslNet)

  # list of accuracies
  accuracies = []

  # train
  for epoch in range(epochs):
    learner.train()
    trainer = pl.Trainer(accelerator="gpu", devices = 1, max_epochs=1)
    trainer.fit(learner, train_loader, val_dataloaders=test_loader)

    # evaluate
    learner.eval()
    learner = learner.to(DEVICE)
    # instantiate the accuracy metric
    metric = Accuracy(task = 'multiclass', num_classes=n_way).to(DEVICE)
    # collect all the embeddings in the test set
    # so we can plot them later
    embedding_table = []
    pbar = tqdm.tqdm(range(len(test_episodes)))
    for episode_idx in pbar:

      support, query = test_episodes[episode_idx]

      # get the embeddings
      logits = learner.FSLnet(query, support)
      if not onehot:
        logits = torch.round(logits)
      # compute the accuracy
      acc = metric(logits, query["target"].to(DEVICE))
      pbar.set_description(f"Episode {episode_idx} // Accuracy: {acc.item():.2f}")
      # compute the total accuracy across all episodes
      total_acc = metric.compute()
    if epoch == 0:
      best_acc = total_acc
    print(f"Total accuracy, averaged across all episodes: {total_acc}")
    print(f"Finished epoch: {epoch}")
    if total_acc>best_acc:
      best_acc = total_acc
      # save
      save_model(learner, file_name + '.pth')


    accuracies.append(total_acc)
    std_dev = torch.std(torch.tensor(accuracies))
    confidenceInt = 1.96*std_dev/torch.sqrt(torch.tensor([len(accuracies)])) # 95% confidence interval assumin normal distribution

  return learner, metric, total_acc, accuracies, std_dev, confidenceInt

def evaluate(fslNet, save_path onehot=True):
  # define the FSL
#  learner = FewShotLearner(fslNet)
  learner = fslNet
  # list of accuracies
  accuracies = []


  # evaluate
  learner.eval()
  learner = learner.to(DEVICE)
  # instantiate the accuracy metric
  metric = Accuracy(task = 'multiclass', num_classes=n_way).to(DEVICE)
  # collect all the embeddings in the test set
  # so we can plot them later
  embedding_table = []
  pbar = tqdm.tqdm(range(len(val_episodes)))
  all_predicted_labels = []
  all_true_labels = []
  for episode_idx in pbar:
    support, query = val_episodes[episode_idx]
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
    pbar.set_description(f"Episode {episode_idx} // Accuracy: {acc.item():.2f}")
    acc = metric.compute()
    accuracies.append(acc)
    # conver the logits to their true label predictions
    predicted_targets = torch.argmax(logits, dim=1)
    predicted_labels = torch.zeros(predicted_targets.shape)
    for i in range(len(predicted_targets)):
      predicted_labels[i] = label_to_target[predicted_targets[i].item()]
    # append to lists of all_labels
    all_predicted_labels.append(predicted_labels)
    all_true_labels.append(query['label'])

  std_dev = torch.std(torch.tensor(accuracies))
  total_acc = torch.mean(torch.tensor(accuracies))
  confidenceInt = 1.96*std_dev/torch.sqrt(torch.tensor([len(accuracies)])) # 95% confidence interval assumin normal distributions
  print(f"Total accuracy, averaged across all episodes: {total_acc} +/- {confidenceInt[0]}")
  print(label_to_target)
  # make a confusion matrix, and calculate the micro-average precision+recall
  all_predicted_labels = torch.cat(all_predicted_labels).cpu().numpy()
  all_true_labels = torch.cat(all_true_labels).cpu().numpy()
  plot_confusion_matrix(all_true_labels, all_predicted_labels, np.unique(all_true_labels), val_features, save_path)
  # get precision and recall


  return learner, metric, total_acc, accuracies, std_dev, confidenceInt

def plot_confusion_matrix(y_true, y_pred, labels, label_names, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.title('Confusion Matrix')
    plt.save(save_path + '_confusion_matrix.png')
    return


##################
##### 3-shot models with the inverse features included
##################

n_way= 4
n_support = 3
n_query = 15
n_train_episodes = 20
n_test_episodes = 100
num_workers=0

train_features = ['single dangling bond' , 'As A' , 'siloxane', 'C defect', 'g1' , 'As A inv' ,
                  'siloxane inv', 'C defect inv', 'g1 inv', 'm1','TiO2_hydroxyl inv', 'h2 inv', 't1 inv',]
test_features = ['h2', 'single dangling bond inv', 'TiO2_vacancy', 'h1', 'm1 inv']
val_features = ['double dangling bond', 'As B', 'single dihydride', 'single DV on Si(001)']

# initialise datasets
trainDS = STM_bright_features(x_all, y_all, 40, features = train_features, training_set=True)
testDS = STM_bright_features(x_all, y_all, 40, features = test_features)
valDS = STM_bright_features(x_all, y_all, 40, features = val_features)
# initialise episodic datasets
trainDSep = EpisodeDataset(trainDS, n_way = n_way,
        n_support = n_support,
        n_query = n_query,
        n_episodes = n_train_episodes)
testDSep = EpisodeDataset(testDS, n_way = n_way,
        n_support = n_support,
        n_query = n_query,
        n_episodes = n_test_episodes)
valDSep = EpisodeDataset(valDS, n_way = n_way,
        n_support = n_support,
        n_query = n_query,
        n_episodes = n_test_episodes)

# initialise data loaders
train_loader = DataLoader(trainDSep, batch_size = None, num_workers = num_workers)
test_loader = DataLoader(testDSep, batch_size = None, num_workers = num_workers)
val_loader = DataLoader(valDSep, batch_size = None, num_workers = num_workers)

n_query2 = 15
n_episodes2 = 100

# load our evaluation data
test_episodes = EpisodeDataset(testDS, n_way = n_way,
        n_support = n_support,
        n_query = n_query2,
        n_episodes = n_episodes2)

val_episodes = EpisodeDataset(valDS, n_way = n_way,
        n_support = n_support,
        n_query = n_query2,
        n_episodes = n_episodes2)

prot_net = PrototypicalNetwork()
prot_net.to(DEVICE)
match_net = MatchingNetwork()
match_net.to(DEVICE)
relation_net = RelationNetwork(res=40)
relation_net.to(DEVICE)

learner_protonet = train_evaluate(prot_net, 'FSL_protonet(3,15)_40pix_inv', epochs=100, onehot=True)
eval_proto = evaluate(learner_protonet[0], 'FSL_protonet(3,15)_40pix_inv', onehot=True)
learner_matchnet = train_evaluate(match_net, 'FSL_matchnet(3,15)_40pix_inv', epochs=100, onehot=False)
eval_match = evaluate(learner_matchnet[0], 'FSL_matchnet(3,15)_40pix_inv', onehot=False)
learner_relnet = train_evaluate(relation_net, 'FSL_relnet(3,15)_40pix_inv', epochs=100, onehot=False)
eval_relnet = evaluate(learner_relnet[0], 'FSL_relnet(3,15)_40pix_inv', onehot=False)


############################################
##### 3-shot, no inverse features included
############################################

n_way= 4
n_support = 3
n_query = 15
n_train_episodes = 20
n_test_episodes = 100
num_workers=0

train_features = ['single dangling bond' , 'As A' , 'siloxane', 'C defect','g1','TiO2_hydroxyl']
test_features = ['h2', 'TiO2_vacancy', 'h1','m1']
val_features = ['double dangling bond', 'As B', 'single dihydride', 'single DV on Si(001)']


# initialise datasets
trainDS = STM_bright_features(x_all, y_all, 40, features = train_features, training_set=True)
testDS = STM_bright_features(x_all, y_all, 40, features = test_features)
valDS = STM_bright_features(x_all, y_all, 40, features = val_features)
# initialise episodic datasets
trainDSep = EpisodeDataset(trainDS, n_way = n_way,
        n_support = n_support,
        n_query = n_query,
        n_episodes = n_train_episodes)
testDSep = EpisodeDataset(testDS, n_way = n_way,
        n_support = n_support,
        n_query = n_query,
        n_episodes = n_test_episodes)
valDSep = EpisodeDataset(valDS, n_way = n_way,
        n_support = n_support,
        n_query = n_query,
        n_episodes = n_test_episodes)

# initialise data loaders
train_loader = DataLoader(trainDSep, batch_size = None, num_workers = num_workers)
test_loader = DataLoader(testDSep, batch_size = None, num_workers = num_workers)
val_loader = DataLoader(valDSep, batch_size = None, num_workers = num_workers)

n_query2 = 15
n_episodes2 = 100

# load our evaluation data
test_episodes = EpisodeDataset(testDS, n_way = n_way,
        n_support = n_support,
        n_query = n_query2,
        n_episodes = n_episodes2)

val_episodes = EpisodeDataset(valDS, n_way = n_way,
        n_support = n_support,
        n_query = n_query2,
        n_episodes = n_episodes2)

prot_net = PrototypicalNetwork()
prot_net.to(DEVICE)
match_net = MatchingNetwork()
match_net.to(DEVICE)
relation_net = RelationNetwork(res=40)
relation_net.to(DEVICE)

learner_protonet = train_evaluate(prot_net, 'FSL_protonet(3,15)_40pix', epochs=100, onehot=True)
eval_proto = evaluate(learner_protonet[0], 'FSL_protonet(3,15)_40pix', onehot=True)
learner_matchnet = train_evaluate(match_net, 'FSL_matchnet(3,15)_40pix', epochs=100, onehot=False)
eval_match = evaluate(learner_matchnet[0], 'FSL_matchnet(3,15)_40pix', onehot=False)
learner_relnet = train_evaluate(relation_net, 'FSL_relnet(3,15)_40pix', epochs=100, onehot=False)
eval_relnet = evaluate(learner_relnet[0], 'FSL_relnet(3,15)_40pix', onehot=False)



##################
##### 1-shot models with the inverse features included
##################

n_way= 4
n_support = 1
n_query = 15
n_train_episodes = 20
n_test_episodes = 100
num_workers=0

train_features = ['single dangling bond' , 'As A' , 'siloxane', 'C defect', 'g1' , 'As A inv' ,
                  'siloxane inv', 'C defect inv', 'g1 inv', 'm1','TiO2_hydroxyl inv', 'h2 inv', 't1 inv',]
test_features = ['h2', 'single dangling bond inv', 'TiO2_vacancy', 'h1', 'm1 inv']
val_features = ['double dangling bond', 'As B', 'single dihydride', 'single DV on Si(001)']

# initialise datasets
trainDS = STM_bright_features(x_all, y_all, 40, features = train_features, training_set=True)
testDS = STM_bright_features(x_all, y_all, 40, features = test_features)
valDS = STM_bright_features(x_all, y_all, 40, features = val_features)
# initialise episodic datasets
trainDSep = EpisodeDataset(trainDS, n_way = n_way,
        n_support = n_support,
        n_query = n_query,
        n_episodes = n_train_episodes)
testDSep = EpisodeDataset(testDS, n_way = n_way,
        n_support = n_support,
        n_query = n_query,
        n_episodes = n_test_episodes)
valDSep = EpisodeDataset(valDS, n_way = n_way,
        n_support = n_support,
        n_query = n_query,
        n_episodes = n_test_episodes)

# initialise data loaders
train_loader = DataLoader(trainDSep, batch_size = None, num_workers = num_workers)
test_loader = DataLoader(testDSep, batch_size = None, num_workers = num_workers)
val_loader = DataLoader(valDSep, batch_size = None, num_workers = num_workers)

n_query2 = 15
n_episodes2 = 100

# load our evaluation data
test_episodes = EpisodeDataset(testDS, n_way = n_way,
        n_support = n_support,
        n_query = n_query2,
        n_episodes = n_episodes2)

val_episodes = EpisodeDataset(valDS, n_way = n_way,
        n_support = n_support,
        n_query = n_query2,
        n_episodes = n_episodes2)

prot_net = PrototypicalNetwork()
prot_net.to(DEVICE)
match_net = MatchingNetwork()
match_net.to(DEVICE)
relation_net = RelationNetwork(res=40)
relation_net.to(DEVICE)

learner_protonet = train_evaluate(prot_net, 'FSL_protonet(1,15)_40pix_inv', epochs=100, onehot=True)
eval_proto = evaluate(learner_protonet[0], 'FSL_protonet(1,15)_40pix_inv', onehot=True)
learner_matchnet = train_evaluate(match_net, 'FSL_matchnet(1,15)_40pix_inv', epochs=100, onehot=False)
eval_match = evaluate(learner_matchnet[0], 'FSL_matchnet(1,15)_40pix_inv', onehot=False)
learner_relnet = train_evaluate(relation_net, 'FSL_relnet(1,15)_40pix_inv', epochs=100, onehot=False)
eval_relnet = evaluate(learner_relnet[0], 'FSL_relnet(1,15)_40pix_inv', onehot=False)


############################################
##### 1-shot, no inverse features included
############################################

n_way= 4
n_support = 1
n_query = 15
n_train_episodes = 20
n_test_episodes = 100
num_workers=0

train_features = ['single dangling bond' , 'As A' , 'siloxane', 'C defect','g1','TiO2_hydroxyl']
test_features = ['h2', 'TiO2_vacancy', 'h1','m1']
val_features = ['double dangling bond', 'As B', 'single dihydride', 'single DV on Si(001)']


# initialise datasets
trainDS = STM_bright_features(x_all, y_all, 40, features = train_features, training_set=True)
testDS = STM_bright_features(x_all, y_all, 40, features = test_features)
valDS = STM_bright_features(x_all, y_all, 40, features = val_features)
# initialise episodic datasets
trainDSep = EpisodeDataset(trainDS, n_way = n_way,
        n_support = n_support,
        n_query = n_query,
        n_episodes = n_train_episodes)
testDSep = EpisodeDataset(testDS, n_way = n_way,
        n_support = n_support,
        n_query = n_query,
        n_episodes = n_test_episodes)
valDSep = EpisodeDataset(valDS, n_way = n_way,
        n_support = n_support,
        n_query = n_query,
        n_episodes = n_test_episodes)

# initialise data loaders
train_loader = DataLoader(trainDSep, batch_size = None, num_workers = num_workers)
test_loader = DataLoader(testDSep, batch_size = None, num_workers = num_workers)
val_loader = DataLoader(valDSep, batch_size = None, num_workers = num_workers)

n_query2 = 15
n_episodes2 = 100

# load our evaluation data
test_episodes = EpisodeDataset(testDS, n_way = n_way,
        n_support = n_support,
        n_query = n_query2,
        n_episodes = n_episodes2)

val_episodes = EpisodeDataset(valDS, n_way = n_way,
        n_support = n_support,
        n_query = n_query2,
        n_episodes = n_episodes2)

prot_net = PrototypicalNetwork()
prot_net.to(DEVICE)
match_net = MatchingNetwork()
match_net.to(DEVICE)
relation_net = RelationNetwork(res=40)
relation_net.to(DEVICE)

learner_protonet = train_evaluate(prot_net, 'FSL_protonet(1,15)_40pix', epochs=100, onehot=True)
eval_proto = evaluate(learner_protonet[0], 'FSL_protonet(1,15)_40pix', onehot=True)
learner_matchnet = train_evaluate(match_net, 'FSL_matchnet(1,15)_40pix', epochs=100, onehot=False)
eval_match = evaluate(learner_matchnet[0], 'FSL_matchnet(1,15)_40pix', onehot=False)
learner_relnet = train_evaluate(relation_net, 'FSL_relnet(1,15)_40pix', epochs=100, onehot=False)
eval_relnet = evaluate(learner_relnet[0], 'FSL_relnet(1,15)_40pix', onehot=False)
