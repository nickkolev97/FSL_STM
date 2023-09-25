# pytorch modules
import torch
from torch.utils.data import Dataset, Subset, DataLoader, RandomSampler
from torchvision import transforms

from typing import List, Dict, Any, Tuple
from collections import defaultdict
import random

class CustomDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label

class STM_bright_features(Dataset):

    def __init__(self, images, labels,
            features: List[str] = None):# a list of features in the dataset e.g. dangling bond
            # sizes: float = 1.0, # this is another thing that could be fed into it to help distinguish different features
                                 # since they're not all of the same size. For now we make all of them 11 pixels big
            # we could include other meta data in here too
        # )

        self.all_features = ['singling dangling bond', 'double dangling bond' ,
                             'As A' , 'As B', 'single DV on Si(001)', 'siloxane',
                             'C defect', 'single dihydride', 'singling dangling bond', 'double dangling bond' ,
                             'As A inv' , 'As B inv', 'single DV on Si(001) inv', 'siloxane inv',
                             'C defect inv', 'single dihydride inv', 'h1', 'h2', 't1', 'g1', 'm1', 
                             'TiO2_vacancy', 'TiO2_hydroxyl']

        self.features = features
        #self.sizes = sizes

        self.images = images
        self.labels = labels

        # all crops should be of size (15,15)
        self.transform = transforms.Resize((40,40))

        # select only the images that are given in the features list
        self.refine_images()

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
        # images is of shape (num_samples, channels, res, res), labels is of shape (num_samples)
        fin_images = []
        fin_labels = []
        # the labels atm are from 0 to len(all_features)-1. If we have a dataset consisting of
        # a list less than all_features, then we need to reassign the labels so they go from
        # 0 to len(features)-1.
        for idx, i in zip(fin_indices, range(len(fin_indices)) ):
            # num of samples in this class
            num_samples_class = self.labels[self.labels==idx].shape
            # give this a new y_true label
            fin_labels.append(i*torch.ones(num_samples_class))
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

        if data['image'].shape != (40,40):
          data['image'] = self.transform(data['image'])

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

        support, query = [], []
        for c in episode_classlist:
            # go through each class and make up the support and query datasets

            # dataset indices for this class
            all_indices = self.dataset.class_to_indices[c]

            # sample the support and query sets for this class
           # print(all_indices, self.n_support,self.n_query)
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
        #            'target': numpy array of shape (number of crops, y_values for this episode) }
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
        for item in list_of_dicts:
            images.append(item['image'])
            targets.append(item['target'])

        images = torch.stack(images, dim=0)
        targets = torch.stack(targets, dim=0)

        return {'image': images, 'target':targets}