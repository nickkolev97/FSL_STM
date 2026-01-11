import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import cv2, argparse
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
import math
from typing import List, Dict, Any, Tuple
import random
import tqdm
from torchmetrics import Accuracy
import time
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# pytorch modules
import torch 
from torch.utils.data import Dataset # primitive for the data
#from torchvision.transforms import ToTensor
from torch.utils.data import WeightedRandomSampler, DataLoader # wraps the data so its iterable
from torch import nn # nn class our model inherits from
import torch.optim as optim


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Function to save the model
def save_model(model, path):
    torch.save(model.state_dict(), path)

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
        if features is None:
            features = self.all_features
        else:
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

    def split(self, test_size=0.3, random_state=42):
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
            training_set=True
        )
        
        val_dataset = STM_bright_features(
            images=X_val, 
            labels=y_val, 
            res=self.res, 
            features=self.features, 
            training_set=False
        )
        
        return train_dataset, val_dataset

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

class NeuralNetwork(nn.Module):
    def __init__(self, channels, crop_size, n_outputs, fc_layers, fc_nodes, dropout):
        super().__init__()
        
        self.fc_layers = fc_layers
        
        self.conv4 = ConvNet()

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(fc_nodes, fc_nodes),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.BatchNorm1d(fc_nodes)
        )
        
        self.linear_relu_stack_last = nn.Sequential(
            nn.Linear(fc_nodes, n_outputs)
        )
        
    def forward(self, x, training = True):
        x = self.convolutional_relu_stack(x)
        for i in range(self.fc_layers-1):
            x = self.linear_relu_stack(x)
        if training == True:
            x = self.linear_relu_stack_last(x)
            return 
        else: 
            x = torch.nn.functional.normalize(x)
            return x

class SimpleShot(nn.Module):
    def __init__(self, model):#, channels, crop_size, n_outputs, fc_layers, fc_nodes, dropout):
        super().__init__()
        self.classifier = model
        

    def forward(self, query, support, support_labels, k_shot):
        # find embeddings of query set
        x_q = self.classifier(query, training=False)
        # find embeddings of support
        x_s = self.classifier(support, training=False)
        
        # make an average feature vector for each class
        average_support_vecs = []
        for i in range(k_shot):
            average_support_vecs.append( x_s[torch.argmax(support_labels,dim=1)==i,:].mean(dim=0) )
        
        average_support_vecs = torch.stack(average_support_vecs)
        
        # L2 normalise output
        x_q = torch.nn.functional.normalize(x_q)
        average_support_vecs = torch.nn.functional.normalize(average_support_vecs)
        # x_q i of shape (n_query, dimension of embedding vector)
        # average_support_vecs is of shape (n_way, dimension of embedding vector)
        
        # find euclidean distances between them
        distances = torch.cdist(x_q, average_support_vecs, p=2).squeeze(0)     
        # distances is a tensor of dimensions (n_query, n_ways)
        distances = distances ** 2
        
        logits = torch.argmin(distances, dim=1)
        
        x_s_norm = torch.nn.functional.normalize(x_s)
        
        return logits, distances, x_q, x_s_norm

# test accuracy function
def testAccuracy(model, dataloader):
    
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
            # the label with the highest energy will be our prediction
            _, predicted = torch.max(outputs.data, 1)
            _, labels = torch.max(labels.data, 1)
            accuracy += (predicted == labels).sum().item()
    
    # compute the accuracy over all test images
    accuracy = (100 * accuracy / total)
    
    return(accuracy)

def train(model, dataloader_train, dataloader_test, loss_, num_epochs, path):
    # define lists to store accuracy gain as we train
    train_acc_gain = []
    test_acc_gain = []
    
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
            
            loss = loss_(outputs, labels.float())
            running_train_loss += loss.item()
        
            # Backward pass
            loss.backward()
            optimizer.step()

        accuracy = testAccuracy(model, dataloader_train)
        print('epoch', epoch, 'train accuracy over whole test set: %d %%' % (accuracy))
            
        # save the model if the accuracy is the best
        #if accuracy > best_accuracy:
        #    save_model(model, path)
        #    best_accuracy = accuracy
                
        # get the test accuracy
        model.eval()
        for i, (crops, labels) in enumerate(dataloader_test):
            # Get the crops and labels
            crops, labels = crops.to(device), labels.to(device)
            # get prediction and loss
            pred = model(crops.float())
            loss = loss_(pred, labels.float())
            
            running_test_loss += loss.item()
            
        accuracy = testAccuracy(model, dataloader_test)
        print('epoch', epoch, 'test accuracy over whole test set: %d %%' % (accuracy))

        # save the model if the accuracy is the best
        if accuracy > best_accuracy:
            print('Saving model from epoch', epoch)
            save_model(model, path)
            best_accuracy = accuracy
        
        

        print('Epoch: %d loss: %.3f' % (epoch + 1, running_test_loss / len(dataloader_test)))

def episodic_testing(network, episodes, n_way, k_shot, n_query, dataset):
    network.eval()
    total_correct = 0
    total_tests = episodes * n_query

    with torch.no_grad():
        for episode in range(episodes):
            # Sample support and query sets
            support, query = dataset[episode]

        #support/query = {'image': numpy array of shape (number of crops, num_channels, res, res),
        #                 'target': numpy array of shape (number of crops, y_values for this episode) 
        #                 'true_target': numpy array of shape (number of crops, true y_values)}
        
            support_crops, support_labels = support['images'].to(device), support['target'].to(device)
            query_crops, query_labels = query['images'].to(device), query['target'].to(device)

            # Get model predictions
            outputs, dists, query_embeds, support_embeds = network(query_crops, support_crops, support_labels, k_shot)

            query_labels_ = torch.argmax(query_labels, dim=1)

            total_correct += torch.sum(outputs == query_labels_).item()

    accuracy = total_correct / total_tests
    return accuracy

def plot_confusion_matrix(y_true, y_pred, labels, label_names, save_path):
    # covert the labels to the numbered labels I use in the paper
    features = ['single dangling bond', 'C defect', 'siloxane', 'As A' ,
                'As B' , 'double dangling bond' , 'single DV on Si(001)',
                'single dihydride',   
                't1', 'g1', 'h1', 'h2', 'm1', 
                'TiO2_vacancy', 'TiO2_hydroxyl']
    label_mapping = {feature: idx for idx, feature in enumerate(features)}
    y_true_mapped = [label_mapping[label_names[label]] for label in y_true]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=y_true_mapped)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.title('Confusion Matrix')
    plt.savefig(save_path + '_confusion_matrix.png')
    return cm

def evaluate(network, save_path, test_episodes, test_features, n_way, onehot=True):
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
        label_to_target = {target.item(): label.item() for label, target in zip(support['label'], support['target'])}
        # get the embeddings
        logits = network(query, support)
        #print('logits before: ',logits)
        if not onehot:
            logits = torch.round(logits)
        # compute the accuracy
        acc = metric(logits, query["target"].to(device))
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


cropped_scans = np.load('data/defSTM.npy')
labels = np.load('data/defSTM_labels.npy')

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

x_all = torch.tensor(cropped_scans).float().to(device)
y_all = torch.tensor(labels).float().to(device)

# change labels to being one hot encoded
OHE = OneHotEncoder()
labels = OHE.fit_transform(np.expand_dims(y_all, axis=1)).toarray()

# convert these to pytorch tensors
x = torch.tensor(x_all)
y = torch.tensor(labels)

train_val_features = ['single dangling bond' , 'As A' , 'siloxane', 'C defect', 'g1' , 'As A inv' ,
                  'siloxane inv', 'g1 inv', 'C defect inv', 'TiO2_vacancy', 'h1', 'm1 inv',
                  'h2', 'single dangling bond inv', 'm1','TiO2_hydroxyl inv', 'h2 inv', 't1 inv', ]
test_features = ['double dangling bond', 'As B', 'single dihydride', 'single DV on Si(001)']


def simple_shot_training_eval(train_val_features, test_features, n_support):
    batch_size = 50
    # Create the datasets
    train_val_data = STM_bright_features(x, y, train_val_features, res=40)
    train_data, val_data = train_val_data.split(test_size=0.2, random_state=42)
    test_data = STM_bright_features(x, y, test_features res=40)
    episodic_test_data_shot = EpisodeDataset(test_data, n_way=4, n_support=n_support, n_query=15, n_episodes=100)

    # Create the sampler and data loader
    data_loader_train = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    data_loader_val = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    model = NeuralNetwork(channels=2, crop_size=40, n_outputs=4, fc_layers=2, fc_nodes=100, dropout=0.2).to(device)

    # Compute the sample weights
    # Get the labels from the dataset
    # Note: train_data.labels is a 1D tensor of indices after refine_images
    labels_tensor = train_data.labels 
    # Count samples per class
    class_sample_count = torch.bincount(labels_tensor.long())
    # Calculate weights (inverse frequency)
    weight = 1.0 / class_sample_count.float()
    samples_weight = weight.numpy()
    print("Class counts:", class_sample_count)
    print("Weights:", samples_weight)

    # Define the loss functions, and optimizer
    criterion_weighted = nn.CrossEntropyLoss(weight = torch.tensor(samples_weight)) # weighted loss function
    criterion = nn.CrossEntropyLoss() # not weighted loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size= 20)#, gamma=0.3) # lr=0.5*lr after every 5 epochs
    number_epochs = 20

    substrate = 'Si'
    n_way = 4
    n_query = 15

    train(model, data_loader_train, data_loader_val, criterion_weighted, 15, f'SimpleShotembeddor{substrate}.pth')

    simple_shot = SimpleShot(model)

    # test the simple shot model
    # we need to do it over 100 eps
    simple_shot, metric, total_acc, accuracies, std_dev, confidenceInt, precisions, recalls, average_precision, average_recall = evaluate(simple_shot, f'SimpleShotembeddor_{substrate}_({n_way},{n_query})_eval', episodic_test_data_shot, test_features, n_way=4, onehot=True)

    return simple_shot, metric, total_acc, accuracies, std_dev, confidenceInt, precisions, recalls, average_precision, average_recall

